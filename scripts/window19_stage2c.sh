#!/usr/bin/env bash
# [ОДИН ГРАФ 30.10] Окно 19 -- третья попытка этапа 2 (разрешение владельца: «добивай этап2 сразу и можно будет
# остановить боевой»). Окна 17-18: пути полного графа на GPU работают (прогрев и 42 захвата без исключений), но память:
# ключ (T, D) -- 81 граф по ~36 МиБ драйвера (~10 тыс. узлов на граф), а графы от 2048 строк включают конвейеры блока/
# строк с побочными потоками и record_stream (память не переиспользуется внутри захвата). Здесь:
#   * граф НЕ зависит от состава шага: ключ -- только T (8 корзин до 2040), спекулятивные строки дополнены до Dmax строками
#     нулевой длины с PAD, обычные -- все R строк в абсолютных координатах токенов, маска спекуляции -- на карте;
#   * корзины < 2048 (конвейеры не включаются), бюджет prefill 2000 (шаг 2000 + 8x5 = 2040);
#  1) A/B: сверка Clef gr 0/2, отпечаток LLM без нагрузки и под нагрузкой Clef, dlinnye, профили;
#  2) насыщенная смесь: бюджет 2048 eager (как в бою) и бюджет 2000 фазами gr 0 / 2 (полный граф).
# Боевой ВСЕГДА поднимается (trap) при RESTORE=1, затем 5 минут живой нагрузки. Только GPU0-1.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
PS="$REPO/sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
BENCH="$(dirname "$REPO")/sm80_work/bench"
REF_CLEF="$(dirname "$REPO")/docs/reports/postanovka-2026-10-19/clef.json"
WT="${WT:?рабочее дерево с правками}"
BASE_TAR="${BASE_TAR:?база правленых файлов (tar)}"
FILES="${FILES:-sm70/integrations/vllm_sm70/fa2sm70_fullmix.py vllm/v1/cudagraph_dispatcher.py vllm/v1/worker/gpu_model_runner.py vllm/v1/attention/backends/gdn_attn.py vllm/model_executor/models/qwen3_next.py vllm/v1/attention/backends/fa2_sm70.py}"
NEW="${NEW:-}"
PORT="${AB_PORT:-18493}"           # не 8085/8086: агенты-потребители не должны цепляться к A/B
URL=http://127.0.0.1:$PORT
RESTORE="${RESTORE:-1}"
# кусочные графы: декод (как в бою) + смешанные шаги до бюджета 2048 + 8 чатов x 5 токенов; полные -- корзины fullmix
FM_SIZES="${FM_SIZES:-256,512,768,1024,1280,1536,1792,2040}"
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/window.log"; }
gpus_free() {
  for c in 0 1; do
    u=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    [ "${u:-99999}" -lt 500 ] || return 1
  done
}
wait_health() {   # url timeout [unit]
  local t=0
  until curl -sf -m 5 "$1/health" >/dev/null; do
    sleep 5; t=$((t + 5)); [ $t -ge "$2" ] && return 1
    if [ -n "${3:-}" ] && [ $t -ge 30 ] && ! systemctl --user is-active -q "$3"; then return 1; fi
  done
}
stop_ab() {
  [ -n "$AB" ] && systemctl --user stop "$AB" 2>/dev/null
  AB=""
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
}
restore() {
  stop_ab
  if [ "$RESTORE" != "1" ]; then log "A/B погашен; боевой НЕ поднимаю (RESTORE=0)"; return; fi
  log "восстановление: поднимаю $LIVE"
  systemctl --user reset-failed "$LIVE" 2>/dev/null
  systemctl --user start "$LIVE"
  systemctl --user start "$OBS"
  local LL; LL="$(dirname "$REPO")/state/acceptance/live-20261029/server.log"
  local S0; S0=$(wc -l < "$LL")
  if wait_health http://127.0.0.1:8085 1500; then
    log "боевой $LIVE снова отвечает на 8085: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
    sleep 300
    log "боевой через 5 мин: падений $(tail -n +$S0 "$LL" | grep -ac 'EngineCore encountered a fatal error'), 500: $(tail -n +$S0 "$LL" | grep -ac '500 Internal Server Error'), 200: $(tail -n +$S0 "$LL" | grep -ac '" 200 OK'), NRestarts $(systemctl --user show $LIVE -p NRestarts --value)"
  else
    log "ВНИМАНИЕ: боевой не ответил за 1500 с: $(systemctl --user show $LIVE -p ActiveState -p SubState | tr '\n' ' ')"
  fi
}
start_ab() {   # имя-захода  доп-окружение...
  local name=$1; shift
  AB=kandidat-ab-$name-$(date +%H%M).service
  log "старт A/B $AB ($*)"
  local envs=()
  for e in "$@"; do envs+=(--setenv="$e"); done
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$name.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$name.jsonl" \
    --setenv=FA2SM70_CLEF_HEAD_ASYNC=1 \
    --setenv=FA2SM70_NOSYNC_ACC2=1 --setenv=FA2SM70_MEM_TRIM_MODE=off \
    --setenv=FA2SM80_MEMTRACE=20 --setenv=AB_PORT="$PORT" --setenv=FA2SM80_GC_LOG_MS=200 \
    --setenv=PROFILE_DIR="$OUT/profile" --setenv=PROFILE_STACK=false \
    "${envs[@]}" \
    "$REPO/scripts/serve_candidate_ab.sh"
  if ! wait_health $URL 3000 "$AB"; then
    log "A/B $name не поднялся"; tail -60 "$OUT/server-$name.log" >> "$OUT/window.log"; return 1
  fi
  log "A/B $name готов, прогрев 30 с"; sleep 30
}
gpu_summary() { grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
warm() {
  "$PYV" - "$URL" <<'PY'
import json, random, sys, urllib.request
for _ in range(3):
    ids = [random.randint(1000, 60000) for _ in range(900)]
    req = urllib.request.Request(sys.argv[1] + "/v1/completions", json.dumps({
        "model": "Qwen3.8-27B-ABLITERATED", "prompt": ids, "max_tokens": 1}).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
PY
}
sw() { printf "$2" > "$OUT/$1"; sleep 2; warm; log "$1: $2"; }
lm() {   # имя аргументы...
  "$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/$1" --url $URL \
    --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" "${@:2}" >> "$OUT/window.log" 2>&1
  log "живая смесь $1: код $?"
}
clef_chk() {   # метка
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$1.json" > "$OUT/clef-$1.log" 2>&1
  log "Clef [$1] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$1.json" 2>&1 | tail -1)"
}
llm_chk() {    # метка [эталон]
  (cd "$BENCH" && URL=$URL OUT="$OUT/q-$1.json" REF="${2:-}" "$PYV" quality_ref.py) > "$OUT/q-$1.log" 2>&1
  log "отпечаток LLM [$1]: $(grep -E '^L=' "$OUT/q-$1.log" | cut -c1-90 | tr '\n' ' ')"
}
trap restore EXIT

printf 0 > "$OUT/gr"; echo "2000:0" > "$OUT/budget"
log "окно 19: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 0) правки -- только сейчас: боевой остановлен, ни один сервер их не импортирует
for f in $FILES; do
  if ! tar -xOf "$BASE_TAR" "$f" | cmp -s - "$REPO/$f"; then
    log "ОТКАЗ: $f в основном дереве отличается от базы -- правки не применены"; exit 1
  fi
done
for f in $NEW; do
  if [ -e "$REPO/$f" ] && ! cmp -s "$WT/$f" "$REPO/$f"; then log "ОТКАЗ: $f уже есть в основном дереве и другой"; exit 1; fi
done
for f in $FILES $NEW; do cp -a "$WT/$f" "$REPO/$f"; done
log "правки применены ($(echo $FILES $NEW | wc -w) файлов)"

G3=1
CEG=1

# 3) A/B №2: шаг с Clef -- 0 eager, 1 кусочный граф has_clef, 2 полный граф (fullmix)
ge() {   # размеры полных графов
  echo FA2SM70_CLEF_GRAPH=1 FA2SM70_CLEF_GRAPH_MAX=2112 FA2SM70_CLEF_GRAPH_FILE="$OUT/gr" \
       FA2SM70_FULL_MIX=1 FA2SM70_FULL_MIX_FILE="$OUT/gr" FA2SM70_FULL_MIX_SIZES="$1" FA2SM70_CLEF_HEAD_ASYNC_GRAPH=1 \
       FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1 \
       FA2SM80_W8_G3=$G3 FA2SM70_EXCH_CE_GRAPH=$CEG
}
# shellcheck disable=SC2046
start_ab gr $(ge "$FM_SIZES") || exit 1
gpu_summary gr
grep -aE "FULLMIX\]|Graph capturing finished|графобезопасная|в графе -- копировщиком" "$OUT/server-gr.log" | grep -a Worker_TP0 | cut -c1-220 | tail -8 >> "$OUT/window.log"
for m in 0 2; do sw gr "$m"; clef_chk "gr_$m"; done
log "Clef gr 2 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-gr_0.json" "$OUT/clef-gr_2.json" 2>&1 | tail -1)"
sw gr 0; llm_chk gr_0
sw gr 2; llm_chk gr_2 "$OUT/q-gr_0.json"
# отпечаток LLM ПОД нагрузкой Clef: смешанные шаги со строками чатов (D > 0)
for m in 0 2; do
  sw gr "$m"
  ( for r in 1 2 3; do "$PYV" "$PS" clef "$PORT" "$OUT/clef-load-$m-$r.json" > /dev/null 2>&1; done ) &
  LOADPID=$!
  sleep 5
  llm_chk "gr_load_$m" "$OUT/q-gr_0.json"
  wait $LOADPID 2>/dev/null
done
sw gr 2
"$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye-gr2.json" 34816 > "$OUT/dlinnye-gr2.log" 2>&1
log "dlinnye [gr 2]: $(grep ИТОГ "$OUT/dlinnye-gr2.log")"
grep -aE "Traceback|Error|fullmix" "$OUT/server-gr.log" | grep -av "INFO" | head -8 >> "$OUT/window.log"
for m in 0 2; do
  sw gr "$m"
  "$PYV" "$REPO/scripts/acceptance_mtpgraphs.py" profile "$OUT/client-prof-gr$m" --url $URL --seconds 3 >> "$OUT/window.log" 2>&1
  mkdir -p "$OUT/profile-gr$m"; mv "$OUT/profile/"*.gz "$OUT/profile-gr$m/" 2>/dev/null
done
echo "2048:0" > "$OUT/budget"; sw gr 0
lm "sat-gr0-b2048" --clef-concurrency 6 --rounds 1 --phase-seconds 240
echo "2000:0" > "$OUT/budget"; sw gr 0
lm "sat-gr-b2000" --clef-concurrency 6 --phase-file "$OUT/gr" --phase-values "0,2" --rounds 2 --phase-seconds 120
echo "2048:0" > "$OUT/budget"
printf 0 > "$OUT/gr"
sleep 20; gpu_summary gr
grep -a "fa2_sm70 GC\]" "$OUT/server-gr.log" | tail -10 >> "$OUT/window.log"
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-gr.log" | sed 's/^/ошибок A\/B №2: /' | tee -a "$OUT/window.log"
