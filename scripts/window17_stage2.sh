#!/usr/bin/env bash
# [ОДИН ГРАФ + GEMM 30.10] Окно 17 -- ПО РАЗРЕШЕНИЮ ВЛАДЕЛЬЦА («добивай этап2 сразу и можно будет остановить боевой»).
# Всё, что подготовлено после окна 14b, под флагами (без них путь прежний):
#   * GEMM v3 (w8a16_sm80_gemm3: x в порядке фрагментов, варп 64x64; FA2SM80_W8_G3 / _FILE), конвейер блока без распаковки;
#   * этап 1 графа: поправка Clef по маске строк в графе (delta_graph_range), конвейеры в захвате has_clef, обмен
#     копировщиком в графе (FA2SM70_EXCH_CE_GRAPH), асинхронная голова при графах (FA2SM70_CLEF_HEAD_ASYNC_GRAPH);
#   * этап 2: ПОЛНЫЙ граф смешанного шага с Clef (fa2sm70_fullmix, ключ (T, D), FA2SM70_FULL_MIX / _FILE), выравнивание
#     кусков префилла по границе блока mamba; журнал долгих сборок мусора (FA2SM80_GC_LOG_MS).
#  0) правки из рабочего дерева $WT -> основное дерево ТОЛЬКО при остановленном боевом (сверка с базой $BASE_TAR),
#     сборка w8a16_sm80/clef/exch через candidate_build.py;
#  1) стенды: v3 на GPU0 (побитно против прямого ядра, срезы, без синхронизаций, граф, время); обмен в графе (2 карты);
#  2) A/B №1 -- v3 0/1 на конфигурации боевого: сверка Clef/LLM/dlinnye, профиль, насыщенная смесь;
#  3) A/B №2 -- шаг с Clef: 0 eager (как в бою), 1 кусочный граф has_clef, 2 полный граф (один файл $OUT/gr);
#     v3 (если стенд прошёл), att, обмен в графе -- во всех фазах; сверка Clef по фазам, отпечаток LLM без нагрузки и
#     ПОД нагрузкой Clef (смешанные шаги с чатами), dlinnye, профили 0/2, насыщенная смесь фазами 0/1/2.
# Боевой ВСЕГДА поднимается (trap) при RESTORE=1, затем 5 минут живой нагрузки без смертей движка. Только GPU0-1.
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
FILES="${FILES:-sm70/fa2_sm70/csrc/clef.cu sm70/fa2_sm70/csrc/exch_i12.cu sm70/fa2_sm70/csrc/w8a16_sm80.cu sm70/integrations/vllm_sm70/fa2sm70_clef.py sm70/integrations/vllm_sm70/fa2sm70_exch_i12.py sm70/integrations/vllm_sm70/fa2sm80_blockpipe.py sm70/integrations/vllm_sm70/fa2sm80_rowpipe.py sm70/integrations/vllm_sm70/fa2sm80_w8.py sm70/integrations/vllm_sm70/sitecustomize.py vllm/forward_context.py vllm/model_executor/models/qwen3_next.py vllm/v1/attention/backends/fa2_sm70.py vllm/v1/attention/backends/gdn_attn.py vllm/v1/core/sched/scheduler.py vllm/v1/cudagraph_dispatcher.py vllm/v1/worker/gpu_model_runner.py}"
NEW="${NEW:-sm70/fa2_sm70/csrc/w8a16_sm80_g3.cuh sm70/integrations/vllm_sm70/fa2sm70_fullmix.py}"
PORT="${AB_PORT:-18493}"           # не 8085/8086: агенты-потребители не должны цепляться к A/B
URL=http://127.0.0.1:$PORT
RESTORE="${RESTORE:-1}"
# кусочные графы: декод (как в бою) + смешанные шаги до бюджета 2048 + 8 чатов x 5 токенов; полные -- корзины fullmix
CG_SIZES="${CG_SIZES:-1,2,4,8,12,16,20,24,28,32,36,40,64,128,256,384,512,768,1024,1536,2048,2112}"
FM_SIZES="${FM_SIZES:-256,512,768,1024,1280,1536,1792,2048,2112}"
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

printf 0 > "$OUT/g3"; printf 0 > "$OUT/gr"; echo "2048:0" > "$OUT/budget"
log "окно 17: стоп $OBS и $LIVE"
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
"$REPO/scripts/candidate_build.py" -- "$PYV" -c "import fa2_sm70._ext as E; print(E.w8a16_sm80_ext().__file__, E.clef_ext().__file__, E.exch_ext().__file__)" \
  > "$OUT/build.log" 2>&1
rc=$?
log "сборка w8a16_sm80 + clef + exch: код $rc"
if [ $rc -ne 0 ]; then tail -30 "$OUT/build.log" >> "$OUT/window.log"; exit 1; fi

# 1) стенды
systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
  bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0; exec "$2" "$1/scripts/w8_g3_test.py" "$3"' \
  _ "$REPO" "$PYV" "$OUT/g3_test.json" > "$OUT/g3_test.log" 2>&1
rc=$?
log "стенд v3: код $rc"
grep -E "^(побитно|НЕ ПОБИТНО|срез|без синхронизаций|граф|GEMM-часть)" "$OUT/g3_test.log" | cut -c1-300 >> "$OUT/window.log"
grep -E "^время" "$OUT/g3_test.log" | "$PYV" -c "
import json, sys
for l in sys.stdin:
    r = json.loads(l[l.index('{'):])
    print(f\"{r['форма']:8s} M={r['M']:5d}  v3 {r['v3_TF']:6.1f} TF {r['v3_cfg']}  прямой {r['прямой_TF']:6.1f}  деквант+cublas {r['деквант+cublas_TF']:6.1f}  cublas {r['cublas_TF']:6.1f}  упаковка {r['упаковка_мкс']:.0f} мкс\")
" >> "$OUT/window.log" 2>&1
G3=0
if [ $rc -eq 0 ] && ! grep -q "^НЕ ПОБИТНО" "$OUT/g3_test.log"; then
  выигрыш=$("$PYV" -c "
import json; r = json.load(open('$OUT/g3_test.json'))
print(sum(1 for x in r['время'] if x['v3_мкс'] < 0.97 * min(x['прямой_мкс'], x['деквант_мкс'] + x['cublas_мкс'])))")
  log "v3 быстрее нынешнего маршрута в $выигрыш точках из $(grep -c '^время' "$OUT/g3_test.log")"
  [ "${выигрыш:-0}" -gt 0 ] && G3=1
else
  log "стенд v3 не пройден -- v3 дальше выключен"
fi
for i in $(seq 1 30); do gpus_free && break; sleep 2; done
systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
  bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0,1 TORCH_CUDA_ARCH_LIST=8.0; exec timeout 900 "$2" "$1/scripts/exch_ce_graph_test.py" "$3"' \
  _ "$REPO" "$PYV" "$OUT/exch_graph.json" > "$OUT/exch_graph.log" 2>&1
log "стенд обмена в графе: код $? $(grep -a '^ранг' "$OUT/exch_graph.log" | cut -c1-260 | tr '\n' ' ')"
CEG=0
grep -aq '"плохих": 0, "eager_плохих": 0' "$OUT/exch_graph.log" && grep -aq '"срыв": \[0, 0\]' "$OUT/exch_graph.log" && CEG=1
log "обмен копировщиком в графе: $([ $CEG = 1 ] && echo годен || echo НЕ годен -- в графе NCCL)"
for i in $(seq 1 30); do gpus_free && break; sleep 2; done

# 2) A/B №1: v3 0/1 (конфигурация боевого)
if [ $G3 = 1 ]; then
  if start_ab g3 FA2SM80_W8_G3_FILE="$OUT/g3"; then
    grep -a "v3 K=" "$OUT/server-g3.log" | grep -a Worker_TP0 | sed 's/^.*\] v3/v3/' | cut -c1-200 >> "$OUT/window.log"
    for m in 0 1 02; do
      sw g3 "${m:0:1}"; clef_chk "g3_$m"
      ref=""; [ "$m" != 0 ] && ref="$OUT/q-g3_0.json"
      llm_chk "g3_$m" "$ref"
    done
    log "Clef g3 1 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-g3_0.json" "$OUT/clef-g3_1.json" 2>&1 | tail -1)"
    sw g3 1
    "$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye-g3.json" 34816 > "$OUT/dlinnye-g3.log" 2>&1
    log "dlinnye [g3 1]: $(grep ИТОГ "$OUT/dlinnye-g3.log")"
    for m in 0 1; do
      sw g3 "$m"
      "$PYV" "$REPO/scripts/acceptance_mtpgraphs.py" profile "$OUT/client-prof-g3$m" --url $URL --seconds 3 >> "$OUT/window.log" 2>&1
      mkdir -p "$OUT/profile-g3$m"; mv "$OUT/profile/"*.gz "$OUT/profile-g3$m/" 2>/dev/null
    done
    sw g3 0
    lm "sat-g3" --clef-concurrency 6 --phase-file "$OUT/g3" --phase-values "0,1" --rounds 3 --phase-seconds 120
    sleep 20; gpu_summary g3
    grep -a "fa2_sm70 GC\]" "$OUT/server-g3.log" | tail -10 >> "$OUT/window.log"
    grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-g3.log" | sed 's/^/ошибок A\/B №1: /' | tee -a "$OUT/window.log"
  fi
  stop_ab
fi

# 3) A/B №2: шаг с Clef -- 0 eager, 1 кусочный граф has_clef, 2 полный граф (fullmix)
GE=(FA2SM70_CLEF_GRAPH=1 FA2SM70_CLEF_GRAPH_MAX=2112 FA2SM70_CLEF_GRAPH_FILE="$OUT/gr"
    FA2SM70_FULL_MIX=1 FA2SM70_FULL_MIX_FILE="$OUT/gr" FA2SM70_FULL_MIX_SIZES="$FM_SIZES"
    FA2SM80_CG_SIZES="$CG_SIZES" FA2SM70_CLEF_HEAD_ASYNC_GRAPH=1
    FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1
    FA2SM80_W8_G3=$G3 FA2SM70_EXCH_CE_GRAPH=$CEG)
if ! start_ab gr "${GE[@]}"; then exit 1; fi
gpu_summary gr
grep -aE "Graph capturing finished|графобезопасная|в графе -- копировщиком|Capturing CUDA graphs" "$OUT/server-gr.log" | grep -av "it/s" | cut -c1-220 | tail -8 >> "$OUT/window.log"
for m in 0 1 2; do sw gr "$m"; clef_chk "gr_$m"; done
log "Clef gr 2 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-gr_0.json" "$OUT/clef-gr_2.json" 2>&1 | tail -1)"
log "Clef gr 1 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-gr_0.json" "$OUT/clef-gr_1.json" 2>&1 | tail -1)"
sw gr 0; llm_chk gr_0
sw gr 2; llm_chk gr_2 "$OUT/q-gr_0.json"
# отпечаток LLM ПОД нагрузкой Clef: смешанные шаги со строками чатов (D > 0)
for m in 0 2; do
  sw gr "$m"
  ( for r in 1 2 3; do "$PYV" "$PS" clef "$PORT" "$OUT/clef-load-$m-$r.json" > /dev/null 2>&1; done ) &
  нагр=$!
  sleep 5
  llm_chk "gr_load_$m" "$OUT/q-gr_0.json"
  kill $нагр 2>/dev/null; wait $нагр 2>/dev/null
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
sw gr 0
lm "sat-gr" --clef-concurrency 6 --phase-file "$OUT/gr" --phase-values "0,1,2" --rounds 2 --phase-seconds 120
printf 0 > "$OUT/gr"
sleep 20; gpu_summary gr
grep -a "fa2_sm70 GC\]" "$OUT/server-gr.log" | tail -10 >> "$OUT/window.log"
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-gr.log" | sed 's/^/ошибок A\/B №2: /' | tee -a "$OUT/window.log"
