#!/usr/bin/env bash
# [ОДИН ГРАФ 30.10] Окно 16 -- ТОЛЬКО С ОТДЕЛЬНОГО РАЗРЕШЕНИЯ ВЛАДЕЛЬЦА. Первый этап единого графа: шаги с Clef до 2112
# токенов (бюджет 2048 + декод 8 чатов) -- кусочными графами семейства has_clef вместо force_eager. Правки уже в основном
# дереве (окно 15); здесь только A/B:
#   * поправка Clef в графе -- по списку строк (целый шаг) и по маске строк (куски конвейера блока, delta_graph_range);
#   * обмен TP в захвате -- копировщиком с номером на устройстве (FA2SM70_EXCH_CE_GRAPH=1), а не NCCL-сбором;
#   * GEMM v3 (FA2SM80_W8_G3=1, без распаковки весов в fp16), пакетные слои att (varlen внимания и GDN) -- включены
#     в обоих плечах; переключатель $OUT/gr: 0 -- шаг с Clef eager (как в бою), 1 -- кусочный граф has_clef.
#  1) A/B на 127.0.0.1:$AB_PORT; сверка Clef / LLM / dlinnye: gr 1 против 0;
#  2) профиль без стеков при 0 и 1; живая смесь (насыщенная) фазами 0/1; пулы графов и повторы выделения.
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
PORT="${AB_PORT:-18493}"           # не 8085/8086: агенты-потребители не должны цепляться к A/B
URL=http://127.0.0.1:$PORT
RESTORE="${RESTORE:-1}"
# размеры графов: декод (как в бою) + смешанные шаги до бюджета 2048 + 8 чатов x 5 токенов
CG_SIZES="${CG_SIZES:-1,2,4,8,12,16,20,24,28,32,36,40,64,96,128,192,256,320,384,448,512,640,768,1024,1280,1536,1792,2048,2112}"
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
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
}
restore() {
  stop_ab
  if [ "$RESTORE" != "1" ]; then log "A/B ($AB) погашен; боевой НЕ поднимаю (RESTORE=0)"; return; fi
  log "восстановление: A/B ($AB) погашен, поднимаю $LIVE"
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
start_ab() {   # имя-захода
  AB=kandidat-ab-$1-$(date +%H%M).service
  log "старт A/B $AB (конфигурация боевого + v3 + att; графы шагов с Clef -- переключатель-файл $OUT/gr)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM70_CLEF_HEAD_ASYNC=1 --setenv=FA2SM70_CLEF_HEAD_ASYNC_GRAPH=1 \
    --setenv=FA2SM70_NOSYNC_ACC2=1 --setenv=FA2SM70_MEM_TRIM_MODE=off \
    --setenv=FA2SM80_W8_G3=1 --setenv=FA2SM70_EXCH_CE_GRAPH=1 \
    --setenv=FA2SM70_CLEF_GRAPH=1 --setenv=FA2SM70_CLEF_GRAPH_MAX=2112 --setenv=FA2SM70_CLEF_GRAPH_FILE="$OUT/gr" \
    --setenv=FA2SM80_CG_SIZES="$CG_SIZES" \
    --setenv=FA2SM80_GDN_DEVMETA=1 --setenv=FA2SM80_GDN_ALL_VARLEN=1 --setenv=FA2SM80_ATTN_VARLEN=1 \
    --setenv=FA2SM80_MIX_DEC_UNIFORM=1 \
    --setenv=FA2SM80_MEMTRACE=20 --setenv=AB_PORT="$PORT" --setenv=FA2SM80_GC_LOG_MS=200 \
    --setenv=PROFILE_DIR="$OUT/profile" --setenv=PROFILE_STACK=false \
    "$REPO/scripts/serve_candidate_ab.sh"
  # первый старт: компиляция под новые размеры графов и захват двух семейств (минуты)
  if ! wait_health $URL 2400 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
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
ns() { printf "$1" > "$OUT/gr"; sleep 2; warm; log "gr: $1"; }
lm() {   # имя аргументы...
  "$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/$1" --url $URL \
    --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" "${@:2}" >> "$OUT/window.log" 2>&1
  log "живая смесь $1: код $?"
}
trap restore EXIT

printf 0 > "$OUT/gr"; echo "2048:0" > "$OUT/budget"
log "окно 16: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 2) A/B: сверка 1 против 0
start_ab gr
grep -aE "графобезопасная|в графе -- копировщиком|v3 K=.*кэш" "$OUT/server-gr.log" | grep -a Worker_TP0 | cut -c1-220 >> "$OUT/window.log"
gpu_summary gr
for m in 0 1 02; do
  ns "${m:0:1}"
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$m.json" > "$OUT/clef-$m.log" 2>&1
  log "Clef [gr $m] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$m.json" 2>&1 | tail -1)"
  ref=""; [ "$m" != 0 ] && ref="$OUT/q-0.json"
  (cd "$BENCH" && URL=$URL OUT="$OUT/q-$m.json" REF="$ref" "$PYV" quality_ref.py) > "$OUT/q-$m.log" 2>&1
  log "отпечаток LLM [gr $m]: $(grep -E '^L=' "$OUT/q-$m.log" | cut -c1-90 | tr '\n' ' ')"
done
log "Clef gr 1 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-1.json" 2>&1 | tail -1)"
log "Clef gr 0 повтор против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-02.json" 2>&1 | tail -1)"
ns 1
"$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye-1.json" 34816 > "$OUT/dlinnye-1.log" 2>&1
log "dlinnye [gr 1]: $(grep ИТОГ "$OUT/dlinnye-1.log")"
grep -aE "Traceback|Error" "$OUT/server-gr.log" | grep -av "INFO" | head -5 >> "$OUT/window.log"

# 3) профиль без стеков при 0 и 1
for m in 0 1; do
  ns "$m"
  "$PYV" "$REPO/scripts/acceptance_mtpgraphs.py" profile "$OUT/client-prof$m" --url $URL --seconds 3 >> "$OUT/window.log" 2>&1
  log "профиль gr $m: код $?"
  mkdir -p "$OUT/profile-gr$m"; mv "$OUT/profile/"*.gz "$OUT/profile-gr$m/" 2>/dev/null
done

# 4) живая смесь (насыщенная) фазами gr 0/1
ns 0
lm "sat-gr" --clef-concurrency 6 --phase-file "$OUT/gr" --phase-values "0,1" --rounds 3 --phase-seconds 120
printf 0 > "$OUT/gr"
sleep 25; gpu_summary gr
grep -a "граф" "$OUT/server-gr.log" | grep -aE "Traceback|ошибк|fail|не удал" | head -5 >> "$OUT/window.log"
grep -a "fa2_sm70 GC\]" "$OUT/server-gr.log" | tail -20 >> "$OUT/window.log"
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
