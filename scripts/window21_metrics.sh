#!/usr/bin/env bash
# [МЕТРИКИ 30.10] Окно 21 -- метрики при конкурентности c=1/4/8 (владелец: «и сними метрики c=1,c=4,c=8, префилл и
# генерация и LLM и clef»; разрешение остановить боевой -- «добивай этап2 сразу и можно будет остановить боевой»).
# Правок кода нет. Два A/B подряд на 127.0.0.1:18493, только GPU0-1:
#   1) prod -- профиль боевого как есть (KV по умолчанию 10.52 ГиБ, бюджет 2048:0, шаг с Clef eager);
#   2) gr   -- этап 2: полный граф смешанного шага (fullmix, gr=2), KV 9.0 ГиБ, бюджет 2000:0, GEMM v3, CE-обмен в графе.
# В каждом: scripts/bench_conc.py (LLM генерация/префилл, Clef, LLM+Clef при c=1/4/8). Боевой поднимается всегда (trap).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
KV_BYTES="${KV_BYTES:-9663676416}"
PORT="${AB_PORT:-18493}"
URL=http://127.0.0.1:$PORT
RESTORE="${RESTORE:-1}"
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
    "${envs[@]}" \
    "$REPO/scripts/serve_candidate_ab.sh"
  if ! wait_health $URL 3000 "$AB"; then
    log "A/B $name не поднялся"; tail -60 "$OUT/server-$name.log" >> "$OUT/window.log"; return 1
  fi
  log "A/B $name готов, прогрев 30 с"; sleep 30
}
gpu_summary() { grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
bench() {   # имя
  "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-$1.json" "$1" >> "$OUT/window.log" 2>&1
  log "метрики $1: код $?"
  sleep 10; gpu_summary "$1"
  grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-$1.log" | sed "s/^/ошибок $1: /" | tee -a "$OUT/window.log"
}
trap restore EXIT

printf 2 > "$OUT/gr"
log "окно 21: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 1) профиль боевого
echo "2048:0" > "$OUT/budget"
if start_ab prod; then gpu_summary prod; bench prod; fi
stop_ab

# 2) этап 2: полный граф смешанного шага
echo "2000:0" > "$OUT/budget"
start_ab gr FA2SM70_CLEF_GRAPH=1 FA2SM70_CLEF_GRAPH_MAX=2112 FA2SM70_CLEF_GRAPH_FILE="$OUT/gr" \
  FA2SM70_FULL_MIX=1 FA2SM70_FULL_MIX_FILE="$OUT/gr" FA2SM70_FULL_MIX_SIZES="$FM_SIZES" FA2SM70_CLEF_HEAD_ASYNC_GRAPH=1 \
  FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1 \
  FA2SM80_W8_G3=1 FA2SM70_EXCH_CE_GRAPH=1 FA2SM70_KV_BYTES=$KV_BYTES || exit 1
gpu_summary gr
bench gr
grep -a "fa2_sm70 GC\]" "$OUT/server-gr.log" | tail -5 >> "$OUT/window.log"
