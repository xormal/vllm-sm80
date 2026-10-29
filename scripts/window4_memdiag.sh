#!/usr/bin/env bash
# [ПАМЯТЬ/ПЛАВНОСТЬ 29.10] Окно 4 (разрешено владельцем): диагностика памяти -- снимок аллокатора со стеками (TP0),
# точное A/B графов тела MTP на одинаковых промптах (переключатель-файл), заход с FA2SM70_MTP_GRAPH=0. Два захода A/B на 127.0.0.1:8086 с профилем и кэшами боевого; боевой ВСЕГДА поднимается (trap).
# Только GPU0-1. Запуск: systemd-run --user --unit=kandidat-window2-... scripts/window4_memdiag.sh OUTDIR
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
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
  log "восстановление: гашу A/B ($AB), поднимаю $LIVE"
  stop_ab
  systemctl --user reset-failed "$LIVE" 2>/dev/null
  systemctl --user start "$LIVE"
  systemctl --user start "$OBS"
  if wait_health http://127.0.0.1:8085 1200; then
    log "боевой $LIVE снова отвечает на 8085: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
  else
    log "ВНИМАНИЕ: боевой не ответил за 1200 с: $(systemctl --user show $LIVE -p ActiveState -p SubState | tr '\n' ' ')"
  fi
}
start_ab() {   # имя-захода графы-тела(0/1) [история-памяти]
  AB=kandidat-ab-$1-$(date +%H%M).service
  log "старт A/B $AB (FA2SM70_MTP_GRAPH=$2)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM70_MTP_GRAPH="$2" --setenv=FA2SM70_MTP_GRAPH_FILE="$OUT/mtpgraph" \
    --setenv=FA2SM80_MEMTRACE=20 ${3:+--setenv=FA2SM80_MEMHIST=$3} \
    "$REPO/scripts/serve_candidate_ab.sh"
  if ! wait_health http://127.0.0.1:8086 900 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
}
cli() { "$REPO/.venv/bin/python" "$REPO/scripts/acceptance_mtpgraphs.py" "$@" --url http://127.0.0.1:8086 >> "$OUT/window.log" 2>&1; log "клиент $1 ${*: -1}: код $?"; }
gpu_summary() { grep -a "MEMTRACE\] GPU" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
trap restore EXIT

echo 0 > "$OUT/budget"; printf 1 > "$OUT/mtpgraph"
log "окно 4: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

start_ab g1 1 "$OUT/memhist"
for tag in on-a off-a on-b off-b; do
  case $tag in on-*) printf 1 > "$OUT/mtpgraph" ;; *) printf 0 > "$OUT/mtpgraph" ;; esac
  sleep 3
  cli sweep "$OUT/sweep" --fixed --concurrency 1,4,8 --contexts 2048,34816 --tag "g1-$tag"
done
printf 1 > "$OUT/mtpgraph"
gpu_summary g1
touch "$OUT/memhist.start"; sleep 25
"$REPO/.venv/bin/python" "$REPO/scripts/acceptance_livemix.py" "$OUT/livemix-g1" --url http://127.0.0.1:8086 \
  --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" --budgets 0 --rounds 2 --phase-seconds 120 >> "$OUT/window.log" 2>&1
log "живая смесь g1: код $?"
touch "$OUT/memhist.trigger"; sleep 30
gpu_summary g1
ls -la "$OUT"/memhist.*.pickle 2>/dev/null | tee -a "$OUT/window.log"
stop_ab

start_ab g0 0
"$REPO/.venv/bin/python" "$REPO/scripts/acceptance_livemix.py" "$OUT/livemix-g0" --url http://127.0.0.1:8086 \
  --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" --budgets 0 --rounds 2 --phase-seconds 120 >> "$OUT/window.log" 2>&1
log "живая смесь g0: код $?"
sleep 25; gpu_summary g0
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
