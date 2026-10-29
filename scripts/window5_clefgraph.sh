#!/usr/bin/env bash
# [ПАМЯТЬ/ПЛАВНОСТЬ 29.10] Окно 5 (разрешено владельцем): графы шагов с Clef (has_clef, графобезопасная поправка) и
# выход сведений побочного потока из пула основного. Сверка ответов Clef и тест ровности декода при полной очереди Clef. Два захода A/B на 127.0.0.1:8086 с профилем и кэшами боевого; боевой ВСЕГДА поднимается (trap).
# Только GPU0-1. Запуск: systemd-run --user --unit=kandidat-window2-... scripts/window5_clefgraph.sh OUTDIR
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
start_ab() {   # имя-захода
  AB=kandidat-ab-$1-$(date +%H%M).service
  log "старт A/B $AB (графы has_clef, выход сведений в основной пул)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM80_CG_SIZES=1,2,4,8,12,16,20,24,28,32,36,40,64,128,256,512 \
    --setenv=FA2SM70_CLEF_GRAPH=1 --setenv=FA2SM70_CLEF_GRAPH_FILE="$OUT/clefgraph" \
    --setenv=FA2SM80_EXCH_OUT_MAIN=1 \
    --setenv=FA2SM80_MEMTRACE=20 \
    "$REPO/scripts/serve_candidate_ab.sh"
  if ! wait_health http://127.0.0.1:8086 900 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
}
cli() { "$REPO/.venv/bin/python" "$REPO/scripts/acceptance_mtpgraphs.py" "$@" --url http://127.0.0.1:8086 >> "$OUT/window.log" 2>&1; log "клиент $1 ${*: -1}: код $?"; }
gpu_summary() { grep -a "MEMTRACE\] GPU" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
trap restore EXIT

echo 0 > "$OUT/budget"; printf 1 > "$OUT/clefgraph"
log "окно 5: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

start_ab cg
grep -aE "графобезопасная поправка|ok clef_ext|ok exch_ext|FAILED" "$OUT/server-cg.log" | head -6 | tee -a "$OUT/window.log"
log "графов has_clef захвачено (TP0, по журналу vLLM): $(grep -a 'Worker_TP0' "$OUT/server-cg.log" | grep -ac 'has_clef=True')"
gpu_summary cg
"$REPO/.venv/bin/python" "$REPO/scripts/acceptance_clefparity.py" "$OUT/parity" --url http://127.0.0.1:8086 \
  --toggle-file "$OUT/clefgraph" >> "$OUT/window.log" 2>&1
log "сверка Clef: код $?"
printf 1 > "$OUT/clefgraph"
"$REPO/.venv/bin/python" "$REPO/scripts/acceptance_livemix.py" "$OUT/smooth" --url http://127.0.0.1:8086 \
  --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" --phase-file "$OUT/clefgraph" --phase-values 1,0 \
  --clef-concurrency 6 --rounds 2 --phase-seconds 150 >> "$OUT/window.log" 2>&1
log "ровность при полной очереди Clef (1 -- графы has_clef, 0 -- прежний путь): код $?"
printf 1 > "$OUT/clefgraph"
sleep 25; gpu_summary cg
grep -a "Worker_TP0" "$OUT/server-cg.log" | grep -aE "delta_graph|голова не посчитана|шаг больше буферов" | tail -3 | tee -a "$OUT/window.log"
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
