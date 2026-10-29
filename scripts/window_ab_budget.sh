#!/usr/bin/env bash
# [ПЛАВНОСТЬ 29.10] Окно обслуживания (разрешено владельцем): A/B бюджета prefill при декоде на живой смеси.
# Останавливает боевой кандидат, поднимает A/B-экземпляр на 127.0.0.1:8086 (тот же профиль/кэши), гоняет
# acceptance_livemix.py, затем ВСЕГДА (trap) гасит A/B и поднимает боевой юнит обратно с проверкой health.
# Только GPU0-1. Запуск: systemd-run --user --unit=kandidat-ab-window-... scripts/window_ab_budget.sh OUTDIR
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=kandidat-ab-budget-$(date +%H%M).service
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
restore() {
  log "восстановление: гашу A/B ($AB), поднимаю $LIVE"
  systemctl --user stop "$AB" 2>/dev/null
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
  systemctl --user reset-failed "$LIVE" 2>/dev/null
  systemctl --user start "$LIVE"
  systemctl --user start "$OBS"
  if wait_health http://127.0.0.1:8085 1200; then
    log "боевой $LIVE снова отвечает на 8085: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
  else
    log "ВНИМАНИЕ: боевой не ответил за 1200 с, состояние: $(systemctl --user show $LIVE -p ActiveState -p SubState | tr '\n' ' ')"
  fi
}
trap restore EXIT

log "окно: стоп $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

echo 0 > "$OUT/budget"
log "старт A/B $AB"
systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
  -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
  -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
  -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server.log" -p StandardError=inherit \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
  --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps.jsonl" \
  "$REPO/scripts/serve_candidate_ab.sh"
if ! wait_health http://127.0.0.1:8086 900 "$AB"; then
  log "A/B не поднялся за 900 с"; tail -40 "$OUT/server.log" >> "$OUT/window.log"; exit 1
fi
log "A/B готов, прогрев 60 с"
sleep 60
log "нагрузка: живая смесь"
"$REPO/.venv/bin/python" "$REPO/scripts/acceptance_livemix.py" "$OUT/livemix" --url http://127.0.0.1:8086 \
  --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" --budgets "${BUDGETS:-0,2048,1024,4096}" \
  --rounds "${ROUNDS:-2}" --phase-seconds "${PHASE:-150}" >> "$OUT/window.log" 2>&1
log "нагрузка завершена (код $?)"
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server.log" | sed 's/^/ошибок в журнале A\/B: /' | tee -a "$OUT/window.log"
