#!/usr/bin/env bash
# [ПЛАВНОСТЬ 30.10] Окно 10b: сверка асинхронной головы после исправления сборщика (замок захвата графов),
# повтор записей Clef в режиме W8A8 clef, A/B головы при бюджете 2048:0. Шапка -- от окна 10:
# [W8A8 29.10] Окно 10 (окно обслуживания по согласию владельца): ядро W8A8 на GPU0 (точность реализации, скорость),
# затем A/B на 127.0.0.1:$AB_PORT (умолчание 18493) с профилем и кэшами боевого: сверка асинхронной головы Clef (сборщик в фоне),
# точность W8A8 раздельно для Clef (FA2SM80_W8A8=clef против off, эталон V100) и LLM (llm против off: отпечаток
# logprob, иглы 60K), скорость префилла и плавность декода под нагрузкой Clef по режимам. Режим W8A8 -- файлом.
# Боевой ВСЕГДА поднимается (trap). Только GPU0-1. Запуск: systemd-run --user --unit=kandidat-window10-... этот OUTDIR
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
RESTORE="${RESTORE:-1}"             # 0 -- боевой не поднимать по выходу (владелец: тесты подряд)
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
  if wait_health http://127.0.0.1:8085 1500; then
    log "боевой $LIVE снова отвечает на 8085: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
  else
    log "ВНИМАНИЕ: боевой не ответил за 1500 с: $(systemctl --user show $LIVE -p ActiveState -p SubState | tr '\n' ' ')"
  fi
}
start_ab() {   # имя-захода
  AB=kandidat-ab-$1-$(date +%H%M).service
  log "старт A/B $AB (W8A8 и голова -- переключатели-файлы)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM70_CLEF_HEAD_ASYNC=1 --setenv=FA2SM70_CLEF_HEAD_ASYNC_FILE="$OUT/head" \
    --setenv=FA2SM80_W8A8_FILE="$OUT/w8a8" --setenv=FA2SM80_W8A8_CHECK=2 \
    --setenv=FA2SM80_MEMTRACE=20 --setenv=AB_PORT="$PORT" \
    "$REPO/scripts/serve_candidate_ab.sh"
  # первый старт после правки трассируемых модулей (fa2sm80_w8, конвейер) -- новая компиляция Inductor
  if ! wait_health $URL 1800 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
}
gpu_summary() { grep -a "MEMTRACE\] GPU" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
warm() {   # два запроса со случайными токенами (без кэша префиксов): переключатель W8A8 читается по счётчику вызовов
  "$PYV" - "$URL" <<'PY'
import json, random, sys, urllib.request
for _ in range(2):
    ids = [random.randint(1000, 60000) for _ in range(700)]
    req = urllib.request.Request(sys.argv[1] + "/v1/completions", json.dumps({
        "model": "Qwen3.8-27B-ABLITERATED", "prompt": ids, "max_tokens": 1}).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
PY
}
mode() { echo "$1" > "$OUT/w8a8"; sleep 1; warm; log "режим W8A8: $1"; }
lm() {   # имя аргументы...
  "$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/$1" --url $URL \
    --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" "${@:2}" >> "$OUT/window.log" 2>&1
  log "живая смесь $1: код $?"
}
trap restore EXIT

echo "2048:0" > "$OUT/budget"; printf 1 > "$OUT/head"; echo off > "$OUT/w8a8"
log "окно 10b: стоп $OBS и $LIVE (если подняты)"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }
start_ab hb
"$PYV" "$REPO/scripts/acceptance_clefparity.py" "$OUT/parity" --url $URL --toggle-file "$OUT/head" >> "$OUT/window.log" 2>&1
log "сверка Clef: голова асинхронно (1) против синхронно (0): код $?"
printf 0 > "$OUT/head"
for m in off clef; do
  mode "$m"
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$m.json" > "$OUT/clef-$m.log" 2>&1
  log "Clef [$m] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$m.json" 2>&1 | tail -1)"
done
log "Clef clef против off: $("$PYV" "$PS" clefsr "$OUT/clef-off.json" "$OUT/clef-clef.json" 2>&1 | tail -1)"
echo off > "$OUT/w8a8"
printf 1 > "$OUT/head"
lm sat-head --clef-concurrency 6 --phase-file "$OUT/head" --phase-values 1,0 --rounds 2 --phase-seconds 120
printf 1 > "$OUT/head"
grep -aE "голова:|не посчитана" "$OUT/server-hb.log" | head -5 >> "$OUT/window.log"
sleep 25; gpu_summary hb
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
