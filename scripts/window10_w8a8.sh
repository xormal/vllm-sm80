#!/usr/bin/env bash
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
log "окно 10: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 1) ядро на одной карте: реализация против fp64 той же математики, ошибка квантования, скорость против декванта+GEMM
log "ядро W8A8 на GPU0"
systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
  bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0; exec "$2" "$1/scripts/w8a8_kernel_test.py" "$3"' \
  _ "$REPO" "$PYV" "$OUT/kernel.json" >> "$OUT/kernel.log" 2>&1
log "тест ядра: код $?"
grep -v "^\s*$" "$OUT/kernel.log" | grep -E '^\{' | cut -c1-400 >> "$OUT/window.log"
A8_OK=$("$PYV" - "$OUT/kernel.json" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    ok = bool(r) and all(x["реализация_relL2"] < 2e-3 and x["out_срез_побитно"] for x in r)
except Exception:
    ok = False
print(1 if ok else 0)
PY
)
log "ядро W8A8 годно для A/B: $A8_OK"
for i in $(seq 1 30); do gpus_free && break; sleep 2; done

start_ab w8
"$PYV" "$REPO/scripts/acceptance_clefparity.py" "$OUT/parity" --url $URL --toggle-file "$OUT/head" >> "$OUT/window.log" 2>&1
log "сверка Clef: голова асинхронно (1) против синхронно (0): код $?"
printf 1 > "$OUT/head"

if [ "$A8_OK" = "1" ]; then
  # 2) точность Clef: записи стенда (35 решений) против эталона V100; off -> clef -> off (пол шума)
  for m in off clef off2; do
    mode "${m%2}"
    "$PYV" "$PS" clef "$PORT" "$OUT/clef-$m.json" > "$OUT/clef-$m.log" 2>&1
    log "Clef [$m] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$m.json" 2>&1 | tail -1)"
  done
  log "Clef clef против off: $("$PYV" "$PS" clefsr "$OUT/clef-off.json" "$OUT/clef-clef.json" 2>&1 | tail -1)"
  log "Clef off2 против off: $("$PYV" "$PS" clefsr "$OUT/clef-off.json" "$OUT/clef-off2.json" 2>&1 | tail -1)"
  for m in off clef; do
    mode "$m"
    "$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye-$m.json" 34816 > "$OUT/dlinnye-$m.log" 2>&1
    log "dlinnye [$m]: $(grep ИТОГ "$OUT/dlinnye-$m.log")"
  done
  # 3) точность LLM: отпечаток (600/2048/5000, 160 токенов, T=0, logprob) и иглы 60K; off -> llm -> off
  for m in off llm off2; do
    mode "${m%2}"
    ref=""; [ "$m" != off ] && ref="$OUT/q-off.json"
    (cd "$BENCH" && URL=$URL OUT="$OUT/q-$m.json" REF="$ref" "$PYV" quality_ref.py) > "$OUT/q-$m.log" 2>&1
    log "отпечаток LLM [$m]: $(grep -E '^L=' "$OUT/q-$m.log" | tr '\n' ' ')"
    (cd "$BENCH" && URL=$URL NL=60000 NEEDLE_TAG="w8a8-$m" "$PYV" needle.py) > "$OUT/needle-$m.log" 2>&1
    log "иглы 60K [$m]: $(grep ИТОГ "$OUT/needle-$m.log")"
  done
  # 4) скорость префилла без нагрузки: свежий префилл 8K и 32K по режимам (off -> all -> off -> all)
  for m in off all off2 all2; do
    mode "${m%2}"
    (cd "$BENCH" && URL=$URL LENS=8192,32768 REPS=1 NTOK=32 "$PYV" e2e_decode.py) > "$OUT/prefill-$m.log" 2>&1
    log "префилл [$m]: $(grep -E '^L=' "$OUT/prefill-$m.log" | cut -c1-60 | tr '\n' ' ')"
  done
  # 5) под нагрузкой: насыщенная очередь Clef + 8 чатов, режимы по фазам
  echo off > "$OUT/w8a8"
  lm sat-w8a8 --clef-concurrency 6 --phase-file "$OUT/w8a8" --phase-values "off,clef,llm,all" --rounds 2 --phase-seconds 100
  echo off > "$OUT/w8a8"
  grep -a "W8A8\]" "$OUT/server-w8.log" | grep -a "Worker_TP0" | grep -aE "сверка|режим" | head -40 | cut -c1-300 >> "$OUT/window.log"
fi
sleep 25; gpu_summary w8
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
