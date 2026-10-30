#!/usr/bin/env bash
# [ПЛАВНОСТЬ 30.10] Окно 22 -- ЧЕРЕДОВАНИЕ шагов чистого декода (FA2SM80_DECODE_INTERLEAVE, третье поле файла бюджета
# "бюджет:порог:K"). Окна 21/21b: при одновременных Clef и чатах каждый шаг несёт кусок Clef ~2000 токенов (~0.6 с),
# чат -- ~5-7 ток/с на поток (без Clef 50-125). Правка -- только vllm/v1/core/sched/scheduler.py (при K=0 поведение
# прежнее). A/B -- профиль боевого (бюджет 2048:0:K), только GPU0-1, 127.0.0.1:18493:
#   1) сверка Clef против V100 и отпечаток LLM при K=4 (чередование меняет только границы шагов);
#   2) bench_conc.py PARTS=mix (LLM+Clef c=1/4/8) при K = 0, 2, 4, 8;
#   3) насыщенная живая смесь (Clef 6 потоков + 8 чатов) фазами K = 0/2/4/8.
# Правка остаётся в основном дереве (при неустановленном флаге -- прежнее поведение); боевой поднимается всегда (trap).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
WT="${WT:?рабочее дерево с правками}"
BASE_TAR="${BASE_TAR:?база правленых файлов (tar)}"
FILES="vllm/v1/core/sched/scheduler.py"
PS="$REPO/sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
BENCH="$(dirname "$REPO")/sm80_work/bench"
REF_CLEF="$(dirname "$REPO")/docs/reports/postanovka-2026-10-19/clef.json"
PORT="${AB_PORT:-18493}"
URL=http://127.0.0.1:$PORT
RESTORE="${RESTORE:-1}"
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

echo "2048:0:0" > "$OUT/budget"
log "окно 22: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 0) правка -- только сейчас: боевой остановлен, ни один сервер её не импортирует
for f in $FILES; do
  if ! tar -xOf "$BASE_TAR" "$f" | cmp -s - "$REPO/$f"; then
    log "ОТКАЗ: $f в основном дереве отличается от базы -- правка не применена"; exit 1
  fi
done
for f in $FILES; do cp -a "$WT/$f" "$REPO/$f"; done
log "правка применена: $FILES"

start_ab prod || exit 1
gpu_summary prod
setk() { echo "2048:0:$1" > "$OUT/budget"; sleep 2; log "K=$1"; }
# 1) точность при K=4 (и K=0 как база в этом же экземпляре)
for k in 0 4; do
  setk $k
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-k$k.json" > "$OUT/clef-k$k.log" 2>&1
  log "Clef [K=$k] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-k$k.json" 2>&1 | tail -1)"
done
setk 0; (cd "$BENCH" && URL=$URL OUT="$OUT/q-k0.json" "$PYV" quality_ref.py) > "$OUT/q-k0.log" 2>&1
setk 4
( for r in 1 2; do "$PYV" "$PS" clef "$PORT" "$OUT/clef-load-$r.json" > /dev/null 2>&1; done ) &
LOADPID=$!
sleep 5
(cd "$BENCH" && URL=$URL OUT="$OUT/q-k4-load.json" REF="$OUT/q-k0.json" "$PYV" quality_ref.py) > "$OUT/q-k4-load.log" 2>&1
log "отпечаток LLM [K=4 под нагрузкой Clef]: $(grep -E '^L=' "$OUT/q-k4-load.log" | cut -c1-90 | tr '\n' ' ')"
wait $LOADPID 2>/dev/null
# 2) LLM+Clef при c=1/4/8
for k in 0 2 4 8; do
  setk $k
  PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-k$k.json" "k$k" >> "$OUT/window.log" 2>&1
  log "смесь K=$k: код $?"
done
# 3) насыщенная живая смесь
"$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/sat-k" --url $URL --model Qwen3.8-27B-ABLITERATED \
  --budget-file "$OUT/budget" --budgets "2048:0:0,2048:0:2,2048:0:4,2048:0:8" --clef-concurrency 6 --rounds 1 \
  --phase-seconds 180 >> "$OUT/window.log" 2>&1
log "живая смесь sat-k: код $?"
echo "2048:0:0" > "$OUT/budget"
sleep 10; gpu_summary prod
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-prod.log" | sed "s/^/ошибок prod: /" | tee -a "$OUT/window.log"
