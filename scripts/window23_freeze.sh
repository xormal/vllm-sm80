#!/usr/bin/env bash
# [ФРИЗЫ 30.10] Окно 23 -- корень фризов чатов под Clef (владелец: «докапывайся до фризов, надо устранить»; «переводи с
# ядер марлин на наши ядра»; «убери лишние выходы в хост на cpu сразу»; «ревизию графа, чтобы лишних блоков по пути
# аллокатора не выделялось в обе стороны флоу обработки параллельных запросов llm/clef»). Окна 20/21: шаг с куском Clef
# ~440 токенов -- 273 мс = GEMM 150 (~79 TF) + обмен TP 62 + прочее; ~12 синхронизаций с картой и ~3600 выделений
# распределителя на смешанный шаг (у декода в графе -- 65). Только GPU0-1, A/B на 127.0.0.1:18493:
#   1) стенд GEMM (GPU0, сборка из рабочего дерева в scratchpad/build_g4): Marlin u8b128 против нашего маршрута и v4
#      (конвейер Marlin в наших ядрах), побитность v4 против прямого ядра;
#   2) пауза под ncu (запускается вручную, sudo): ждёт $OUT/ncu.done до 25 мин;
#   3) правки Python из рабочего дерева (флаги по умолчанию выключены): аудит синхронизаций/выделений смешанного шага
#      (FA2SM70_SYNC_AUDIT / FA2SM70_ALLOC_AUDIT), индексы токенов GDN видами без синхронизации (FA2SM70_GDN_IDX_VIEW_FILE),
#      чередование декода в планировщике (третье поле файла бюджета);
#   4) A/B -- профиль боевого: сверка Clef/LLM при индексах видом 0/1 (в т.ч. под нагрузкой Clef), аудит под смесью
#      LLM+Clef при 0 и 1, затем смесь c=4 при чередовании K=0/4 (без снимка распределителя).
# Боевой поднимается всегда (trap); правки остаются в основном дереве с выключенными флагами.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
WT="${WT:?рабочее дерево с правками}"
BASE_TAR="${BASE_TAR:?база правленых файлов (tar)}"
FILES="vllm/v1/attention/backends/gdn_attn.py vllm/v1/worker/gpu_model_runner.py vllm/v1/core/sched/scheduler.py"
BUILD="${BUILD:?сборка расширения с v4 (scratchpad/build_g4)}"
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

echo "2048:0:0" > "$OUT/budget"; printf 0 > "$OUT/iv"
log "окно 23: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 1) стенд GEMM на GPU0 (расширение -- из сборки рабочего дерева; боевые .so не трогаются)
log "стенд GEMM: Marlin / наш маршрут / v4"
CUDA_VISIBLE_DEVICES=0 FA2SM70_PATH="$WT/sm70" FA2SM70_BUILD_DIR="$BUILD" \
  "$(dirname "$REPO")/../sm80_work/bin/memrun" 10G "$PYV" "$REPO/scripts/marlin_w8_bench.py" "$OUT/marlin_bench.json" \
  > "$OUT/marlin_bench.log" 2>&1
log "стенд GEMM: код $?; $(grep -c '' "$OUT/marlin_bench.log") строк"
grep -E "^проход" "$OUT/marlin_bench.log" | tee -a "$OUT/window.log"

# 2) пауза под ncu (вручную)
touch "$OUT/ncu.wait"; log "пауза под ncu: жду $OUT/ncu.done (до 25 мин)"
for i in $(seq 1 300); do [ -e "$OUT/ncu.done" ] && break; sleep 5; done
log "ncu: $( [ -e "$OUT/ncu.done" ] && echo готово || echo 'не дождался, дальше')"
gpus_free || { log "GPU0-1 заняты после ncu"; exit 1; }

# 3) правки Python -- только сейчас: боевой остановлен, ни один сервер их не импортирует
for f in $FILES; do
  if ! tar -xOf "$BASE_TAR" "$f" | cmp -s - "$REPO/$f"; then
    log "ОТКАЗ: $f в основном дереве отличается от базы -- правки не применены"; exit 1
  fi
done
for f in $FILES; do cp -a "$WT/$f" "$REPO/$f"; done
log "правки применены: $FILES"

# 4) A/B -- профиль боевого + аудит
start_ab prod FA2SM70_SYNC_AUDIT="$OUT/aud_sync" FA2SM70_ALLOC_AUDIT="$OUT/aud_alloc" FA2SM70_ALLOC_AUDIT_STEPS=20 \
  FA2SM70_GDN_IDX_VIEW_FILE="$OUT/iv" || exit 1
gpu_summary prod
setiv() { printf "$1" > "$OUT/iv"; sleep 2; log "iv=$1"; }
setiv 0
"$PYV" "$PS" clef "$PORT" "$OUT/clef-iv0.json" > "$OUT/clef-iv0.log" 2>&1
log "Clef [iv=0] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-iv0.json" 2>&1 | tail -1)"
(cd "$BENCH" && URL=$URL OUT="$OUT/q-iv0.json" "$PYV" quality_ref.py) > "$OUT/q-iv0.log" 2>&1
# аудит при iv=0: смесь LLM+Clef c=4 (снимок распределителя -- первые 20 смешанных шагов)
CONC=4 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/aud-mix-iv0.json" "aud-iv0" >> "$OUT/window.log" 2>&1
sleep 35; for r in 0 1; do cp -f "$OUT/aud_sync.rank$r.json" "$OUT/aud_sync_iv0.rank$r.json" 2>/dev/null; done
log "аудит iv=0 снят: $(ls "$OUT"/aud_alloc* 2>/dev/null | tr '\n' ' ')"
setiv 1
"$PYV" "$PS" clef "$PORT" "$OUT/clef-iv1.json" > "$OUT/clef-iv1.log" 2>&1
log "Clef [iv=1] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-iv1.json" 2>&1 | tail -1)"
log "Clef iv=1 против iv=0: $("$PYV" "$PS" clefsr "$OUT/clef-iv0.json" "$OUT/clef-iv1.json" 2>&1 | tail -1)"
(cd "$BENCH" && URL=$URL OUT="$OUT/q-iv1.json" REF="$OUT/q-iv0.json" "$PYV" quality_ref.py) > "$OUT/q-iv1.log" 2>&1
log "отпечаток LLM [iv=1 против iv=0]: $(grep -E '^L=' "$OUT/q-iv1.log" | cut -c1-90 | tr '\n' ' ')"
( for r in 1 2; do "$PYV" "$PS" clef "$PORT" "$OUT/clef-load-$r.json" > /dev/null 2>&1; done ) &
LOADPID=$!
sleep 5
(cd "$BENCH" && URL=$URL OUT="$OUT/q-iv1-load.json" REF="$OUT/q-iv0.json" "$PYV" quality_ref.py) > "$OUT/q-iv1-load.log" 2>&1
log "отпечаток LLM [iv=1 под нагрузкой Clef против iv=0]: $(grep -E '^L=' "$OUT/q-iv1-load.log" | cut -c1-90 | tr '\n' ' ')"
wait $LOADPID 2>/dev/null
CONC=4 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/aud-mix-iv1.json" "aud-iv1" >> "$OUT/window.log" 2>&1
sleep 35; for r in 0 1; do cp -f "$OUT/aud_sync.rank$r.json" "$OUT/aud_sync_iv1.rank$r.json" 2>/dev/null; done
log "аудит iv=1 снят"
# чередование декода (iv=1): K=0 и K=4, смесь c=4 и c=8
for k in 0 4; do
  echo "2048:0:$k" > "$OUT/budget"; sleep 2; log "K=$k"
  CONC=4,8 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/mix-k$k.json" "k$k" >> "$OUT/window.log" 2>&1
done
echo "2048:0:0" > "$OUT/budget"
sleep 10; gpu_summary prod
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-prod.log" | sed "s/^/ошибок prod: /" | tee -a "$OUT/window.log"
