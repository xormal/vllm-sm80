#!/usr/bin/env bash
# [ФРИЗЫ 30.10] Окно 24b (продолжение окна 24 с A/B) -- всё вместе против боевого (владелец: «включай K=4 на боевом и готовь окно 24»; «смотри где пути
# можно совместить, чтобы переиспользовать буферы аллокатора»; исправления по ревизии: ячейка Clef до конца tap, голова
# графовой ветки на побочном потоке, публикация результатов только сборщиком, согласованный между рангами TP файл режима,
# строгий стенд). Только GPU0-1, A/B на 127.0.0.1:18493:
#   0) правки из рабочего дерева (Python + clef.cu; флаги новых путей по умолчанию выключены), пересборка clef в
#      боевом каталоге сборки через candidate_build (marlin8 собран заранее), проверка загрузки без сборки;
#   1) стенд marlin8 на GPU0: точность, деквант из раскладки Marlin побитно против нашего, граф, время;
#   2) A/B base -- профиль боевого как сейчас (K=4) + исправления Clef: сверка Clef/LLM, метрики c=1/4/8;
#   3) A/B new -- base + FA2SM80_W8_MARLIN=1 + FA2SM80_BLOCKPIPE_BUFS=1 + FA2SM70_GDN_IDX_VIEW=1: сверка Clef (V100 и
#      base), LLM против base (в т.ч. под Clef), короткий аудит синхронизаций (файл) и снимок распределителя, метрики.
# Боевой поднимается всегда (trap) на коде окна с выключенными флагами новых путей (исправления Clef -- активны).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
WT="${WT:?рабочее дерево с правками}"
BASE_TAR="${BASE_TAR:?база правленых файлов (tar)}"
FILES="sm70/integrations/vllm_sm70/fa2sm80_w8.py sm70/integrations/vllm_sm70/fa2sm80_blockpipe.py sm70/integrations/vllm_sm70/fa2sm80_rowpipe.py sm70/integrations/vllm_sm70/fa2sm70_exch_i12.py sm70/integrations/vllm_sm70/fa2sm70_clef.py sm70/integrations/vllm_sm70/fa2sm70_fullmix.py sm70/fa2_sm70/_ext.py vllm/v1/worker/gpu_model_runner.py sm70/fa2_sm70/csrc/clef.cu"
BD="$(dirname "$REPO")/sm80_work/build_sm80"
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

echo "2048:0:4" > "$OUT/budget"; printf 0 > "$OUT/aud"
log "окно 24: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# [окно 24b] правки уже применены окном 24 (19:44), clef пересобран, стенд marlin8 пройден; base упал на сборщике
# Clef (опрос события вне замка захвата -> сорван захват графа MTP) -- исправленный fa2sm70_clef.py уже в дереве.
log "окно 24b: продолжение с A/B (исправлен сборщик Clef)"

# 2) base
start_ab base || exit 1
gpu_summary base
"$PYV" "$PS" clef "$PORT" "$OUT/clef-base.json" > "$OUT/clef-base.log" 2>&1
log "Clef [base] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-base.json" 2>&1 | tail -1)"
QB="$OUT/q-base.json"
(cd "$BENCH" && URL=$URL OUT="$QB" "$PYV" quality_ref.py) > "$OUT/q-base.log" 2>&1
"$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-base.json" base >> "$OUT/window.log" 2>&1
log "метрики base: код $?"
sleep 5; gpu_summary base
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-base.log" | sed "s/^/ошибок base: /" | tee -a "$OUT/window.log"
stop_ab

# 3) new
start_ab new FA2SM80_W8_MARLIN=1 FA2SM80_BLOCKPIPE_BUFS=1 FA2SM70_GDN_IDX_VIEW=1 \
  FA2SM70_SYNC_AUDIT="$OUT/aud_sync" FA2SM70_SYNC_AUDIT_FILE="$OUT/aud" \
  FA2SM70_ALLOC_AUDIT="$OUT/aud_alloc" FA2SM70_ALLOC_AUDIT_STEPS=20 || exit 1
gpu_summary new
grep -aE "Marlin: перепаковано|marlin" "$OUT/server-new.log" | grep -a Worker_TP0 | tail -2 | cut -c1-200 >> "$OUT/window.log"
"$PYV" "$PS" clef "$PORT" "$OUT/clef-new.json" > "$OUT/clef-new.log" 2>&1
log "Clef [new] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-new.json" 2>&1 | tail -1)"
log "Clef new против base: $("$PYV" "$PS" clefsr "$OUT/clef-base.json" "$OUT/clef-new.json" 2>&1 | tail -1)"
QN="$OUT/q-new.json"
(cd "$BENCH" && URL=$URL OUT="$QN" REF="$QB" "$PYV" quality_ref.py) > "$OUT/q-new.log" 2>&1
log "отпечаток LLM [new против base]: $(grep -E '^L=' "$OUT/q-new.log" | cut -c1-90 | tr '\n' ' ')"
( for r in 1 2; do "$PYV" "$PS" clef "$PORT" "$OUT/clef-load-$r.json" > /dev/null 2>&1; done ) &
LOADPID=$!
sleep 5
QL="$OUT/q-new-load.json"
(cd "$BENCH" && URL=$URL OUT="$QL" REF="$QB" "$PYV" quality_ref.py) > "$OUT/q-new-load.log" 2>&1
log "отпечаток LLM [new под нагрузкой Clef против base]: $(grep -E '^L=' "$OUT/q-new-load.log" | cut -c1-90 | tr '\n' ' ')"
wait $LOADPID 2>/dev/null
printf 1 > "$OUT/aud"; sleep 2
CONC=4 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/aud-mix.json" "aud" >> "$OUT/window.log" 2>&1
sleep 35; printf 0 > "$OUT/aud"; sleep 2
log "аудит синхронизаций снят: $(ls "$OUT"/aud_sync* "$OUT"/aud_alloc* 2>/dev/null | wc -l) файлов"
"$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-new.json" new >> "$OUT/window.log" 2>&1
log "метрики new: код $?"
sleep 5; gpu_summary new
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-new.log" | sed "s/^/ошибок new: /" | tee -a "$OUT/window.log"
