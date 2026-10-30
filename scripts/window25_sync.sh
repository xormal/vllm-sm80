#!/usr/bin/env bash
# [ФРИЗЫ 30.10] Окно 25 -- остаток синхронизаций и обмен на средних M (владелец: «и продолжай сразу код и исправления по
# моим замечаниям»). Поверх окна 24b (Marlin, постоянные буферы конвейера, индексы GDN видом):
#   * построитель GDN: выборки по префиксной маске -- срезами (было 7 nonzero = синхронизаций на смешанный шаг),
#     num_computed не-спекулятивных строк на процессоре (FA2SM70_GDN_NCOMP_CPU, со сверкой FA2SM70_GDN_NCOMP_CHECK);
#   * метаданные causal_conv1d -- закреплённая память, неблокирующая копия, постоянные буферы (FA2SM70_CONV_META_PINNED);
#   * att (один файл $OUT/att): GDN_DEVMETA (метаданные чанков FLA на карте -- 7 синхронизаций), GDN_ALL_VARLEN (без цикла
#     по запросам с .to("cpu")), ATTN_VARLEN, MIX_DEC_UNIFORM;
#   * конвейер блока на средних M (256..2047) на Marlin (FA2SM80_BLOCKPIPE_M8_MIN=256): обмен TP куска -- за умножениями.
# Аудит синхронизаций -- по ВСЕМ шагам (FA2SM70_SYNC_AUDIT_ALL). Сравнение -- с base/new окна 24b.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
WT="${WT:?рабочее дерево с правками}"
BASE_TAR="${BASE_TAR:?база правленых файлов (tar)}"
FILES="vllm/v1/attention/backends/gdn_attn.py vllm/v1/attention/backends/utils.py vllm/v1/worker/gpu_model_runner.py sm70/integrations/vllm_sm70/fa2sm80_w8.py sm70/integrations/vllm_sm70/fa2sm80_blockpipe.py"
REF24="${REF24:?каталог окна 24b (q-base.json, clef-base.json)}"
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

echo "2048:0:4" > "$OUT/budget"; printf 0 > "$OUT/aud"; printf 1 > "$OUT/att"
log "окно 25: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }
for f in $FILES; do
  if ! tar -xOf "$BASE_TAR" "$f" | cmp -s - "$REPO/$f"; then
    log "ОТКАЗ: $f в основном дереве отличается от базы -- правки не применены"; exit 1
  fi
done
for f in $FILES; do cp -a "$WT/$f" "$REPO/$f"; done
log "правки применены: $(echo $FILES | wc -w) файлов"

start_ab w25 FA2SM80_W8_MARLIN=1 FA2SM80_BLOCKPIPE_BUFS=1 FA2SM70_GDN_IDX_VIEW=1 \
  FA2SM80_GDN_DEVMETA_FILE="$OUT/att" FA2SM80_GDN_ALL_VARLEN_FILE="$OUT/att" FA2SM80_ATTN_VARLEN_FILE="$OUT/att" \
  FA2SM80_MIX_DEC_UNIFORM=1 FA2SM80_MIX_DEC_FILE="$OUT/att" \
  FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_GDN_NCOMP_CHECK=300 FA2SM70_CONV_META_PINNED=1 FA2SM80_BLOCKPIPE_M8_MIN=256 \
  FA2SM70_SYNC_AUDIT="$OUT/aud_sync" FA2SM70_SYNC_AUDIT_FILE="$OUT/aud" FA2SM70_SYNC_AUDIT_ALL=1 \
  FA2SM70_ALLOC_AUDIT="$OUT/aud_alloc" FA2SM70_ALLOC_AUDIT_STEPS=20 || exit 1
gpu_summary w25
"$PYV" "$PS" clef "$PORT" "$OUT/clef-w25.json" > "$OUT/clef-w25.log" 2>&1
log "Clef [w25] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-w25.json" 2>&1 | tail -1)"
log "Clef w25 против base 24b: $("$PYV" "$PS" clefsr "$REF24/clef-base.json" "$OUT/clef-w25.json" 2>&1 | tail -1)"
QB="$REF24/q-base.json"
QN="$OUT/q-w25.json"
(cd "$BENCH" && URL=$URL OUT="$QN" REF="$QB" "$PYV" quality_ref.py) > "$OUT/q-w25.log" 2>&1
log "отпечаток LLM [w25 против base 24b]: $(grep -E '^L=' "$OUT/q-w25.log" | cut -c1-90 | tr '\n' ' ')"
( for r in 1 2; do "$PYV" "$PS" clef "$PORT" "$OUT/clef-load-$r.json" > /dev/null 2>&1; done ) &
LOADPID=$!
sleep 5
QL="$OUT/q-w25-load.json"
(cd "$BENCH" && URL=$URL OUT="$QL" REF="$QB" "$PYV" quality_ref.py) > "$OUT/q-w25-load.log" 2>&1
log "отпечаток LLM [w25 под нагрузкой Clef против base 24b]: $(grep -E '^L=' "$OUT/q-w25-load.log" | cut -c1-90 | tr '\n' ' ')"
wait $LOADPID 2>/dev/null
"$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye.json" 34816 > "$OUT/dlinnye.log" 2>&1
log "dlinnye [w25]: $(grep ИТОГ "$OUT/dlinnye.log")"
printf 1 > "$OUT/aud"; sleep 2
CONC=4 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/aud-mix.json" "aud" >> "$OUT/window.log" 2>&1
sleep 35; printf 0 > "$OUT/aud"; sleep 2
log "аудит снят; сверка ncomp: расхождений $(grep -ac 'СВЕРКА ncomp' "$OUT/server-w25.log")"
"$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-w25.json" w25 >> "$OUT/window.log" 2>&1
log "метрики w25 (att=1): код $?"
printf 0 > "$OUT/att"; sleep 3
CONC=4,8 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-w25-att0.json" w25att0 >> "$OUT/window.log" 2>&1
log "смесь w25 (att=0): код $?"
printf 1 > "$OUT/att"
# длинная пауза = один шаг с куском Clef: бюджет 1024 при K=4 (p99/max против потери пропускной Clef)
echo "1024:0:4" > "$OUT/budget"; sleep 3
CONC=4,8 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-w25-b1024.json" w25b1024 >> "$OUT/window.log" 2>&1
log "смесь w25 (бюджет 1024, K=4): код $?"
echo "2048:0:4" > "$OUT/budget"
sleep 5; gpu_summary w25
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT/server-w25.log" | sed "s/^/ошибок w25: /" | tee -a "$OUT/window.log"
