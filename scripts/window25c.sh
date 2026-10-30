#!/usr/bin/env bash
# [ФРИЗЫ 30.10] Окно 25c -- после найденной причины (признак раскладки Marlin был по форме (K, N): веса MTP НАШЕЙ раскладки той же
# формы шли в Marlin -> мусорный черновик -> illegal memory access); теперь признак -- у веса/слоя. Прежняя шапка 25b:
# Окно 25b -- поиск сбоя окна 25 (illegal memory access в первой тяжёлой смеси LLM+Clef) делением и
# исправления второй ревизии: повтор ячейки Clef упорядочен на карте (без ожидания хоста), результат головы копируется на
# хост сразу за головой, синхронный tap ждёт поток головы; полный граф -- только нативная свёртка; стенд -- паузы по полным
# ответам. Прогоны (каждый -- свой A/B): Clef против V100 + смесь c=4 (как в окне 25) + проверка падения:
#   A = конфиг окна 25 без FA2SM70_GDN_NCOMP_CPU и FA2SM70_CONV_META_PINNED;
#   падение A -> B = A без FA2SM80_BLOCKPIPE_M8_MIN; падение B -> C = B без att (файл); устоял A -> D = A + CONV_META,
#   затем E = D + NCOMP_CPU. Последний устоявший -- полные метрики c=1/4/8 + смесь при бюджете 1024.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
WT="${WT:?рабочее дерево с правками}"
BASE_TAR="${BASE_TAR:?база правленых файлов (tar)}"
FILES="sm70/integrations/vllm_sm70/fa2sm80_w8.py sm70/integrations/vllm_sm70/fa2sm80_rowpipe.py vllm/model_executor/models/qwen3_next.py"
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

echo "2048:0:4" > "$OUT/budget"; printf 1 > "$OUT/att"
log "окно 25c: стоп $OBS и $LIVE"
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
log "правки применены: $(echo $FILES | wc -w) файла"

COMMON="FA2SM80_W8_MARLIN=1 FA2SM80_BLOCKPIPE_BUFS=1 FA2SM70_GDN_IDX_VIEW=1 FA2SM80_GDN_DEVMETA_FILE=$OUT/att FA2SM80_GDN_ALL_VARLEN_FILE=$OUT/att FA2SM80_ATTN_VARLEN_FILE=$OUT/att FA2SM80_MIX_DEC_UNIFORM=1 FA2SM80_MIX_DEC_FILE=$OUT/att"
crashed() { ! curl -sf -m 5 "$URL/health" >/dev/null || grep -aqE "illegal memory access|EngineDeadError|CUDA error" "$OUT/server-$1.log"; }
run_cfg() {   # имя env...
  local name=$1; shift
  start_ab "$name" "$@" || { log "$name: не поднялся"; return 2; }
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$name.json" > "$OUT/clef-$name.log" 2>&1
  log "Clef [$name] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$name.json" 2>&1 | tail -1)"
  CONC=4 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/mix-$name.json" "$name" >> "$OUT/window.log" 2>&1
  CONC=8 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/mix8-$name.json" "$name" >> "$OUT/window.log" 2>&1
  if crashed "$name"; then
    log "$name: УПАЛ ($(grep -aoE 'illegal memory access|EngineDeadError|CUDA error[^.]*' "$OUT/server-$name.log" | head -1))"
    stop_ab; return 1
  fi
  log "$name: устоял"
  return 0
}
RESULT=""
# shellcheck disable=SC2086
if run_cfg E $COMMON FA2SM80_BLOCKPIPE_M8_MIN=256 FA2SM70_CONV_META_PINNED=1 FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_GDN_NCOMP_CHECK=300; then
  RESULT="E"
  QB="$REF24/q-base.json"
  (cd "$BENCH" && URL=$URL OUT="$OUT/q-E.json" REF="$QB" "$PYV" quality_ref.py) > "$OUT/q-E.log" 2>&1
  log "отпечаток LLM [E против base 24b]: $(grep -E '^L=' "$OUT/q-E.log" | cut -c1-90 | tr '\n' ' ')"
  log "Clef E против base 24b: $("$PYV" "$PS" clefsr "$REF24/clef-base.json" "$OUT/clef-E.json" 2>&1 | tail -1)"
  log "сверка ncomp: расхождений $(grep -ac 'СВЕРКА ncomp' "$OUT/server-E.log")"
fi
log "итог деления: последний устоявший -- ${RESULT:-нет}"
if [ -n "$RESULT" ]; then
  if ! curl -sf -m 5 "$URL/health" >/dev/null; then
    case "$RESULT" in
      A) run_cfg A2 $COMMON FA2SM80_BLOCKPIPE_M8_MIN=256 ;;
      D) run_cfg D2 $COMMON FA2SM80_BLOCKPIPE_M8_MIN=256 FA2SM70_CONV_META_PINNED=1 ;;
      B|C) run_cfg B2 $COMMON ;;
    esac
  fi
  "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-final.json" final >> "$OUT/window.log" 2>&1
  log "метрики ($RESULT): код $?"
  echo "1024:0:4" > "$OUT/budget"; sleep 3
  CONC=4,8 PARTS=mix "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/bench-b1024.json" b1024 >> "$OUT/window.log" 2>&1
  log "смесь (бюджет 1024, K=4): код $?"
  echo "2048:0:4" > "$OUT/budget"
fi
sleep 5
for n in A B C D E A2 B2 D2; do [ -e "$OUT/server-$n.log" ] && gpu_summary "$n" >/dev/null && grep -a "MEMTRACE\] .*GPU0:" "$OUT/server-$n.log" | tail -1 | sed "s/^.*MEMTRACE\] /$n: /" | tee -a "$OUT/window.log"; done
