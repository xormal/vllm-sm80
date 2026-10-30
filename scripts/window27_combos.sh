#!/usr/bin/env bash
# [МЕТРИКИ 30.10] Окно 27 -- раздельно префилл и декод LLM и Clef и их сочетания (владелец 30.10: «раздельно скорость
# clef без обращения к llm и скорость llm без обращения к clef; при 4 потоках llm, 1 clef; при максимальном clef сколько
# декода/префилла в llm на 1/4/8 и больше; все с разделением на префилл декод»). Код -- тот, что в основном дереве
# после окна 26 (правок не вносит); конфиг E окна 26 (флаги новых путей поверх боевого, K=4).
#   1) бюджет 2048: отдельно LLM (генерация и префилл c=1/4/8/16), отдельно Clef (c=1/4/8; ячеек Clef 3 -- выше c=4
#      только очередь), сочетания L:C = 4:1, 1:8, 4:8, 8:8, 16:8 (Clef c=8 -- насыщение) -- декод и префилл LLM;
#   2) бюджет 1024: те же сочетания;
#   3) бюджет 2048: смесь c=4 при att=1 и att=0 (файл, переключение в простое) -- шаги 2047-2200 токенов против +30 мс
#      окна 26 (метки строк файла шагов в window.log).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
PORT="${AB_PORT:-18493}"
URL=http://127.0.0.1:$PORT
RESTORE="${RESTORE:-1}"
COMBOS="${COMBOS:-4:1,1:8,4:8,8:8,16:8}"
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
crashed() { ! curl -sf -m 5 "$URL/health" >/dev/null || grep -aqE "illegal memory access|EngineDeadError|CUDA error|CUDA out of memory" "$OUT/server-$1.log"; }
mark() { log "метка шагов [$1]: строка $(wc -l < "$OUT/steps-E.jsonl" 2>/dev/null || echo 0)"; }
bench() {   # файл метка  (окружение CONC/PARTS/COMBOS -- у вызова)
  "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/$1" "$2" >> "$OUT/window.log" 2>&1
  local rc=$?
  log "$2: код $rc"
  if crashed E; then log "A/B УПАЛ после $2: $(grep -aoE 'illegal memory access|EngineDeadError|CUDA error[^.]*|CUDA out of memory' "$OUT/server-E.log" | head -1)"; exit 1; fi
}
trap restore EXIT

echo "2048:0:4" > "$OUT/budget"; printf 1 > "$OUT/att"
log "окно 27: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }
log "код: $(md5sum "$REPO/sm70/integrations/vllm_sm70/fa2sm80_w8.py" "$REPO/sm70/integrations/vllm_sm70/fa2sm70_clef.py" "$REPO/vllm/v1/core/sched/scheduler.py" | cut -c1-8 | tr '\n' ' ')"

COMMON="FA2SM80_W8_MARLIN=1 FA2SM80_BLOCKPIPE_BUFS=1 FA2SM70_GDN_IDX_VIEW=1 FA2SM80_GDN_DEVMETA_FILE=$OUT/att FA2SM80_GDN_ALL_VARLEN_FILE=$OUT/att FA2SM80_ATTN_VARLEN_FILE=$OUT/att FA2SM80_MIX_DEC_UNIFORM=1 FA2SM80_MIX_DEC_FILE=$OUT/att"
# shellcheck disable=SC2086
start_ab E $COMMON FA2SM80_BLOCKPIPE_M8_MIN=256 FA2SM70_CONV_META_PINNED=1 FA2SM70_GDN_NCOMP_CPU=1 || exit 1

mark "b2048 отдельно"
CONC=1,4,8,16 PARTS=gen,prefill bench alone-llm.json "отдельно LLM (бюджет 2048)"
CONC=1,4,8 PARTS=clef bench alone-clef.json "отдельно Clef"
mark "b2048 сочетания"
CONC=1 PARTS=combo COMBOS="$COMBOS" bench combo-b2048.json "сочетания (бюджет 2048)"
echo "1024:0:4" > "$OUT/budget"; sleep 3
mark "b1024 сочетания"
CONC=1 PARTS=combo COMBOS="$COMBOS" bench combo-b1024.json "сочетания (бюджет 1024)"
echo "2048:0:4" > "$OUT/budget"; sleep 3
mark "att=1 смесь c=4"
CONC=4 PARTS=mix bench mix-att1.json "смесь c=4 att=1"
printf 0 > "$OUT/att"; sleep 3
mark "att=0 смесь c=4"
CONC=4 PARTS=mix bench mix-att0.json "смесь c=4 att=0"
printf 1 > "$OUT/att"
mark "конец"
log "ошибок в журнале A/B: $(grep -acE 'CUDA out of memory|EngineDeadError|Traceback' "$OUT/server-E.log")"
grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-E.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"
