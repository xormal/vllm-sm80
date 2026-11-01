#!/usr/bin/env bash
# [ПАМЯТЬ 01.11] Окно 31 -- пулы графов черновика MTP и фаззинг утечек (владелец 01.11: «надо с утечкой памяти
# бороться, проверить всё фаззингом»; «боевой можно дернуть и накатить фиксы»). 01.11 15:47 боевой упал CUDA OOM на
# префилле 8192: графы цикла черновика захватывались каждый на НОВОМ потоке (graph_capture(device)), аллокатор не
# отдавал им промежутки прежних захватов -- приватные пулы 92 -> 496 МиБ за 14 ч при занятых ~27.
#   1) A/B O = боевой профиль как есть + фаззер 7 мин: воспроизводит ли фаззер рост пулов;
#   2) A/B N = + FA2SM80_MTP_LOOP_ONE_STREAM=1, пределы графов цикла и тела 16 (LRU), tracemalloc в ядре движка
#      (прирост от первого снимка -- рост памяти EngineCore на хосте): фаззер 12 мин, затем Clef и отпечаток LLM;
#   3) N чисто -> в боевой (строки в скрипте + перезапуск, проверка 5 мин, откат при сбое); иначе -- прежний профиль.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
LIVE_SH="$REPO/scripts/serve_candidate_live.sh"
LL="$(dirname "$REPO")/state/acceptance/live-20261029/server.log"
PS="$REPO/sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
BENCH="$(dirname "$REPO")/sm80_work/bench"
REF_CLEF="$(dirname "$REPO")/docs/reports/postanovka-2026-10-19/clef.json"
REF24="$(dirname "$REPO")/state/acceptance/window24b-20261030"
PORT="${AB_PORT:-18493}"
URL=http://127.0.0.1:$PORT
FUZZ_O="${FUZZ_O:-420}"
FUZZ_N="${FUZZ_N:-720}"
mkdir -p "$OUT/backup"
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
    if [ -n "${3:-}" ] && [ $t -ge 60 ] && ! systemctl --user is-active -q "$3"; then return 1; fi
  done
}
stop_ab() {
  [ -n "$AB" ] && systemctl --user stop "$AB" 2>/dev/null
  AB=""
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
}
start_live() {
  systemctl --user reset-failed "$LIVE" 2>/dev/null
  systemctl --user start "$LIVE"
  systemctl --user start "$OBS"
}
check_live() {   # метка S0 -> 0 чисто
  local S0=$2
  if ! wait_health http://127.0.0.1:8085 1500 "$LIVE"; then log "$1: боевой не ответил за 1500 с"; return 1; fi
  log "$1: боевой отвечает: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
  sleep 300
  local F E5 NR
  F=$(tail -n +$S0 "$LL" | grep -ac 'EngineCore encountered a fatal error')
  E5=$(tail -n +$S0 "$LL" | grep -ac '500 Internal Server Error')
  NR=$(systemctl --user show $LIVE -p NRestarts --value)
  log "$1 через 5 мин: падений $F, 500: $E5, 200: $(tail -n +$S0 "$LL" | grep -ac '" 200 OK'), NRestarts $NR, Traceback: $(tail -n +$S0 "$LL" | grep -ac Traceback)"
  [ "$F" -eq 0 ] && [ "${NR:-0}" -eq 0 ] && [ "$E5" -eq 0 ]
}
DEPLOYED=0
finish() {
  stop_ab
  [ "$DEPLOYED" = 1 ] && return
  log "боевой в прежнем профиле"
  local S0; S0=$(wc -l < "$LL")
  start_live
  check_live "прежний профиль" "$S0" || log "ВНИМАНИЕ: прежний профиль не чист -- смотреть журнал"
}
trap finish EXIT
trap 'exit 143' TERM INT   # остановка окна сигналом -- тоже через finish (боевой поднимается)
start_ab() {   # имя env...
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
  local T0; T0=$(date +%s)
  if ! wait_health $URL 3600 "$AB"; then
    log "A/B $name не поднялся"; tail -60 "$OUT/server-$name.log" >> "$OUT/window.log"; return 1
  fi
  log "A/B $name готов за $(( $(date +%s) - T0 )) с, прогрев 30 с"; sleep 30
}
crashed() { ! curl -sf -m 5 "$URL/health" >/dev/null || grep -aqE "illegal memory access|EngineDeadError|CUDA error|CUDA out of memory" "$OUT/server-$1.log"; }
fuzz() {   # имя секунд
  "$PYV" "$REPO/scripts/leak_fuzz.py" "$URL" "$OUT/fuzz-$1.json" "$2" "$OUT/server-$1.log" > "$OUT/fuzz-$1.log" 2>&1
  log "фаззер [$1] $2 с: код $?"
  grep -a '^ФАЗЗ' "$OUT/fuzz-$1.log" | sed "s/^/  [$1] /" | tee -a "$OUT/window.log"
  local G; G=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/server-$1.log" | grep -a 'TP0')
  log "графы MTP [$1]: $(echo "$G" | grep -aoE '\[MTP ГРАФ (ЦИКЛА|ТЕЛА)\] пул .*$' | sort -u | tr '\n' ';') выселений (печать 1-3 и каждого сотого): цикл $(echo "$G" | grep -ac 'ГРАФ ЦИКЛА\] выселен'), тело $(echo "$G" | grep -ac 'ГРАФ ТЕЛА\] выселен'); последнее: $(echo "$G" | grep -aoE 'выселений [0-9]+' | tail -1)"
}

echo "1024:0:4" > "$OUT/budget"
cp -a "$LIVE_SH" "$OUT/backup/"
log "окно 31: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

E="FA2SM80_W8_MARLIN=1 FA2SM80_BLOCKPIPE_BUFS=1 FA2SM80_BLOCKPIPE_M8_MIN=256 FA2SM70_GDN_IDX_VIEW=1 FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_CONV_META_PINNED=1 FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1 FA2SM80_M8DQ=42 FA2SM80_CG_SIZES=1,2,4,8,12,16,20,24,28,32,36,40,45 FA2SM80_MTP_LOOP_GRAPH_MAXB=9 FA2SM80_MTP_MLP_ROWS=1024"
FIX="FA2SM80_MTP_LOOP_ONE_STREAM=1 FA2SM80_MTP_LOOP_GRAPH_MAX=16 FA2SM70_MTP_GRAPH_MAX=16"

# 1) O -- профиль боевого как есть: воспроизводит ли фаззер рост пулов
# shellcheck disable=SC2086
if start_ab O $E; then
  fuzz O "$FUZZ_O"
  crashed O && log "O: УПАЛ ($(grep -aoE 'illegal memory access|EngineDeadError|CUDA error[^.]*|CUDA out of memory' "$OUT/server-O.log" | head -1))"
fi
stop_ab

# 2) N -- один поток захвата + пределы графов; tracemalloc только в ядре движка
# shellcheck disable=SC2086
start_ab N $E $FIX FA2SM80_MEMTRACE_TM=1 FA2SM80_MEMTRACE_TM_ONLY=EngineCore FA2SM80_MEMTRACE_TM_BASE=1 || exit 1
SL="$OUT/server-N.log"
fuzz N "$FUZZ_N"
if crashed N; then log "N: УПАЛ ($(grep -aoE 'illegal memory access|EngineDeadError|CUDA error[^.]*|CUDA out of memory' "$SL" | head -1))"; exit 1; fi
log "tracemalloc ядра движка (последний замер, прирост от первого снимка):"
sed 's/\x1b\[[0-9;]*m//g' "$SL" | grep -a 'EngineCore' | grep -a 'MEMTRACE\]   tm ' | tail -12 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"
"$PYV" "$PS" clef "$PORT" "$OUT/clef-N.json" > "$OUT/clef-N.log" 2>&1
CL1=$("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-N.json" 2>&1 | tail -1)
CL2=$("$PYV" "$PS" clefsr "$REF24/clef-base.json" "$OUT/clef-N.json" 2>&1 | tail -1)
log "Clef [N] против V100: $CL1"
log "Clef [N] против base 24b: $CL2"
(cd "$BENCH" && URL=$URL OUT="$OUT/q-N.json" REF="$REF24/q-base.json" "$PYV" quality_ref.py) > "$OUT/q-N.log" 2>&1
log "отпечаток LLM [N против base 24b]: $(grep -E '^L=' "$OUT/q-N.log" | cut -c1-90 | tr '\n' ' ')"
if crashed N; then log "N: УПАЛ после проверок"; exit 1; fi
grep -a "MEMTRACE\] .*GPU[01]" "$SL" | tail -4 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"

# 3) решение
ok=1
VM=$("$PYV" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['журнал'].get('вердикт_памяти'), '|', d['вердикт_запросов'])" "$OUT/fuzz-N.json" 2>&1)
log "вердикт фаззера [N]: $VM"
echo "$VM" | grep -q '^чисто | чисто$' || { ok=0; log "фаззер N не чист"; }
NFAIL=$(grep -ac 'отказ захвата' "$SL")
[ "$NFAIL" -eq 0 ] || { ok=0; log "отказов захвата графов: $NFAIL"; }
grep -aq 'поток захвата один, предел 16' "$SL" || { ok=0; log "один поток захвата не включился"; }
echo "$CL1" | grep -qE "34/35|35/35" || { ok=0; log "Clef против V100 не как прежде"; }
[ "$(grep -cE '^L=[0-9]+: совпало 160/160' "$OUT/q-N.log")" -eq 3 ] || { ok=0; log "отпечаток LLM: не 160/160 на всех длинах"; }
if [ $ok -ne 1 ]; then log "N в боевой НЕ идёт"; exit 0; fi
stop_ab
"$PYV" - "$LIVE_SH" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
якорь = 'export VLLM_CACHE_ROOT='
assert s.count(якорь) == 1 and "FA2SM80_MTP_LOOP_ONE_STREAM" not in s and "FA2SM70_MTP_GRAPH_MAX" not in s
блок = """# [ПАМЯТЬ 01.11] Окно 31: графы черновика MTP -- один поток захвата графов цикла (общий пул переиспользует память
# прежних захватов) и пределы LRU числа графов цикла и тела. 01.11 15:47 CUDA OOM на префилле 8192: пулы графов
# черновика за 14 ч выросли 92 -> 496 МиБ (каждый захват на новом потоке). Откат: убрать три строки и перезапустить.
export FA2SM80_MTP_LOOP_ONE_STREAM=1
export FA2SM80_MTP_LOOP_GRAPH_MAX=16
export FA2SM70_MTP_GRAPH_MAX=16
"""
open(p, "w").write(s.replace(якорь, блок + якорь))
EOF
S0=$(wc -l < "$LL")
log "в боевой: один поток захвата графов цикла, пределы графов 16"
start_live
if check_live "боевой с пределами графов" "$S0"; then
  DEPLOYED=1
  log "графы MTP в боевом: $(tail -n +$S0 "$LL" | sed 's/\x1b\[[0-9;]*m//g' | grep -a 'TP0' | grep -aoE '\[MTP ГРАФ (ЦИКЛА|ТЕЛА)\] пул .*$' | sort -u | tr '\n' ';')"
  log "окно 31: пределы графов черновика в бою"
else
  log "ОТКАТ профиля боевого"
  systemctl --user stop "$OBS" "$LIVE"
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
  cp -a "$OUT/backup/serve_candidate_live.sh" "$LIVE_SH"
fi
