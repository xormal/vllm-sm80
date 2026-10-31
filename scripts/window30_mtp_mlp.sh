#!/usr/bin/env bash
# [ПАМЯТЬ 31.10] Окно 30 -- MLP головы MTP по кускам строк (FA2SM80_MTP_MLP_ROWS=1024, fa2sm70_mtp_w8.py; владелец
# 01.11: «делай защиту от OOM, боевой можно остановить»). 31.10 19:30 боевой упал OOM в черновике на шаге 8192 токенов:
# gate_up головы [M, 17408] + SiLU [M, 8704] = 408 МиБ сверх прочего, не хватило 6 МиБ.
#   0) стенд: нарезка, SiLU против inductor побитно, пик gate_up + SiLU целиком и кусками;
#   1) A/B M = боевой профиль + FA2SM80_MTP_MLP_ROWS=1024 + сверка первых 4 резаных вызовов с целым проходом;
#   2) точность: сверка кусков побитно, Clef против V100 и base 24b, отпечаток LLM против base 24b;
#   3) сочетания 9:8 и 16:8: префилл 8192 токенов под Clef -- тот шаг, на котором упали; память;
#   4) всё чисто -> в боевой (строка в скрипте + перезапуск, проверка 5 мин, откат при сбое); иначе -- прежний профиль.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
MEMRUN="$(dirname "$REPO")/../sm80_work/bin/memrun"
LIVE_SH="$REPO/scripts/serve_candidate_live.sh"
LL="$(dirname "$REPO")/state/acceptance/live-20261029/server.log"
PS="$REPO/sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
BENCH="$(dirname "$REPO")/sm80_work/bench"
REF_CLEF="$(dirname "$REPO")/docs/reports/postanovka-2026-10-19/clef.json"
REF24="$(dirname "$REPO")/state/acceptance/window24b-20261030"
PORT="${AB_PORT:-18493}"
URL=http://127.0.0.1:$PORT
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

echo "1024:0:4" > "$OUT/budget"
cp -a "$LIVE_SH" "$OUT/backup/"
log "окно 30: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 0) стенд на одной карте
CUDA_VISIBLE_DEVICES=0 "$MEMRUN" 10G "$PYV" "$REPO/scripts/mtp_mlp_rows_test.py" > "$OUT/unit.log" 2>&1
UNIT=$?
log "стенд: код $UNIT: $(grep -aE '^(нарезка|SiLU|пик)' "$OUT/unit.log" | tr '\n' ';' | cut -c1-400)"
[ "$UNIT" -eq 0 ] || { log "стенд не прошёл -- в боевой не идёт"; exit 1; }

# 1) A/B M -- профиль боевого + MLP головы по кускам
E="FA2SM80_W8_MARLIN=1 FA2SM80_BLOCKPIPE_BUFS=1 FA2SM80_BLOCKPIPE_M8_MIN=256 FA2SM70_GDN_IDX_VIEW=1 FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_CONV_META_PINNED=1 FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1 FA2SM80_M8DQ=42 FA2SM80_CG_SIZES=1,2,4,8,12,16,20,24,28,32,36,40,45 FA2SM80_MTP_LOOP_GRAPH_MAXB=9"
# shellcheck disable=SC2086
start_ab M $E FA2SM80_MTP_MLP_ROWS=1024 FA2SM80_MTP_MLP_CHECK=4 || exit 1
SL="$OUT/server-M.log"
log "установка: $(grep -ac 'MLP головы: gate_up + SiluAndMul одной операцией' "$SL") процессов; компиляция: $(sed 's/\x1b\[[0-9;]*m//g' "$SL" | grep -aoE 'Directly load[^,]*|Compiling a graph[^,]*|torch.compile takes [0-9.]+ s[^,]*' | sort | uniq -c | tr '\n' ';')"
log "кэш графа головы: $(sed 's/\x1b\[[0-9;]*m//g' "$SL" | grep -aoE 'Using cache directory: [^ ]*eagle_head' | head -1 | sed 's/^.*acceptance\///')"
log "захват: $(sed 's/\x1b\[[0-9;]*m//g' "$SL" | grep -aoE 'Capturing CUDA graphs \([a-z -]+, [A-Z]+\): +100%[^|]*\|[^|]*\| *[0-9]+/[0-9]+' | sed -E 's/: +100%.*\| *([0-9]+\/[0-9]+)/ \1/' | sort | uniq -c | tr '\n' ';')"

# 2) точность
"$PYV" "$PS" clef "$PORT" "$OUT/clef-M.json" > "$OUT/clef-M.log" 2>&1
CL1=$("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-M.json" 2>&1 | tail -1)
CL2=$("$PYV" "$PS" clefsr "$REF24/clef-base.json" "$OUT/clef-M.json" 2>&1 | tail -1)
log "Clef [M] против V100: $CL1"
log "Clef [M] против base 24b: $CL2"
(cd "$BENCH" && URL=$URL OUT="$OUT/q-M.json" REF="$REF24/q-base.json" "$PYV" quality_ref.py) > "$OUT/q-M.log" 2>&1
QM=$(grep -E '^L=' "$OUT/q-M.log" | cut -c1-90 | tr '\n' ' ')
log "отпечаток LLM [M против base 24b]: $QM"
NCHK=$(grep -ac 'сверка MLP головы по кускам' "$SL")
NBAD=$(grep -a 'сверка MLP головы по кускам' "$SL" | grep -avc 'различий 0 из')
NREAL=$(grep -a 'сверка MLP головы по кускам' "$SL" | grep -ac 'строки разные')
log "сверка кусков с целым проходом: $NCHK (на живых строках $NREAL), с различиями $NBAD: $(grep -a 'сверка MLP головы по кускам' "$SL" | grep -a 'строки разные' | sed 's/^.*сверка/сверка/' | head -2 | tr '\n' ';')"

# 3) нагрузка: префилл 8192 под Clef и декод под Clef
CONC=1 PARTS=combo COMBOS=9:8,16:8 "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/combo-M.json" M >> "$OUT/window.log" 2>&1
log "сочетания 9:8, 16:8 (бюджет 1024): код $?"
if crashed M; then log "M: УПАЛ ($(grep -aoE 'illegal memory access|EngineDeadError|CUDA error[^.]*|CUDA out of memory' "$SL" | head -1))"; exit 1; fi
log "MLP головы по кускам: $(grep -a 'MLP головы по кускам работает' "$SL" | sed 's/^.*работает/работает/' | tail -1); отказов памяти на куске: $(grep -ac 'отказ памяти на куске' "$SL")"
grep -a "MEMTRACE\] .*GPU[01]:" "$SL" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"

# 4) решение
ok=1
echo "$CL1" | grep -qE "34/35|35/35" || { ok=0; log "Clef против V100 не как прежде"; }
[ "$(grep -cE '^L=[0-9]+: совпало 160/160' "$OUT/q-M.log")" -eq 3 ] || { ok=0; log "отпечаток LLM: не 160/160 на всех длинах"; }
[ "$NREAL" -ge 1 ] && [ "$NBAD" -eq 0 ] || { ok=0; log "сверка кусков: нет сверок на живых строках или есть различия"; }
grep -aq 'MLP головы по кускам работает' "$SL" || { ok=0; log "резаный путь не сработал ни разу"; }
if [ $ok -ne 1 ]; then log "M в боевой НЕ идёт"; exit 0; fi
stop_ab
"$PYV" - "$LIVE_SH" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
якорь = 'export VLLM_CACHE_ROOT='
assert s.count(якорь) == 1 and "FA2SM80_MTP_MLP_ROWS" not in s
блок = """# [ПАМЯТЬ 01.11] Окно 30: MLP головы MTP по кускам строк (fa2sm70_mtp_w8.py) -- 31.10 19:30 OOM в черновике на шаге
# 8192 токенов (gate_up [M, 17408] + SiLU [M, 8704] = 408 МиБ); кусками пик -- выход + кусок (~170 МиБ), выход побитно тот же.
# Откат: убрать строку и перезапустить.
export FA2SM80_MTP_MLP_ROWS=1024
"""
open(p, "w").write(s.replace(якорь, блок + якорь))
EOF
S0=$(wc -l < "$LL")
log "в боевой: MLP головы по кускам"
start_live
if check_live "боевой с MLP головы по кускам" "$S0"; then
  DEPLOYED=1
  log "установка в боевом: $(tail -n +$S0 "$LL" | grep -ac 'MLP головы: gate_up + SiluAndMul одной операцией') процессов; резаный путь: $(tail -n +$S0 "$LL" | grep -a 'MLP головы по кускам работает' | sed 's/^.*работает/работает/' | tail -1)"
  log "окно 30: MLP головы по кускам в бою"
else
  log "ОТКАТ профиля боевого"
  systemctl --user stop "$OBS" "$LIVE"
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
  cp -a "$OUT/backup/serve_candidate_live.sh" "$LIVE_SH"
fi
