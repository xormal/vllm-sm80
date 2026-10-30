#!/usr/bin/env bash
# [ФРИЗЫ 30.10] Окно 28 -- деквант Marlin v4 и выкатка E с бюджетом 1024 на боевой (владелец 30.10: «меряй деквант,
# если он лучше, то его, иначе E-1024 ставь»; «деквант ультраоптимизируй и сверь sass»).
#   1) marlin8_sm80.cu + marlin8_dequant.cuh из рабочего дерева в основное, сборка marlin8 в боевом каталоге; провал
#      сборки -- прежние исходник и сборка из копии (v2), выкатка всё равно;
#   2) стенд вариантов декванта (2 -- прежний, 41/42 -- v4 по 128/256 k): побитно против нашего (обычные и крошечные
#      масштабы) и время; лучший побитный -> FA2SM80_M8DQ (ни одного -> 2);
#   3) скрипт боевого: E-флаги (окна 26-27) + бюджет 1024 + FA2SM80_M8DQ; запуск, проверка 5 мин; не поднялся или
#      падения -- откат скрипта, исходника и сборки и прежний боевой.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
WT="${WT:?рабочее дерево с правками}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
PYV="$REPO/.venv/bin/python"
BD="$(dirname "$REPO")/sm80_work/build_sm80"
MEMRUN="$(dirname "$REPO")/../sm80_work/bin/memrun"
LIVE_SH="$REPO/scripts/serve_candidate_live.sh"
LL="$(dirname "$REPO")/state/acceptance/live-20261029/server.log"
CU=sm70/fa2_sm70/csrc/marlin8_sm80.cu
CUH=sm70/fa2_sm70/csrc/marlin8_dequant.cuh
BK="$OUT/backup"
mkdir -p "$BK"
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
restore_code() {
  cp -a "$BK/marlin8_sm80.cu" "$REPO/$CU"
  rm -f "$REPO/$CUH"
  rm -rf "$BD/fa2sm80_marlin8" && cp -a "$BK/build_marlin8" "$BD/fa2sm80_marlin8"
  log "исходник и сборка marlin8 -- из копии (v2)"
}
start_live() {
  systemctl --user reset-failed "$LIVE" 2>/dev/null
  systemctl --user start "$LIVE"
  systemctl --user start "$OBS"
}
rollback_all() {
  log "ОТКАТ: $1"
  systemctl --user stop "$OBS" "$LIVE"
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
  cp -a "$BK/serve_candidate_live.sh" "$LIVE_SH"
  restore_code
  start_live
  if wait_health http://127.0.0.1:8085 1500 "$LIVE"; then
    log "боевой в прежнем профиле отвечает: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
  else
    log "ВНИМАНИЕ: прежний боевой не ответил за 1500 с: $(systemctl --user show $LIVE -p ActiveState -p SubState | tr '\n' ' ')"
  fi
  exit 1
}

cp -a "$LIVE_SH" "$BK/"
cp -a "$REPO/$CU" "$BK/"
cp -a "$BD/fa2sm80_marlin8" "$BK/build_marlin8"
log "окно 28: копии -- $BK; стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
if ! gpus_free; then log "GPU0-1 не освободились -- боевой обратно"; start_live; exit 1; fi

# 1) новый деквант
cp "$WT/$CU" "$REPO/$CU"
cp "$WT/$CUH" "$REPO/$CUH"
T0=$(date +%s)
/usr/bin/python3 -I "$REPO/scripts/candidate_build.py" -- env FA2SM70_BUILD_DIR="$BD" FA2SM70_PATH="$REPO/sm70" \
  PYTHONPATH="$REPO/sm70" "$PYV" -c "
import fa2_sm70._ext as E
m = E.marlin8_ext(); print('marlin8', hasattr(m, 'marlin8_dequant'))
" >> "$OUT/build.log" 2>&1
RC=$?
log "сборка marlin8 (v4): код $RC за $(( $(date +%s) - T0 )) с: $(tail -1 "$OUT/build.log" | cut -c1-100)"
M8DQ=""
if [ $RC -ne 0 ]; then
  restore_code
else
  # 2) стенд вариантов (флаги сборки -- как у сервера: sm_80, без пересборки)
  TORCH_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=0 FA2SM70_BUILD_DIR="$BD" FA2SM70_PATH="$REPO/sm70" \
    "$MEMRUN" 10G "$PYV" "$REPO/scripts/marlin8_dequant_bench.py" "$OUT/dq.json" > "$OUT/dq.log" 2>&1
  log "стенд декванта: код $?"
  grep -E "^(gate_up|down|qkvz|out) |деквант всех" "$OUT/dq.log" | cut -c1-400 | tee -a "$OUT/window.log"
  M8DQ=$("$PYV" -c "import json; v = json.load(open('$OUT/dq.json')).get('лучший'); print(v if v else 2)" 2>/dev/null || echo 2)
  ls -la --time-style=+%T "$BD/fa2sm80_marlin8/" | grep -E "\.so$" | tee -a "$OUT/window.log"
fi
log "деквант: FA2SM80_M8DQ=${M8DQ:-нет (прежняя сборка v2)}"

# 3) профиль боевого: E + бюджет 1024
M8DQ="$M8DQ" "$PYV" - "$LIVE_SH" <<'EOF'
import os, sys
p = sys.argv[1]
s = open(p).read()
old = "export FA2SM80_PREFILL_BUDGET_DECODING=2048 FA2SM80_PREFILL_BUDGET_BACKLOG=0"
assert s.count(old) == 1, "строка бюджета не найдена"
s = s.replace(old, "export FA2SM80_PREFILL_BUDGET_DECODING=1024 FA2SM80_PREFILL_BUDGET_BACKLOG=0")
m8 = os.environ.get("M8DQ", "")
блок = """# [ФРИЗЫ 30.10] Решение владельца после окон 26-28 («меряй деквант, если он лучше, то его, иначе E-1024 ставь»):
# GEMM весов -- Marlin8 (M <= 1024; больше -- деквант из раскладки Marlin + cuBLAS), конвейер блока с постоянными
# буферами и на средних M, без синхронизаций построителя GDN и внимания; бюджет prefill при декоде -- 1024 (выше).
# Окна 26-27: чаты под Clef 33 / 25-27 ток/с на поток (было 20 / 17), пауза p99 0.35-0.45 с (было 0.58-0.60), Clef в
# смеси -12..-23 %; точность: Clef 34/35 против V100, LLM 160/160, длинные 50/50. Откат: бюджет 2048, убрать блок,
# перезапустить (прежний скрипт -- state/acceptance/window28-20261030/backup).
export FA2SM80_W8_MARLIN=1
export FA2SM80_BLOCKPIPE_BUFS=1 FA2SM80_BLOCKPIPE_M8_MIN=256
export FA2SM70_GDN_IDX_VIEW=1 FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_CONV_META_PINNED=1
export FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1
"""
if m8:
    блок += f"# деквант Marlin -- вариант {m8} (окно 28: стенд marlin8_dequant_bench.py, побитно как наш)\nexport FA2SM80_M8DQ={m8}\n"
якорь = 'export VLLM_CACHE_ROOT='
assert s.count(якорь) == 1
s = s.replace(якорь, блок + якорь)
open(p, "w").write(s)
print("профиль боевого обновлён")
EOF
[ $? -eq 0 ] || rollback_all "правка скрипта боевого не прошла"
log "профиль: $(grep -cE '^export (FA2SM80_W8_MARLIN|FA2SM80_BLOCKPIPE_BUFS|FA2SM70_GDN_IDX_VIEW|FA2SM80_GDN_DEVMETA|FA2SM80_M8DQ)=' "$LIVE_SH") строк E, $(grep -o 'BUDGET_DECODING=[0-9]*' "$LIVE_SH")"

S0=$(wc -l < "$LL")
log "старт боевого с E-1024"
start_live
if ! wait_health http://127.0.0.1:8085 1500 "$LIVE"; then
  tail -n +$S0 "$LL" | grep -aE "Error|error|Traceback" | tail -20 >> "$OUT/window.log"
  rollback_all "боевой с E-1024 не ответил за 1500 с"
fi
log "боевой с E-1024 отвечает: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
log "в журнале: $(tail -n +$S0 "$LL" | grep -ao 'Marlin: перепаковано[^|]*' | head -1); чередование: $(tail -n +$S0 "$LL" | grep -ac 'ЧЕРЕДОВАНИЕ')"
sleep 300
F=$(tail -n +$S0 "$LL" | grep -ac 'EngineCore encountered a fatal error')
E5=$(tail -n +$S0 "$LL" | grep -ac '500 Internal Server Error')
NR=$(systemctl --user show $LIVE -p NRestarts --value)
log "боевой E-1024 через 5 мин: падений $F, 500: $E5, 200: $(tail -n +$S0 "$LL" | grep -ac '" 200 OK'), NRestarts $NR, Traceback: $(tail -n +$S0 "$LL" | grep -ac Traceback)"
if [ "$F" -gt 0 ] || [ "${NR:-0}" -gt 0 ] || [ "$E5" -gt 0 ]; then
  rollback_all "падения или 500 в первые 5 мин"
fi
log "окно 28: E-1024 в бою"
