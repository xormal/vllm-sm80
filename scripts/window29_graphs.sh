#!/usr/bin/env bash
# [ГРАФЫ 31.10] Окно 29 -- полные графы декода и для 9 запросов (владелец 30.10: «сверь, что полный граф, наши ядра, все
# пути, все в граф встает, если что-то не так, то чини»). Окно 27: размеры FA2SM80_CG_SIZES (1..40) при k=4 округляются
# до кратных 5 -> полные графы декода на 1..8 запросов; пул GDN пускает 9 -- шаг декода с 9 запросами eager (81.6 против
# 46 мс). Граф цикла черновика MTP -- до FA2SM80_MTP_LOOP_GRAPH_MAXB=8.
#   0) стенд декванта (предел нашей схемы трафика -- приведение int8 -> fp16);
#   1) A/B G = боевой профиль (E-1024, деквант v4) + размер 44 (-> 45) + MAXB 9; размеры захвата -- в хэше конфига
#      компиляции: холодная компиляция при первом старте (кэш общий с боевым);
#   2) захват: полных графов декода 9; Clef против V100 и base 24b, отпечаток LLM против base 24b;
#   3) декод c=8/9/16, сочетания 9:8 и 16:8 (бюджет 1024), шаги чистого декода по числу запросов, память;
#   4) устоял, точность как прежде, шаг с 9 запросами < 60 мс -> в боевой (скрипт + перезапуск, проверка 5 мин, откат при
#      сбое); иначе -- боевой в прежнем профиле.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
BD="$(dirname "$REPO")/sm80_work/build_sm80"
MEMRUN="$(dirname "$REPO")/../sm80_work/bin/memrun"
LIVE_SH="$REPO/scripts/serve_candidate_live.sh"
LL="$(dirname "$REPO")/state/acceptance/live-20261029/server.log"
PS="$REPO/sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
BENCH="$(dirname "$REPO")/sm80_work/bench"
REF_CLEF="$(dirname "$REPO")/docs/reports/postanovka-2026-10-19/clef.json"
REF24="$(dirname "$REPO")/state/acceptance/window24b-20261030"
PORT="${AB_PORT:-18493}"
URL=http://127.0.0.1:$PORT
# 45, а не 44: vLLM округляет размер ВВЕРХ до кратного k+1 и отбрасывает, если больше наибольшего размера списка
# (окно 29: 44 -> 45 > 44 -> выброшен, графов осталось 8)
CG_NEW="1,2,4,8,12,16,20,24,28,32,36,40,45"
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
log "окно 29: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 0) стенд декванта (флаги как у сервера: sm_80, CC/CXX -- без пересборки); SKIP_DQ=1 -- уже снят
[ "${SKIP_DQ:-0}" = 1 ] || TORCH_CUDA_ARCH_LIST=8.0 CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDA_VISIBLE_DEVICES=0 FA2SM70_BUILD_DIR="$BD" \
  FA2SM70_PATH="$REPO/sm70" "$MEMRUN" 10G "$PYV" "$REPO/scripts/marlin8_dequant_bench.py" "$OUT/dq.json" > "$OUT/dq.log" 2>&1
log "стенд декванта: код $?: $(grep 'деквант всех' "$OUT/dq.log" | cut -c1-300)"

# 1) A/B G -- профиль боевого (E-1024, v4) + графы на 9 запросов
E="FA2SM80_W8_MARLIN=1 FA2SM80_BLOCKPIPE_BUFS=1 FA2SM80_BLOCKPIPE_M8_MIN=256 FA2SM70_GDN_IDX_VIEW=1 FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_CONV_META_PINNED=1 FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1 FA2SM80_M8DQ=42"
# shellcheck disable=SC2086
start_ab G $E FA2SM80_CG_SIZES=$CG_NEW FA2SM80_MTP_LOOP_GRAPH_MAXB=9 || exit 1
log "захват: $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/server-G.log" | grep -aoE 'Capturing CUDA graphs \([a-z -]+, [A-Z]+\): +100%[^|]*\|[^|]*\| *[0-9]+/[0-9]+' | sed -E 's/: +100%.*\| *([0-9]+\/[0-9]+)/ \1/' | sort | uniq -c | tr '\n' ';')"
log "компиляция: $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/server-G.log" | grep -aoE 'Directly load[^,]*|Compiling a graph[^,]*|torch.compile takes [0-9.]+ s[^,]*' | sort | uniq -c | tr '\n' ';')"

# 2) точность
"$PYV" "$PS" clef "$PORT" "$OUT/clef-G.json" > "$OUT/clef-G.log" 2>&1
CL1=$("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-G.json" 2>&1 | tail -1)
CL2=$("$PYV" "$PS" clefsr "$REF24/clef-base.json" "$OUT/clef-G.json" 2>&1 | tail -1)
log "Clef [G] против V100: $CL1"
log "Clef [G] против base 24b: $CL2"
(cd "$BENCH" && URL=$URL OUT="$OUT/q-G.json" REF="$REF24/q-base.json" "$PYV" quality_ref.py) > "$OUT/q-G.log" 2>&1
QG=$(grep -E '^L=' "$OUT/q-G.log" | cut -c1-90 | tr '\n' ' ')
log "отпечаток LLM [G против base 24b]: $QG"

# 3) нагрузка
L0=$(wc -l < "$OUT/steps-G.jsonl" 2>/dev/null || echo 0)
CONC=8,9,16 PARTS=gen "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/gen-G.json" G >> "$OUT/window.log" 2>&1
log "декод c=8/9/16: код $?"
CONC=1 PARTS=combo COMBOS=9:8,16:8 "$PYV" "$REPO/scripts/bench_conc.py" "$URL" "$OUT/combo-G.json" G >> "$OUT/window.log" 2>&1
log "сочетания 9:8, 16:8 (бюджет 1024): код $?"
if crashed G; then log "G: УПАЛ ($(grep -aoE 'illegal memory access|EngineDeadError|CUDA error[^.]*|CUDA out of memory' "$OUT/server-G.log" | head -1))"; exit 1; fi
D9=$("$PYV" - "$OUT/steps-G.jsonl" "$L0" <<'EOF'
import json, sys, statistics as st
R = [json.loads(x) for x in open(sys.argv[1]) if x.strip()][int(sys.argv[2]):]
д = {}
for r in R:
    if r.get("dt") and r["dec"] > 0 and r["pf"] == 0 and r["clef"] == 0 and r["dt"] < 2:
        д.setdefault(r["dec"], []).append(r["dt"])
open(sys.argv[1] + ".декод.txt", "w").write(" ".join(f"{k}:{st.median(v)*1e3:.1f}мс(n={len(v)})" for k, v in sorted(д.items()) if len(v) >= 5))
v = д.get(9, [])
print(round(st.median(v) * 1e3, 1) if len(v) >= 5 else "нет")
EOF
)
log "шаг чистого декода с 9 запросами: $D9 мс (окно 27: 81.6); по числу запросов: $(cat "$OUT/steps-G.jsonl.декод.txt" 2>/dev/null)"
grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-G.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"

# 4) решение
ok=1
echo "$CL1" | grep -qE "34/35|35/35" || { ok=0; log "Clef против V100 не как прежде"; }
[ "$(grep -cE '^L=[0-9]+: совпало 160/160' "$OUT/q-G.log")" -eq 3 ] || { ok=0; log "отпечаток LLM: не 160/160 на всех длинах"; }
[ "$D9" != "нет" ] && awk "BEGIN{exit !($D9 < 60)}" || { ok=0; log "шаг с 9 запросами не ускорился ($D9)"; }
if [ $ok -ne 1 ]; then log "G в боевой НЕ идёт"; exit 0; fi
stop_ab
"$PYV" - "$LIVE_SH" "$CG_NEW" <<'EOF'
import sys
p, cg = sys.argv[1], sys.argv[2]
s = open(p).read()
якорь = 'export VLLM_CACHE_ROOT='
assert s.count(якорь) == 1 and "FA2SM80_CG_SIZES" not in s
блок = f"""# [ГРАФЫ 31.10] Окно 29: полные графы декода и для 9 запросов (пул GDN пускает 9; размеры при k=4 -- кратные 5, было
# до 40 токенов = 8 запросов, шаг с 9 запросами шёл eager: 81.6 против 46 мс), граф цикла черновика MTP -- до 9.
# Откат: убрать две строки и перезапустить (размеры захвата -- в хэше компиляции: прежний кэш остаётся).
export FA2SM80_CG_SIZES={cg}
export FA2SM80_MTP_LOOP_GRAPH_MAXB=9
"""
open(p, "w").write(s.replace(якорь, блок + якорь))
EOF
S0=$(wc -l < "$LL")
log "в боевой: графы на 9 запросов"
start_live
if check_live "боевой с графами на 9" "$S0"; then
  DEPLOYED=1
  log "окно 29: графы на 9 запросов в бою"
else
  log "ОТКАТ профиля боевого"
  systemctl --user stop "$OBS" "$LIVE"
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
  cp -a "$OUT/backup/serve_candidate_live.sh" "$LIVE_SH"
fi
