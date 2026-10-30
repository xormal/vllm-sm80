#!/usr/bin/env bash
# [GEMM 30.10] Окно 15 -- ТОЛЬКО С ОТДЕЛЬНОГО РАЗРЕШЕНИЯ ВЛАДЕЛЬЦА (владелец 30.10: «надо добиваться единого графа и делать
# GEMM; подготовь все, но боевой трогать с отдельного разрешения»).
# W8A16 v3 (w8a16_sm80_gemm3: x в порядке фрагментов, плитка варпа 64x64; SASS: 481 инструкция на 128 HMMA против 846 у v2,
# MOV 13 против 299) -- обычный маршрут M > 64 и конвейер блока, под переключателем-файлом $OUT/g3 ("0"/"1").
#  0) правки из рабочего дерева $WT -> основное дерево ТОЛЬКО при остановленном боевом (сверка с базой $BASE_TAR);
#     сборка w8a16_sm80 -- через candidate_build.py (лимиты сборки), до любого запуска на GPU;
#  1) GPU0: scripts/w8_g3_test.py -- v3 против прямого ядра побитно, срезы, без синхронизаций, граф, время по формам;
#     если v3 нигде не быстрее прямого / деквант+GEMM -- A/B не запускается (окно короче);
#  2) A/B на 127.0.0.1:$AB_PORT с конфигурацией боевого (бюджет 2048:0, асинхронная голова, уровень 2); подбор v3 при
#     загрузке пишет кэш state/w8tune/*_v2g3.json; сверка Clef / LLM / dlinnye: g3 1 против 0;
#  3) профиль без стеков при 0 и 1; живая смесь (насыщенная очередь Clef + 8 чатов) фазами 0/1.
# Боевой ВСЕГДА поднимается (trap) при RESTORE=1, затем 5 минут живой нагрузки без смертей движка. Только GPU0-1.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?OUTDIR}"
LIVE=kandidat-acceptance-live-20261029.service
OBS=kandidat-live-observer.service
AB=""
PYV="$REPO/.venv/bin/python"
PS="$REPO/sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
BENCH="$(dirname "$REPO")/sm80_work/bench"
REF_CLEF="$(dirname "$REPO")/docs/reports/postanovka-2026-10-19/clef.json"
WT="${WT:?рабочее дерево с правками}"
BASE_TAR="${BASE_TAR:?база правленых файлов (tar)}"
# GEMM v3 + первый этап графа (поправка Clef по окну строк в графе, обмен копировщиком в графе, конвейеры в захвате
# has_clef) -- всё под флагами, без них инертно; окно 16 только запускает A/B графов
FILES="${FILES:-sm70/fa2_sm70/csrc/w8a16_sm80.cu sm70/integrations/vllm_sm70/fa2sm80_w8.py sm70/integrations/vllm_sm70/fa2sm80_blockpipe.py sm70/fa2_sm70/csrc/clef.cu sm70/fa2_sm70/csrc/exch_i12.cu sm70/integrations/vllm_sm70/fa2sm70_clef.py sm70/integrations/vllm_sm70/fa2sm70_exch_i12.py sm70/integrations/vllm_sm70/fa2sm80_rowpipe.py sm70/integrations/vllm_sm70/sitecustomize.py vllm/v1/worker/gpu_model_runner.py}"
NEW="${NEW:-sm70/fa2_sm70/csrc/w8a16_sm80_g3.cuh}"
PORT="${AB_PORT:-18493}"           # не 8085/8086: агенты-потребители не должны цепляться к A/B
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
  for i in $(seq 1 60); do gpus_free && break; sleep 2; done
}
restore() {
  stop_ab
  if [ "$RESTORE" != "1" ]; then log "A/B ($AB) погашен; боевой НЕ поднимаю (RESTORE=0)"; return; fi
  log "восстановление: A/B ($AB) погашен, поднимаю $LIVE"
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
start_ab() {   # имя-захода
  AB=kandidat-ab-$1-$(date +%H%M).service
  log "старт A/B $AB (конфигурация боевого; v3 -- переключатель-файл $OUT/g3)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM70_CLEF_HEAD_ASYNC=1 \
    --setenv=FA2SM70_NOSYNC_ACC2=1 --setenv=FA2SM70_MEM_TRIM_MODE=off \
    --setenv=FA2SM80_W8_G3_FILE="$OUT/g3" \
    --setenv=FA2SM80_MEMTRACE=20 --setenv=AB_PORT="$PORT" --setenv=FA2SM80_GC_LOG_MS=200 \
    --setenv=PROFILE_DIR="$OUT/profile" --setenv=PROFILE_STACK=false \
    "$REPO/scripts/serve_candidate_ab.sh"
  # первый старт: подбор v3 по формам при загрузке (минуты), затем профилирование памяти и захват графов
  if ! wait_health $URL 2400 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
}
gpu_summary() { grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
warm() {
  "$PYV" - "$URL" <<'PY'
import json, random, sys, urllib.request
for _ in range(3):
    ids = [random.randint(1000, 60000) for _ in range(900)]
    req = urllib.request.Request(sys.argv[1] + "/v1/completions", json.dumps({
        "model": "Qwen3.8-27B-ABLITERATED", "prompt": ids, "max_tokens": 1}).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
PY
}
ns() { printf "$1" > "$OUT/g3"; sleep 2; warm; log "g3: $1"; }
lm() {   # имя аргументы...
  "$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/$1" --url $URL \
    --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" "${@:2}" >> "$OUT/window.log" 2>&1
  log "живая смесь $1: код $?"
}
trap restore EXIT

printf 0 > "$OUT/g3"; echo "2048:0" > "$OUT/budget"
log "окно 15: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 0) правки -- только сейчас: боевой остановлен, ни один сервер их не импортирует
for f in $FILES; do
  if ! tar -xOf "$BASE_TAR" "$f" | cmp -s - "$REPO/$f"; then
    log "ОТКАЗ: $f в основном дереве отличается от базы -- правки не применены"; exit 1
  fi
done
for f in $NEW; do
  if [ -e "$REPO/$f" ] && ! cmp -s "$WT/$f" "$REPO/$f"; then log "ОТКАЗ: $f уже есть в основном дереве и другой"; exit 1; fi
done
for f in $FILES $NEW; do cp -a "$WT/$f" "$REPO/$f"; done
log "правки применены: $FILES $NEW"
"$REPO/scripts/candidate_build.py" -- "$PYV" -c "import fa2_sm70._ext as E; print(E.w8a16_sm80_ext().__file__, E.clef_ext().__file__, E.exch_ext().__file__)" \
  > "$OUT/build.log" 2>&1
rc=$?
log "сборка w8a16_sm80 + clef + exch: код $rc"
if [ $rc -ne 0 ]; then tail -30 "$OUT/build.log" >> "$OUT/window.log"; exit 1; fi

# 1) стенд v3 на GPU0
systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
  bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0; exec "$2" "$1/scripts/w8_g3_test.py" "$3"' \
  _ "$REPO" "$PYV" "$OUT/g3_test.json" > "$OUT/g3_test.log" 2>&1
rc=$?
log "стенд v3: код $rc"
grep -E "^(побитно|НЕ ПОБИТНО|срез|без синхронизаций|граф|GEMM-часть)" "$OUT/g3_test.log" | cut -c1-300 >> "$OUT/window.log"
grep -E "^время" "$OUT/g3_test.log" | "$PYV" -c "
import json, sys
for l in sys.stdin:
    r = json.loads(l[l.index('{'):])
    print(f\"{r['форма']:8s} M={r['M']:5d}  v3 {r['v3_TF']:6.1f} TF {r['v3_cfg']}  прямой {r['прямой_TF']:6.1f}  деквант+cublas {r['деквант+cublas_TF']:6.1f}  cublas {r['cublas_TF']:6.1f}  упаковка {r['упаковка_мкс']:.0f} мкс\")
" >> "$OUT/window.log" 2>&1
for i in $(seq 1 30); do gpus_free && break; sleep 2; done
# 1б) обмен копировщиком в CUDA-графе -- две карты
systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
  bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0,1 TORCH_CUDA_ARCH_LIST=8.0; exec timeout 900 "$2" "$1/scripts/exch_ce_graph_test.py" "$3"' \
  _ "$REPO" "$PYV" "$OUT/exch_graph.json" > "$OUT/exch_graph.log" 2>&1
log "стенд обмена в графе: код $? $(grep -a '^ранг' "$OUT/exch_graph.log" | cut -c1-260 | tr '\n' ' ')"
for i in $(seq 1 30); do gpus_free && break; sleep 2; done
if [ $rc -ne 0 ] || grep -q "^НЕ ПОБИТНО" "$OUT/g3_test.log"; then
  log "стенд не пройден -- A/B не запускаю"; exit 1
fi
выигрыш=$("$PYV" -c "
import json; r = json.load(open('$OUT/g3_test.json'))
print(sum(1 for x in r['время'] if x['v3_мкс'] < 0.97 * min(x['прямой_мкс'], x['деквант_мкс'] + x['cublas_мкс'])))")
log "v3 быстрее нынешнего маршрута в $выигрыш точках из $(grep -c '^время' "$OUT/g3_test.log")"
[ "${выигрыш:-0}" -gt 0 ] || { log "v3 нигде не быстрее -- A/B не запускаю"; exit 0; }

# 2) A/B: сверка 1 против 0
start_ab g3
grep -a "v3 K=" "$OUT/server-g3.log" | grep -a Worker_TP0 | sed 's/^.*\] v3/v3/' | cut -c1-200 >> "$OUT/window.log"
for m in 0 1 02; do
  ns "${m:0:1}"
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$m.json" > "$OUT/clef-$m.log" 2>&1
  log "Clef [g3 $m] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$m.json" 2>&1 | tail -1)"
  ref=""; [ "$m" != 0 ] && ref="$OUT/q-0.json"
  (cd "$BENCH" && URL=$URL OUT="$OUT/q-$m.json" REF="$ref" "$PYV" quality_ref.py) > "$OUT/q-$m.log" 2>&1
  log "отпечаток LLM [g3 $m]: $(grep -E '^L=' "$OUT/q-$m.log" | cut -c1-90 | tr '\n' ' ')"
done
log "Clef g3 1 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-1.json" 2>&1 | tail -1)"
log "Clef g3 0 повтор против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-02.json" 2>&1 | tail -1)"
ns 1
"$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye-1.json" 34816 > "$OUT/dlinnye-1.log" 2>&1
log "dlinnye [g3 1]: $(grep ИТОГ "$OUT/dlinnye-1.log")"
grep -aE "Traceback|Error" "$OUT/server-g3.log" | grep -av "INFO" | head -5 >> "$OUT/window.log"

# 3) профиль без стеков при 0 и 1
for m in 0 1; do
  ns "$m"
  "$PYV" "$REPO/scripts/acceptance_mtpgraphs.py" profile "$OUT/client-prof$m" --url $URL --seconds 3 >> "$OUT/window.log" 2>&1
  log "профиль g3 $m: код $?"
  mkdir -p "$OUT/profile-g3$m"; mv "$OUT/profile/"*.gz "$OUT/profile-g3$m/" 2>/dev/null
done

# 4) живая смесь (насыщенная) фазами g3 0/1
ns 0
lm "sat-g3" --clef-concurrency 6 --phase-file "$OUT/g3" --phase-values "0,1" --rounds 3 --phase-seconds 120
printf 0 > "$OUT/g3"
sleep 25; gpu_summary g3
grep -a "fa2_sm70 GC\]" "$OUT/server-g3.log" | tail -20 >> "$OUT/window.log"
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
