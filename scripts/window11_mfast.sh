#!/usr/bin/env bash
# [ЗАГРУЗКА ВЕСОВ 30.10] Окно 11 (окно обслуживания по согласию владельца): фризы 100-200 мс на шагах с записью Clef.
#  1) ядро на GPU0: прежняя боевая .so против новой -- прямой GEMM mfast=0/1 побитно, out=/x срезом, время
#     прежнего порядка блоков против M-первыми и против декванта + cuBLAS;
#  2) A/B на 127.0.0.1:$AB_PORT с профилем боевого (бюджет 2048:0, голова асинхронно): сверка Clef (35 решений против
#     V100) и отпечаток LLM при mfast 0 -> 1 -> 0;
#  3) живая смесь (насыщенная очередь Clef + 8 чатов): подрезка always/off/nodecode при mfast 0, затем при mfast 1;
#     поток Пуассона: подрезка always/off при mfast 1. Время каждого шага -- steps-*.jsonl, повторы выделения -- MEMTRACE.
# Боевой ВСЕГДА поднимается (trap), если RESTORE=1. Только GPU0-1.
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
SO_OLD="$(dirname "$REPO")/sm80_work/build_sm80/fa2sm70_w8a16_sm80/fa2sm70_w8a16_sm80.so"
NEW_BUILD="${NEW_BUILD:?каталог сборки новой .so (FA2SM70_BUILD_DIR)}"
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
  if wait_health http://127.0.0.1:8085 1500; then
    log "боевой $LIVE снова отвечает на 8085: $(systemctl --user show $LIVE -p MainPID -p NRestarts | tr '\n' ' ')"
  else
    log "ВНИМАНИЕ: боевой не ответил за 1500 с: $(systemctl --user show $LIVE -p ActiveState -p SubState | tr '\n' ' ')"
  fi
}
start_ab() {   # имя-захода
  AB=kandidat-ab-$1-$(date +%H%M).service
  log "старт A/B $AB (mfast и подрезка -- переключатели-файлы)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM70_CLEF_HEAD_ASYNC=1 \
    --setenv=FA2SM80_W8_MFAST_FILE="$OUT/mfast" --setenv=FA2SM70_MEM_TRIM_FILE="$OUT/trim" \
    --setenv=FA2SM80_MEMTRACE=20 --setenv=AB_PORT="$PORT" \
    "$REPO/scripts/serve_candidate_ab.sh"
  # первый старт после правки w8a16_sm80.cu -- пересборка расширения и замер маршрута больших M (корзины 1024/2048)
  if ! wait_health $URL 2400 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
  grep -a "маршрут больших M" "$OUT/server-$1.log" | grep -a Worker_TP0 | sed 's/^.*W8\] //' | head -14 >> "$OUT/window.log"
}
gpu_summary() { grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
warm() {   # запросы со случайными токенами: переключатель mfast читается по счётчику вызовов больших M
  "$PYV" - "$URL" <<'PY'
import json, random, sys, urllib.request
for _ in range(3):
    ids = [random.randint(1000, 60000) for _ in range(900)]
    req = urllib.request.Request(sys.argv[1] + "/v1/completions", json.dumps({
        "model": "Qwen3.8-27B-ABLITERATED", "prompt": ids, "max_tokens": 1}).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
PY
}
mf() { printf "$1" > "$OUT/mfast"; sleep 1; warm; log "mfast: $1"; }
lm() {   # имя аргументы...
  "$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/$1" --url $URL \
    --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" "${@:2}" >> "$OUT/window.log" 2>&1
  log "живая смесь $1: код $?"
}
trap restore EXIT

echo "2048:0" > "$OUT/budget"; printf 0 > "$OUT/mfast"; echo always > "$OUT/trim"
cp -a "$SO_OLD" "$OUT/so_old.so"
log "окно 11: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 1) ядро: прежняя .so сохраняет выходы, новая сверяет и меряет
kt() {   # аргументы теста; окружение -- в bash -c
  systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
    bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 FA2SM70_BUILD_DIR="$2"; shift 2; exec "$@"' \
    _ "$REPO" "$NEW_BUILD" "$PYV" "$REPO/scripts/w8_mfast_test.py" "$@"
}
log "ядро: прежняя .so -> выходы"
kt --so "$OUT/so_old.so" --save "$OUT/old_out.pt" > "$OUT/kernel_old.log" 2>&1
log "ядро (прежняя): код $?"
kt --check "$OUT/old_out.pt" --json "$OUT/kernel.json" > "$OUT/kernel.log" 2>&1
log "ядро (новая): код $?"
"$PYV" - "$OUT/kernel.json" <<'PY' | tee -a "$OUT/window.log"
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("kernel.json не читается:", e); sys.exit()
ключи = ("mfast0_как_старая_so", "mfast1_как_старая_so", "mfast1_как_mfast0", "out_срез_побитно", "out_соседи_целы")
for k in ключи:
    плохо = [f"{x['форма']}/{x['M']}" for x in r if not x.get(k, False)]
    print(f"{k}: {len(r) - len(плохо)}/{len(r)}" + (f" ПЛОХО {плохо[:8]}" if плохо else ""))
for x in r:
    print(f"{x['форма']:8s} M={x['M']:5d} mi={x['mi']} sp={x['splits']}: прежний {x['TF_прежний']:6.1f} TF, "
          f"M-первыми {x['TF_mfast']:6.1f} TF, деквант+cuBLAS {x['TF_деквант']:6.1f} TF, relL2 {x['relL2_к_fp32']:.1e}")
PY
for i in $(seq 1 30); do gpus_free && break; sleep 2; done

# 2) A/B: сверка mfast 0 -> 1 -> 0
start_ab mf
for m in 0 1 02; do
  mf "${m:0:1}"
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$m.json" > "$OUT/clef-$m.log" 2>&1
  log "Clef [mfast $m] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$m.json" 2>&1 | tail -1)"
  ref=""; [ "$m" != 0 ] && ref="$OUT/q-0.json"
  (cd "$BENCH" && URL=$URL OUT="$OUT/q-$m.json" REF="$ref" "$PYV" quality_ref.py) > "$OUT/q-$m.log" 2>&1
  log "отпечаток LLM [mfast $m]: $(grep -E '^L=' "$OUT/q-$m.log" | tr '\n' ' ')"
done
log "Clef mfast 1 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-1.json" 2>&1 | tail -1)"
log "Clef mfast 0 повтор против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-02.json" 2>&1 | tail -1)"

# 3) живая смесь: подрезка по фазам при mfast 0, затем 1; Пуассон при mfast 1
for m in 0 1; do
  mf "$m"
  echo always > "$OUT/trim"
  lm "sat-mf$m" --clef-concurrency 6 --phase-file "$OUT/trim" --phase-values "always,off,nodecode" --rounds 2 --phase-seconds 120
  gpu_summary mf
done
echo always > "$OUT/trim"
lm "poisson-mf1" --phase-file "$OUT/trim" --phase-values "always,off" --rounds 2 --phase-seconds 150
echo always > "$OUT/trim"; printf 0 > "$OUT/mfast"
sleep 25; gpu_summary mf
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
