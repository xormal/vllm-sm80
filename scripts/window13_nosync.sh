#!/usr/bin/env bash
# [БЕЗ ХОСТА 30.10] Окно 13 (окно обслуживания; владелец: «делай по порядку, лучшего кандидата в бой и следи за ним»).
# Профиль окна 12 (конфигурация боевого, без стеков): шаг с записью Clef ~400 токенов -- ~300 мс, из них прямой GEMM
# 112 мс и 48-107 мс простоя GPU, пока главный поток ждёт хост: .cpu() приёмки MTP (21-90 мс на каждом смешанном шаге),
# cudaStreamSynchronize в req_begin Clef, .item() проверки срыва обмена (раз в 256 обменов, 18-81 мс).
#  0) правки из рабочего дерева $WT -> основное дерево ТОЛЬКО при остановленном боевом (сверка с базой $BASE_TAR);
#  1) GPU0: scripts/w8_midm_bench.py -- v2 (w8a16_sm80_gemm2) против прямого ядра на средних M, побитно;
#  2) A/B на 127.0.0.1:$AB_PORT с профилем боевого; уровень в $OUT/ns (FA2SM70_NOSYNC_FILE):
#     0 -- как в бою; 1 -- приёмка на устройстве + req_begin без синхронизации + асинхронная проверка срыва;
#     2 -- 1 + без подрезки; 3 -- 2 + v2 на средних M. Сверка Clef / отпечаток LLM / dlinnye: 3 против 0;
#  3) профиль без стеков при 0 и 3; живая смесь (насыщенная и Пуассон) фазами 0/1/2/3.
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
FILES="vllm/v1/worker/gpu_model_runner.py sm70/integrations/vllm_sm70/fa2sm70_exch_i12.py sm70/integrations/vllm_sm70/fa2sm70_clef.py sm70/fa2_sm70/csrc/clef.cu sm70/integrations/vllm_sm70/fa2sm80_w8.py"
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
    # /health мало (урок 30.10: падение на первом запросе с M 65..512) -- 5 минут живой нагрузки без смертей движка
    sleep 300
    log "боевой через 5 мин: падений $(tail -n +$S0 "$LL" | grep -ac 'EngineCore encountered a fatal error'), 500: $(tail -n +$S0 "$LL" | grep -ac '500 Internal Server Error'), 200: $(tail -n +$S0 "$LL" | grep -ac '" 200 OK'), NRestarts $(systemctl --user show $LIVE -p NRestarts --value)"
  else
    log "ВНИМАНИЕ: боевой не ответил за 1500 с: $(systemctl --user show $LIVE -p ActiveState -p SubState | tr '\n' ' ')"
  fi
}
start_ab() {   # имя-захода
  AB=kandidat-ab-$1-$(date +%H%M).service
  log "старт A/B $AB (уровень без-хоста -- переключатель-файл $OUT/ns)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM70_CLEF_HEAD_ASYNC=1 \
    --setenv=FA2SM70_NOSYNC_FILE="$OUT/ns" \
    --setenv=FA2SM80_MEMTRACE=20 --setenv=AB_PORT="$PORT" \
    --setenv=PROFILE_DIR="$OUT/profile" --setenv=PROFILE_STACK=false \
    "$REPO/scripts/serve_candidate_ab.sh"
  # первый старт после правки clef.cu -- пересборка расширения Clef; подбор v2 средних M при загрузке
  if ! wait_health $URL 2400 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
  grep -a "v2 средних M" "$OUT/server-$1.log" | grep -a Worker_TP0 | sed 's/^.*W8\] //' | head -20 >> "$OUT/window.log"
}
gpu_summary() { grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
warm() {   # запросы со случайными токенами (переключатель перечитывается раз в секунду / по счётчику)
  "$PYV" - "$URL" <<'PY'
import json, random, sys, urllib.request
for _ in range(3):
    ids = [random.randint(1000, 60000) for _ in range(900)]
    req = urllib.request.Request(sys.argv[1] + "/v1/completions", json.dumps({
        "model": "Qwen3.8-27B-ABLITERATED", "prompt": ids, "max_tokens": 1}).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
PY
}
ns() { printf "$1" > "$OUT/ns"; sleep 2; warm; log "уровень: $1"; }
lm() {   # имя аргументы...
  "$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/$1" --url $URL \
    --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" "${@:2}" >> "$OUT/window.log" 2>&1
  log "живая смесь $1: код $?"
}
trap restore EXIT

printf 0 > "$OUT/ns"; echo "2048:0" > "$OUT/budget"
log "окно 13: стоп $OBS и $LIVE"
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
for f in $FILES; do cp -a "$WT/$f" "$REPO/$f"; done
log "правки применены: $FILES"

# 1) GEMM средних M на GPU0
systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
  bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0; exec "$2" "$1/scripts/w8_midm_bench.py" --json "$3"' \
  _ "$REPO" "$PYV" "$OUT/midm.json" > "$OUT/midm.log" 2>&1
log "GEMM средних M: код $?"
"$PYV" - "$OUT/midm.json" <<'PY' | tee -a "$OUT/window.log"
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("midm.json не читается:", e); sys.exit()
for x in r:
    b = x.get("v2_лучший") or {}
    eq = [v for v in x["v2"] if v.get("побитно_как_прямой_sp1") is not None]
    print(f"{x['форма']:8s} M={x['M']:5d} прямой{tuple(x['текущий'])} {x['TF_текущий']:6.1f} TF | деквант {x['TF_деквант']:6.1f} | "
          f"v2 лучший {b.get('TF', 0):6.1f} TF (nj={b.get('nj')} st={b.get('st')} pin={b.get('pin')} sp={b.get('sp')}) | "
          f"побитно sp1: {sum(v['побитно_как_прямой_sp1'] for v in eq)}/{len(eq)}")
PY
for i in $(seq 1 30); do gpus_free && break; sleep 2; done

# 2) A/B: сверка 3 против 0
start_ab ns
for m in 0 3; do
  ns "$m"
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$m.json" > "$OUT/clef-$m.log" 2>&1
  log "Clef [уровень $m] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$m.json" 2>&1 | tail -1)"
  ref=""; [ "$m" != 0 ] && ref="$OUT/q-0.json"
  (cd "$BENCH" && URL=$URL OUT="$OUT/q-$m.json" REF="$ref" "$PYV" quality_ref.py) > "$OUT/q-$m.log" 2>&1
  log "отпечаток LLM [уровень $m]: $(grep -E '^L=' "$OUT/q-$m.log" | cut -c1-90 | tr '\n' ' ')"
done
log "Clef 3 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-3.json" 2>&1 | tail -1)"
ns 3
"$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye-3.json" 34816 > "$OUT/dlinnye-3.log" 2>&1
log "dlinnye [уровень 3]: $(grep ИТОГ "$OUT/dlinnye-3.log")"
grep -aE "Traceback|Error" "$OUT/server-ns.log" | grep -av "INFO" | head -5 >> "$OUT/window.log"

# 3) профиль без стеков при 0 и 3
for m in 0 3; do
  ns "$m"
  "$PYV" "$REPO/scripts/acceptance_mtpgraphs.py" profile "$OUT/client-prof$m" --url $URL --seconds 3 >> "$OUT/window.log" 2>&1
  log "профиль уровень $m: код $?"
  mkdir -p "$OUT/profile-ns$m"; mv "$OUT/profile/"*.gz "$OUT/profile-ns$m/" 2>/dev/null
done

# 4) живая смесь фазами уровней
ns 0
lm "sat-ns" --clef-concurrency 6 --phase-file "$OUT/ns" --phase-values "0,1,2,3" --rounds 2 --phase-seconds 100
gpu_summary ns
ns 0
lm "poisson-ns" --phase-file "$OUT/ns" --phase-values "0,1,2,3" --rounds 2 --phase-seconds 120
printf 0 > "$OUT/ns"
sleep 25; gpu_summary ns
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
