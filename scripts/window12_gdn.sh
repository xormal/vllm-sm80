#!/usr/bin/env bash
# [ОДИН ГРАФ 30.10] Окно 12 (окно обслуживания по согласию владельца): префилл GDN без обращений к хосту.
#  1) GPU0: scripts/gdn_devmeta_test.py -- метаданные чанков на устройстве против хостовых побитно, путь без
#     синхронизаций (sync_debug_mode=error), захват в CUDA-граф и повтор с другим разбиением, время CPU/GPU;
#  2) A/B на 127.0.0.1:$AB_PORT с профилем боевого: переключатель-файл $OUT/gdn ("0"/"1") сразу для
#     FA2SM80_GDN_DEVMETA и FA2SM80_GDN_ALL_VARLEN (один varlen-скан на слой в режиме 'all'); сверка Clef и отпечаток
#     LLM 0 -> 1 -> 0 (ожидается побитно);
#  3) профиль torch без стеков (C8 + записи Clef ~400 токенов) при gdn 0 и 1 -- разбор GPU/CPU шага в конфигурации боевого;
#  4) живая смесь (насыщенная очередь Clef + 8 чатов) и Пуассон: фазы gdn 0/1. Подрезка -- $TRIM (умолчание always).
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
NEW_BUILD="${NEW_BUILD:?каталог сборки gdn (FA2SM70_BUILD_DIR)}"
TRIM="${TRIM:-always}"
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
  log "старт A/B $AB (gdn -- переключатель-файл, подрезка $TRIM)"
  systemd-run --user --unit="$AB" --slice=sm80.slice --collect \
    -p Type=exec -p Restart=no -p MemoryHigh=20G -p MemoryMax=21G -p MemorySwapMax=0 -p CPUQuota=1600% \
    -p TasksMax=4096 -p OOMPolicy=kill -p OOMScoreAdjust=900 -p KillMode=control-group -p TimeoutStopSec=45 \
    -p WorkingDirectory="$REPO" -p StandardOutput=append:"$OUT/server-$1.log" -p StandardError=inherit \
    --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
    --setenv=FA2SM80_PREFILL_BUDGET_FILE="$OUT/budget" --setenv=FA2SM80_STEP_TIMING="$OUT/steps-$1.jsonl" \
    --setenv=FA2SM70_CLEF_HEAD_ASYNC=1 \
    --setenv=FA2SM80_GDN_DEVMETA_FILE="$OUT/gdn" --setenv=FA2SM80_GDN_ALL_VARLEN_FILE="$OUT/gdn" \
    --setenv=FA2SM70_MEM_TRIM_MODE="$TRIM" \
    --setenv=FA2SM80_MEMTRACE=20 --setenv=AB_PORT="$PORT" \
    --setenv=PROFILE_DIR="$OUT/profile" --setenv=PROFILE_STACK=false \
    "$REPO/scripts/serve_candidate_ab.sh"
  # первый старт после правки gdn_scan.cu -- пересборка расширения GDN
  if ! wait_health $URL 2400 "$AB"; then
    log "A/B $1 не поднялся"; tail -40 "$OUT/server-$1.log" >> "$OUT/window.log"; exit 1
  fi
  log "A/B $1 готов, прогрев 30 с"; sleep 30
}
gpu_summary() { grep -a "MEMTRACE\] .*GPU[01]:" "$OUT/server-$1.log" | tail -2 | sed 's/^.*MEMTRACE\] //' | tee -a "$OUT/window.log"; }
warm() {   # запросы со случайными токенами (переключатель gdn перечитывается раз в секунду)
  "$PYV" - "$URL" <<'PY'
import json, random, sys, urllib.request
for _ in range(3):
    ids = [random.randint(1000, 60000) for _ in range(900)]
    req = urllib.request.Request(sys.argv[1] + "/v1/completions", json.dumps({
        "model": "Qwen3.8-27B-ABLITERATED", "prompt": ids, "max_tokens": 1}).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
PY
}
gd() { printf "$1" > "$OUT/gdn"; sleep 2; warm; log "gdn: $1"; }
lm() {   # имя аргументы...
  "$PYV" "$REPO/scripts/acceptance_livemix.py" "$OUT/$1" --url $URL \
    --model Qwen3.8-27B-ABLITERATED --budget-file "$OUT/budget" "${@:2}" >> "$OUT/window.log" 2>&1
  log "живая смесь $1: код $?"
}
trap restore EXIT

printf 0 > "$OUT/gdn"; echo "2048:0" > "$OUT/budget"
log "окно 12: стоп $OBS и $LIVE"
systemctl --user stop "$OBS"
systemctl --user stop "$LIVE"
systemctl --user reset-failed "$LIVE" 2>/dev/null
for i in $(seq 1 60); do gpus_free && break; sleep 2; done
gpus_free || { log "GPU0-1 не освободились"; exit 1; }

# 1) стенд GDN на GPU0
systemd-run --user --scope --quiet --collect --slice=sm80.slice -p MemoryMax=12G -p MemorySwapMax=0 \
  bash -c 'source "$1/scripts/candidate_env.sh"; export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 FA2SM70_BUILD_DIR="$2"; exec "$3" "$1/scripts/gdn_devmeta_test.py" "$4"' \
  _ "$REPO" "$NEW_BUILD" "$PYV" "$OUT/gdn_test.json" > "$OUT/gdn_test.log" 2>&1
log "стенд GDN: код $?"
grep -E "^(побитно|без синхронизаций|граф|время)" "$OUT/gdn_test.log" | cut -c1-220 >> "$OUT/window.log"
for i in $(seq 1 30); do gpus_free && break; sleep 2; done

# 2) A/B: сверка gdn 0 -> 1 -> 0
start_ab gd
for m in 0 1 02; do
  gd "${m:0:1}"
  "$PYV" "$PS" clef "$PORT" "$OUT/clef-$m.json" > "$OUT/clef-$m.log" 2>&1
  log "Clef [gdn $m] против V100: $("$PYV" "$PS" clefsr "$REF_CLEF" "$OUT/clef-$m.json" 2>&1 | tail -1)"
  ref=""; [ "$m" != 0 ] && ref="$OUT/q-0.json"
  (cd "$BENCH" && URL=$URL OUT="$OUT/q-$m.json" REF="$ref" "$PYV" quality_ref.py) > "$OUT/q-$m.log" 2>&1
  log "отпечаток LLM [gdn $m]: $(grep -E '^L=' "$OUT/q-$m.log" | cut -c1-90 | tr '\n' ' ')"
done
log "Clef gdn 1 против 0: $("$PYV" "$PS" clefsr "$OUT/clef-0.json" "$OUT/clef-1.json" 2>&1 | tail -1)"
for m in 0 1; do
  gd "$m"
  "$PYV" "$PS" dlinnye "$PORT" "$OUT/dlinnye-$m.json" 34816 > "$OUT/dlinnye-$m.log" 2>&1
  log "dlinnye [gdn $m]: $(grep ИТОГ "$OUT/dlinnye-$m.log")"
done
grep -a "наш скан\|гибрид отказал\|откат" "$OUT/server-gd.log" | grep -a Worker_TP0 | head -5 >> "$OUT/window.log"

# 3) профиль (без стеков, конфигурация боевого): C8 декод + записи Clef ~400 токенов каждые 0,5 с; gdn 0 и 1
for m in 0 1; do
  gd "$m"
  "$PYV" "$REPO/scripts/acceptance_mtpgraphs.py" profile "$OUT/client-prof$m" --url $URL --seconds 3 >> "$OUT/window.log" 2>&1
  log "профиль gdn $m: код $?"
  mkdir -p "$OUT/profile-gdn$m"; mv "$OUT/profile/"*.gz "$OUT/profile-gdn$m/" 2>/dev/null
  ls -la "$OUT/profile-gdn$m" | tail -n +2 | tee -a "$OUT/window.log"
done

# 4) живая смесь: фазы gdn 0/1
gd 0
lm "sat-gdn" --clef-concurrency 6 --phase-file "$OUT/gdn" --phase-values "0,1" --rounds 2 --phase-seconds 120
gd 0
lm "poisson-gdn" --phase-file "$OUT/gdn" --phase-values "0,1" --rounds 2 --phase-seconds 150
printf 0 > "$OUT/gdn"
sleep 25; gpu_summary gd
grep -acE "CUDA out of memory|EngineDeadError|Traceback" "$OUT"/server-*.log | sed 's/^/ошибок: /' | tee -a "$OUT/window.log"
