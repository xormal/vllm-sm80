#!/usr/bin/env bash
# Authorized live-load trial. The original production unit remains unchanged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/candidate_env.sh"
export CANDIDATE_VARIANT=candidate
export PY="$CANDIDATE_REPO/scripts/acceptance_python.sh"
export CARDS=0,1 HOST=0.0.0.0 PORT=8085
export SERVED_NAMES="huihui-qwen3.5-35b-a3b-abliterated-q8_0 gpt-4-0125-preview Qwen3.6-27B-AWQ Qwen3.8-27B-ABLITERATED"
export FA2SM70_CLEF=1 FA2SM80_SEL_DENSE_NB=1024 FA2SM70_EXCH12=1
export FA2SM80_RESUME_ADMISSION=1 FA2SM70_GRAN_CONV_ALL=1
export FA2SM80_BLOCKPIPE_FALLBACK_ROWS=0 FA2SM80_CONV_META_FUSED=0
export FA2SM80_MEMTRACE=900
# [ПЛАВНОСТЬ 30.10] по решению владельца после окон 9-10b: бюджет prefill 2048 токенов на шаг, пока идёт декод
# (без порога очереди), и голова Clef вне критического пути (сборщик с замком захвата графов)
export FA2SM80_PREFILL_BUDGET_DECODING=1024 FA2SM80_PREFILL_BUDGET_BACKLOG=0
export FA2SM70_CLEF_HEAD_ASYNC=1
# [БЕЗ ХОСТА 30.10] после окна 13 (владелец: «лучшего кандидата в бой и следи за ним»): главный поток больше не ждёт хост
# на каждом смешанном шаге -- приёмка MTP переносится на устройстве, req_begin Clef без синхронизации потока, проверка
# срыва обмена асинхронно; подрезка выключена (иначе empty_cache становится следующей точкой ожидания карты).
# Насыщенная очередь Clef + 8 чатов: чанков чатов/с 14.67 -> 16.93, p99 0.723 -> 0.665, Clef 2886 -> 3040 ток/с;
# сверка побитная (Clef 35/35 Δp 0, LLM 160/160, dlinnye 50/50). Откат: убрать две строки ниже и перезапустить.
export FA2SM70_NOSYNC_ACC2=1
export FA2SM70_MEM_TRIM_MODE=off
# [ПЛАВНОСТЬ 30.10] Чередование декода: после шага с куском prefill при декоде -- 4 шага чистого декода (scheduler.py
# FA2SM80_DECODE_INTERLEAVE). Окно 23 (смесь LLM+Clef): чаты 20.0 против 5.0 ток/с на поток при c=4, 17.4 против 4.5
# при c=8, пауза p50 0.04-0.05 с против 0.6 с; Clef -16..-22 %. Решение владельца 30.10 («включай K=4 на боевом»).
# Откат -- убрать строку и перезапустить.
export FA2SM80_DECODE_INTERLEAVE=4
# [ФРИЗЫ 30.10] Решение владельца после окон 26-28 («меряй деквант, если он лучше, то его, иначе E-1024 ставь»):
# GEMM весов -- Marlin8 (M <= 1024; больше -- деквант из раскладки Marlin + cuBLAS), конвейер блока с постоянными
# буферами и на средних M, без синхронизаций построителя GDN и внимания; бюджет prefill при декоде -- 1024 (выше).
# Окна 26-27: чаты под Clef 33 / 25-27 ток/с на поток (было 20 / 17), пауза p99 0.35-0.45 с (было 0.58-0.60), Clef в
# смеси -12..-23 %; точность: Clef 34/35 против V100, LLM 160/160, длинные 50/50. Откат: бюджет 2048, убрать блок,
# перезапустить (прежний скрипт -- state/acceptance/window28-20261030/backup).
export FA2SM80_W8_MARLIN=1
export FA2SM80_BLOCKPIPE_BUFS=1 FA2SM80_BLOCKPIPE_M8_MIN=256
export FA2SM70_GDN_IDX_VIEW=1 FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_CONV_META_PINNED=1
export FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1
# деквант Marlin -- вариант 42 (окно 28: стенд marlin8_dequant_bench.py, побитно как наш)
export FA2SM80_M8DQ=42
# [ГРАФЫ 31.10] Окно 29: полные графы декода и для 9 запросов (пул GDN пускает 9; размеры при k=4 -- кратные 5, было
# до 40 токенов = 8 запросов, шаг с 9 запросами шёл eager: 81.6 против 46 мс), граф цикла черновика MTP -- до 9.
# Откат: убрать две строки и перезапустить (размеры захвата -- в хэше компиляции: прежний кэш остаётся).
export FA2SM80_CG_SIZES=1,2,4,8,12,16,20,24,28,32,36,40,45
export FA2SM80_MTP_LOOP_GRAPH_MAXB=9
export VLLM_CACHE_ROOT="$STATE_DIR/acceptance/vllm-candidate-fallback-0-conv-all-1"
export FA2SM70_CLEF_OUT="$STATE_DIR/acceptance/clef-candidate"
cd "$CANDIDATE_REPO"
exec scripts/serve_qwen38_sm80.sh
