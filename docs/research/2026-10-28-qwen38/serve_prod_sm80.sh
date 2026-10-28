#!/usr/bin/env bash
# Боевой сервер sm80 (A100, GPU0-1): тот же набор, что у боевого V100 -- 0.0.0.0:8085, Clef, имена моделей.
# Конфиг рычагов -- vllm-sm70/configs/qwen38-27b-int8-sm80.env (+ clef.env). Запускается юнитом
# ~/.config/systemd/user/sm80-vllm.service (автозапуск при загрузке через linger); вручную -- так же.
set -euo pipefail
R=/home/alex/devpack_sm70_20261022/vllm-sm70
W=/home/alex/devpack_sm70_20261022/sm80_work
cd "$R"
export PY=.venv/bin/python
export MODEL_DIR=/home/alex/devpack_sm70_20261022/model/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef
export CARDS=0,1 HOST=0.0.0.0 PORT=8085
export FA2SM70_CLEF=1
export SERVED_NAMES="huihui-qwen3.5-35b-a3b-abliterated-q8_0 gpt-4-0125-preview Qwen3.6-27B-AWQ Qwen3.8-27B-ABLITERATED"
export CUDA_HOME=/usr/local/cuda-12.9
export FA2SM70_BUILD_DIR=$W/build_sm80
export FA2SM80_SEL_DENSE_NB=1024
# [26.10] прибор памяти хоста: только классы отображений раз в 15 мин (без tracemalloc/gc -- не влияет на скорость)
export FA2SM80_MEMTRACE=900
exec scripts/serve_qwen38_sm80.sh
