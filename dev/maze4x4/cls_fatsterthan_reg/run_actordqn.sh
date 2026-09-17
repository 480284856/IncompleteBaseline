#!/bin/bash
set -e

cd /Users/jay/Desktop/RL/project/

MODEL_SAVE_DIR="/Users/jay/Desktop/RL/project/IncompleteBaseline/dev/maze4x4/cls_fatsterthan_reg/models"
LOG_DIR="IncompleteBaseline/dev/logs/actorqdn/maze4x4/cls_fatsterthan_reg/"

python -m IncompleteBaseline.dev.main_actor \
    --model-save-dir "$MODEL_SAVE_DIR/" \
    --tensorboard-log-dir "$LOG_DIR/" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --max-episode-steps 1600 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --seed 42