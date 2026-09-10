#!/bin/bash
set -e

cd "$(dirname "$0")"

MODEL_DIR="models"
OUTPUT_DIR="probe_results"

python train_probe.py \
    --dqn-target "$MODEL_DIR/q_target_network_dqn.pt" \
    --actordqn-target "$MODEL_DIR/q_target_network_actordqn.pt" \
    --output-dir "$OUTPUT_DIR" \
    --num-transitions 102400 \
    --validation-fraction 0.2 \
    --steps 102400 \
    --eval-every 10000 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --max-episode-steps 1600 \
    --seed 42
