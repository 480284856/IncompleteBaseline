python main.py --tensorboard-log-dir "./logs/qdn/maze4x4/fixed_epsilon/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --epsilon-strategy "91Epsilon" \
    --seed 42
python main.py --tensorboard-log-dir "./logs/qdn/maze4x4/fixed_epsilon/v2" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --epsilon-strategy "91Epsilon" \
    --seed 43
python main.py --tensorboard-log-dir "./logs/qdn/maze4x4/fixed_epsilon/v3" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --epsilon-strategy "91Epsilon" \
    --seed 44