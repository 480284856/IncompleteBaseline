cd /Users/jay/Desktop/RL/project/

# --max-episode-steps 2500: 100 times of maze size
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/fixed_epsilon/mabrollout/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --max-episode-steps-warmup 5000 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --epsilon-strategy "91Epsilon" \
    --seed 42