cd /Users/jay/Desktop/RL/project/

# --max-episode-steps 2500: 100 times of maze size
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/fixed_epsilon/step-25/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --max-episode-steps-warmup 25 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --seed 42
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/fixed_epsilon/step-25/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --max-episode-steps-warmup 25 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --seed 43
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/fixed_epsilon/step-25/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --max-episode-steps-warmup 25 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --seed 44 