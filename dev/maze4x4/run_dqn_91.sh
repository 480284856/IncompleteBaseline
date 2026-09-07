cd /Users/jay/Desktop/RL/project/

# --max-episode-steps 1600: 100 times of maze size 
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/fixed_epsilon/step-1600/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --max-episode-steps 1600 \
    --max-episode-steps-warmup 1600 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --seed 42

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/fixed_epsilon/step-16/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --max-episode-steps 1600 \
    --max-episode-steps-warmup 16 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --max-episode-steps 1600 \
    --seed 43

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/fixed_epsilon/step-1/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --max-episode-steps 1600 \
    --max-episode-steps-warmup 1 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --seed 44