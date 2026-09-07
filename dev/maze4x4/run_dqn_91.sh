cd /Users/jay/Desktop/RL/project/

# --max-episode-steps 1600: 100 times of maze size 
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/fixed_epsilon/heavy_random_sampling/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --max-episode-steps 1600 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --seed 42

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/fixed_epsilon/heavy_random_sampling/v2" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --max-episode-steps 1600 \
    --seed 43

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/fixed_epsilon/heavy_random_sampling/v3" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --max-episode-steps 1600 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --epsilon-strategy "91Epsilon" \
    --seed 44