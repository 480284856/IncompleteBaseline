cd /Users/jay/Desktop/RL/project/

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 160 \
    --max-episode-steps 160 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --seed 42

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/v2" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 160 \
    --max-episode-steps 160 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --seed 43

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/v3" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 160 \
    --max-episode-steps 160 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --seed 44