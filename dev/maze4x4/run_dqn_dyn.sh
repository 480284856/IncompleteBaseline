cd /Users/jay/Desktop/RL/project/

python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/dynamic_epsilon/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --epsilon-strategy "StepDecay" \
    --epsilon-decay 10240 \
    --seed 42
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/dynamic_epsilon/v2" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --epsilon-strategy "StepDecay" \
    --epsilon-decay 10240 \
    --seed 43
python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze4x4/dynamic_epsilon/v3" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --epsilon-strategy "StepDecay" \
    --epsilon-decay 10240 \
    --seed 44