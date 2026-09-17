python -m IncompleteBaseline.dev.main \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/normal/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --epsilon-strategy "91Epsilon" \
    --seed 42
python -m IncompleteBaseline.dev.main \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/normal/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --epsilon-strategy "91Epsilon" \
    --seed 43
python -m IncompleteBaseline.dev.main \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/normal/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --epsilon-strategy "91Epsilon" \
    --seed 44