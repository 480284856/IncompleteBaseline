cd /Users/jay/Desktop/RL/project/


python -m IncompleteBaseline.dev.main \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/normal/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 250 \
    --max-episode-steps 250 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 42
python -m IncompleteBaseline.dev.main \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/normal/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 250 \
    --max-episode-steps 250 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 43
python -m IncompleteBaseline.dev.main \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/normal/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 250 \
    --max-episode-steps 250 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 44