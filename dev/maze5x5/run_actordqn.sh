cd /Users/jay/Desktop/RL/project/

python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/normal/step-250/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 250 \
    --max-episode-steps 250 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 42
python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/normal/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 128 \
    --max-episode-steps 2048 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 43
python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/normal/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 128 \
    --max-episode-steps 2048 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 44