python -m IncompleteBaseline.dev.main_actor \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 42
python -m IncompleteBaseline.dev.main_actor \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 43
python -m IncompleteBaseline.dev.main_actor \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --max-episode-steps 2500 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 44