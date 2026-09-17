git checkout 83d462b2692b2aa35f99addc7e5e633d996f64e6

python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/random_baseline/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 42
python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/random_baseline/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 43
python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze5x5/random_baseline/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 1228800 \
    --learning-starts 819200 \
    --seed 44