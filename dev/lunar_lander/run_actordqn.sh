cd /Users/jay/Desktop/RL/project/

python -m IncompleteBaseline.dev.lunar_lander.main_actor \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/lunarlander/v1" \
    --replay-capacity 50000 \
    --max-episode-steps-eval 1000 --max-episode-steps 1000 \
    --total-time-steps 409600 \
    --learning-starts 51200 \
    --hidden-sizes 128 64 \
    --seed 42

python -m IncompleteBaseline.dev.lunar_lander.main_actor \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/lunarlander/v2" \
    --replay-capacity 50000 \
    --max-episode-steps-eval 1000 --max-episode-steps 1000 \
    --total-time-steps 409600 \
    --learning-starts 51200 \
    --hidden-sizes 128 64 \
    --seed 43

python -m IncompleteBaseline.dev.lunar_lander.main_actor \
    --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/lunarlander/v3" \
    --replay-capacity 50000 \
    --max-episode-steps-eval 1000 --max-episode-steps 1000 \
    --total-time-steps 409600 \
    --learning-starts 51200 \
    --hidden-sizes 128 64 \
    --seed 44