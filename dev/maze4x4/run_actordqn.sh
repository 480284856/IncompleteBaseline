python main_actor.py --tensorboard-log-dir "../logs/actorqdn/maze4x4/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --seed 42
python main_actor.py --tensorboard-log-dir "../logs/actorqdn/maze4x4/v2" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --seed 43
python main_actor.py --tensorboard-log-dir "../logs/actorqdn/maze4x4/v3" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 409600 \
    --learning-starts 204800 \
    --seed 44