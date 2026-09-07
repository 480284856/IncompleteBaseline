# python main_actor.py --tensorboard-log-dir "../logs/actorqdn/maze5x5/large/v1" \
#     --width 5 \
#     --height 5 \
#     --max-episode-steps-eval 25 \
#     --total-time-steps 819200 \
#     --learning-starts 409600 \
#     --seed 42
python main_actor.py --tensorboard-log-dir "../logs/actorqdn/maze5x5/large/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --seed 43
python main_actor.py --tensorboard-log-dir "../logs/actorqdn/maze5x5/large/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --seed 44