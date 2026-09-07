# python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/fixed_epsilon/large/v1" \
#     --width 5 \
#     --height 5 \
#     --max-episode-steps-eval 25 \
#     --total-time-steps 819200 \
#     --learning-starts 409600 \
#     --epsilon-strategy "91Epsilon" \
#     --seed 42
python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/fixed_epsilon/large/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --epsilon-strategy "91Epsilon" \
    --seed 43
python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/fixed_epsilon/large/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --epsilon-strategy "91Epsilon" \
    --seed 44