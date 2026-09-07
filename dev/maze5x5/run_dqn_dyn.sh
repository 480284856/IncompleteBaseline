# python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/dynamic_epsilon/large/v1" \
#     --width 5 \
#     --height 5 \
#     --max-episode-steps-eval 25 \
#     --total-time-steps 819200 \
#     --learning-starts 409600 \
#     --epsilon-strategy "StepDecay" \
#     --epsilon-decay 20480 \
#     --seed 42
# python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/dynamic_epsilon/large/v2" \
#     --width 5 \
#     --height 5 \
#     --max-episode-steps-eval 25 \
#     --total-time-steps 819200 \
#     --learning-starts 409600 \
#     --epsilon-strategy "StepDecay" \
#     --epsilon-decay 20480 \
#     --seed 43
# python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/dynamic_epsilon/large/v3" \
#     --width 5 \
#     --height 5 \
#     --max-episode-steps-eval 25 \
#     --total-time-steps 819200 \
#     --learning-starts 409600 \
#     --epsilon-strategy "StepDecay" \
#     --epsilon-decay 20480 \
#     --seed 44

python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/dynamic_epsilon/large/more_exp/v1" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --epsilon-strategy "StepDecay" \
    --epsilon-decay 204800 \
    --seed 42
python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/dynamic_epsilon/large/more_exp/v2" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --epsilon-strategy "StepDecay" \
    --epsilon-decay 204800 \
    --seed 43
python main.py --tensorboard-log-dir "../logs/qdn/maze5x5/dynamic_epsilon/large/more_exp/v3" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 819200 \
    --learning-starts 409600 \
    --epsilon-strategy "StepDecay" \
    --epsilon-decay 204800 \
    --seed 44