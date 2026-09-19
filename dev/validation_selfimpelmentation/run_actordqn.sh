cd /Users/jay/Desktop/RL/project/

# python -m IncompleteBaseline.dev.debug.main_actor \
#     --tensorboard-log-dir "IncompleteBaseline/dev/debug/logs/algorithm/cartpole/v3" \
#     --replay-capacity 204800 \
#     --max-episode-steps-eval 500 \
#     --max-episode-steps 500 \
#     --total-time-steps 409600 \
#     --learning-starts 102400 \
#     --seed 42

python -m IncompleteBaseline.dev.debug.main_actor \
    --tensorboard-log-dir "IncompleteBaseline/dev/debug/logs/algorithm/cartpole/v1_small_lr" \
    --replay-capacity 50000 \
    --max-episode-steps-eval 500 --max-episode-steps 500 \
    --total-time-steps 409600 \
    --learning-starts 51200 \
    --seed 42