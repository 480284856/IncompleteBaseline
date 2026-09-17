cd /Users/jay/Desktop/RL/project/

python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze4x4/normal/v1" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 128 \
    --max-episode-steps 512 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --seed 42 

python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze4x4/normal/v2" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 128 \
    --max-episode-steps 512 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --seed 43

python -m IncompleteBaseline.dev.main_actor --tensorboard-log-dir "IncompleteBaseline/dev/logs/actorqdn/maze4x4/normal/v3" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 128 \
    --max-episode-steps 512 \
    --total-time-steps 204800 \
    --learning-starts 102400 \
    --seed 44