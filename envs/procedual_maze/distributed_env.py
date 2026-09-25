from functools import partial
from env import MazeV2, generate_maze
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(width, height, step_limitation, dataset):
    return MazeV2(width=width, height=height, step_limitation=step_limitation, dataset=dataset)

def create_dist_environments(
    *,
    n_env: int,
    seed: int,
    width: int,
    height: int,
    step_limitation: int,
):
    training_set, eval_set = generate_maze(seed=seed, split_ratio=0.2, width=width, height=height)
    train_env_fns = [
        partial(
            make_env,
            width=width,
            height=height,
            step_limitation=step_limitation,
            dataset=training_set,
        )
        for _ in range(n_env)
    ]
    training_env = SubprocVecEnv(
        train_env_fns
    )
    eval_env = MazeV2(width=width, height=height, step_limitation=step_limitation, dataset=eval_set)
    return training_env, eval_env

if __name__ == "__main__":
    create_dist_environments(
        n_env=2,
        width=5,
        height=5,
        step_limitation=1000,
        seed=42,
    )