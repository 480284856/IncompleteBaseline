from env import Maze
import gymnasium as gym
from functools import partial

def make_maze_env(
    width,
    height,
    training_mazes,
    evaluation_mazes,
):
    env = Maze(width=width, height=height)
    env.training_mazes = training_mazes
    env.evaluation_mazes = evaluation_mazes
    return env


def create_vector_environment(width: int, height: int, seed: int, logger, num_train_envs,num_eval_envs) -> tuple[Maze, Maze]:
    """Create independent environments backed by one generated maze dataset."""
    source_environment = Maze(width=width, height=height)

    # Generation is deterministic for a seed and can be expensive. Both
    # environments may share these arrays because reset() copies a sampled maze
    # before changing it.
    maze_generator_code = source_environment.generate_maze.__func__.__code__
    logger.warning(
        "Maze generation is starting and may take a while (%s:%d).",
        maze_generator_code.co_filename,
        maze_generator_code.co_firstlineno,
    )
    source_environment.generate_maze(seed=seed)


    training_environment = gym.vector.SyncVectorEnv(
        [
            partial(
                make_maze_env,
                width,
                height,
                source_environment.training_mazes,
                source_environment.evaluation_mazes,
            )
            for _ in range(num_train_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    eval_environment = gym.vector.SyncVectorEnv(
            [
                partial(
                    make_maze_env,
                    width,
                    height,
                    source_environment.training_mazes,
                    source_environment.evaluation_mazes,
                )
                for _ in range(num_eval_envs)
            ],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, # call reset() for each env at the same time
        )

    actor_eval_environment = gym.vector.SyncVectorEnv(
                [
                    partial(
                        make_maze_env,
                        width,
                        height,
                        source_environment.training_mazes,
                        source_environment.evaluation_mazes,
                    )
                    for _ in range(num_eval_envs)
                ],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, # call reset() for each env at the same time
            )

    return training_environment, eval_environment, actor_eval_environment