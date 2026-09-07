"""Train and evaluate the DQN agent on the maze environment."""

from __future__ import annotations

import logging
import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch

# Allow direct execution from ActorDQN while importing its sibling package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from IncompleteBaseline.algorithms.common.exploration_rate_calculation import StepDecay
from IncompleteBaseline.algorithms.actordqn.actordqn_agent import ActorDQNAgent
from IncompleteBaseline.envs.procedual_maze.env import Maze

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a training run."""
    parser = argparse.ArgumentParser(
        description="Train the DQN agent on generated mazes."
    )
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--total-time-steps", type=int, default=20_000_000, help="The total time to call env.step(action)")
    parser.add_argument("--max-episode-steps", 
                        type=int, default=100_000, 
                        help="The maximum step to work on an episode in training. " \
                        "The episode will be truncated if the step limit used is larger than this parameter.")
    parser.add_argument("--max-episode-steps-eval", 
                        type=int, 
                        default=16, 
                        help="The maximum step to work on an episode in evaluation. " \
                            "The episode will be truncated if the step limit used is larger than this parameter.")
    parser.add_argument("--evaluation-episodes", type=int, default=100, help="The number of episodes used to evaluate.")
    parser.add_argument(
        "--evaluation-frequency",
        type=int,
        default=10_000,
        help="Evaluate every N training environment steps; 0 disables periodic evaluation.",
    )
    parser.add_argument(
        "--tensorboard-log-dir", "--tensorboard_log_dir",
        type=str,
        default=None,
        help="TensorBoard output directory (default: an automatic run directory under runs/).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="The size of data to sample in the replay buffer during each training")
    parser.add_argument("--learning-starts", type=int, default=2048)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=200_000)
    parser.add_argument("--training-freq", type=int, default=4)
    parser.add_argument("--grad-step-per-train", type=int, default=4)
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Fraction of online-network weights mixed into the target per update.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    return args


class TensorObservation(gym.ObservationWrapper):
    """Provide the baseline's single-environment float32 tensor input."""

    def __init__(self, env: Maze):
        super().__init__(env)
        input_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, input_dim), dtype=np.float32
        )

    def observation(self, observation):
        return torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)


def create_environments(
    width: int, height: int, seed: int, logger,
    max_episode_steps: int, max_episode_steps_eval: int,
) -> tuple[TensorObservation, TensorObservation]:
    """Create independent environments sharing one generated maze dataset."""
    training_environment = Maze(
        width=width, height=height,
        train_step_limitation=max_episode_steps,
        eval_step_limitation=max_episode_steps_eval,
    )
    evaluation_environment = Maze(
        width=width, height=height,
        train_step_limitation=max_episode_steps_eval,
        eval_step_limitation=max_episode_steps_eval,
    )
    logger.info("Generating maze dataset for %sx%s mazes.", width, height)
    training_environment.generate_maze(seed=seed)
    logger.info(f"The size of a training set is {training_environment.training_mazes.shape[0]}")
    logger.info(f"The size of a evaluation set is {training_environment.evaluation_mazes.shape[0]}")
    evaluation_environment.training_mazes = training_environment.training_mazes
    evaluation_environment.evaluation_mazes = training_environment.evaluation_mazes

    training_environment = TensorObservation(training_environment)
    evaluation_environment = TensorObservation(evaluation_environment)
    training_environment.reset(seed=seed, options={"is_evaluation": False})
    evaluation_environment.reset(seed=seed + 1, options={"is_evaluation": True})
    return training_environment, evaluation_environment

def main(argv: Sequence[str] | None = None) -> None:
    """Run one reproducible DQN training experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    args = parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    training_environment, evaluation_environment = create_environments(
        width=args.width,
        height=args.height,
        seed=args.seed,
        logger=logger,
        max_episode_steps=args.max_episode_steps,
        max_episode_steps_eval=args.max_episode_steps_eval,
    )
    try:
        agent = ActorDQNAgent(
            input_dim=training_environment.observation_space.shape[1],
            output_dim=training_environment.action_space.n,
            seed=args.seed,
            replay_buffer_size=args.replay_capacity,
            training_env=training_environment,
            eval_env=evaluation_environment,
            sample_batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            total_time_steps=args.total_time_steps,
            learning_start=args.learning_starts,
            training_freq=args.training_freq,
            grad_step_per_train=args.grad_step_per_train,
            num_eval_episodes=args.evaluation_episodes,
            eval_freq=args.evaluation_frequency or None,
            tensorboard_log_dir=args.tensorboard_log_dir,
        )
        logger.info(
            "Training DQN on %sx%s mazes for %s timesteps using CPU.",
            args.width, args.height, args.total_time_steps,
        )
        logger.info("TensorBoard logs: %s", agent.tensorboard_writer.log_dir)
        agent.train()
    finally:
        training_environment.close()
        evaluation_environment.close()


if __name__ == "__main__":
    main()
