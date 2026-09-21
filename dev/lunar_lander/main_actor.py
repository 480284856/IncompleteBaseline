"""Train and evaluate the ActorDQN agent on CartPole."""

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

# Allow both package execution and direct execution of this debug entry point.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from IncompleteBaseline.algorithms.actordqn.actordqn_agent4lunarlander import (
        ActorDQNAgent4LunarLander,
    )
else:
    from ...algorithms.actordqn.actordqn_agent4lunarlander import ActorDQNAgent4LunarLander


class TensorObservation(gym.ObservationWrapper):
    """Flatten CartPole observations into the tensor shape expected by the agent."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        input_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=np.asarray(env.observation_space.low).reshape(1, input_dim),
            high=np.asarray(env.observation_space.high).reshape(1, input_dim),
            dtype=np.float32,
        )

    def observation(self, observation):
        return torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a training run."""
    parser = argparse.ArgumentParser(
        description="Train the Actor DQN agent on cart pole."
    )

    parser.add_argument("--total-time-steps", type=int, default=20_000_000, help="The total time to call env.step(action)")
    parser.add_argument("--max-episode-steps", 
                        type=int, default=10_000, 
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

    parser.add_argument("--training-freq", type=int, default=1)
    parser.add_argument("--grad-step-per-train", type=int, default=1)

    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Fraction of online-network weights mixed into the target per update.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--debug-log-interval",
        type=int,
        default=1000,
        help="When --debug is set, write per-step debug scalars every N steps.",
    )
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=None, help="Hidden layer sizes.")

    args = parser.parse_args(argv)

    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    return args


def create_environments(
    seed: int, 
    max_episode_steps: int, max_episode_steps_eval: int,
) -> tuple[TensorObservation, TensorObservation]:
    """Create independent training and evaluation CartPole environments."""
    training_environment = TensorObservation(
        gym.make("LunarLander-v3", max_episode_steps=max_episode_steps)
    )
    evaluation_environment = TensorObservation(
        gym.make("LunarLander-v3", max_episode_steps=max_episode_steps_eval)
    )

    training_environment.reset(seed=seed)
    evaluation_environment.reset(seed=seed + 1)
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
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        max_episode_steps_eval=args.max_episode_steps_eval,
    )
    try:
        agent = ActorDQNAgent4LunarLander(
            seed=args.seed,
            input_dim=training_environment.observation_space.shape[1],
            output_dim=training_environment.action_space.n,
            hidden_sizes=args.hidden_sizes,

            sample_batch_size=args.batch_size,
            replay_buffer_size=args.replay_capacity,
            gamma=args.gamma,
            tau=args.tau,

            training_env=training_environment,
            eval_env=evaluation_environment,
            total_time_steps=args.total_time_steps,
            learning_start=args.learning_starts,
            training_freq=args.training_freq,
            grad_step_per_train=args.grad_step_per_train,
            num_eval_episodes=args.evaluation_episodes,
            eval_freq=args.evaluation_frequency or None,
            tensorboard_log_dir=args.tensorboard_log_dir,

            debug=args.debug,
            debug_log_interval=args.debug_log_interval,
        )
        logger.info(
            "Training ActorDQN on LunarLander-V3 for %s timesteps using CPU.",
            args.total_time_steps,
        )
        logger.info("TensorBoard logs: %s", agent.tensorboard_writer.log_dir)
        agent.train()
    finally:
        training_environment.close()
        evaluation_environment.close()


if __name__ == "__main__":
    main()
