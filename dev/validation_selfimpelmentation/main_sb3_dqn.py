"""Train a Stable-Baselines3 DQN agent on the procedural maze environment.

The maze splits its generated mazes into a training pool and a held-out
evaluation pool and samples one on every reset. All evaluation metrics,
including the success rate, come from SB3's own EvalCallback: this script only
pins the requested split and surfaces the goal signal as info["is_success"].
Exploration is SB3's linear epsilon schedule rather than a fixed epsilon.

Run from the repository root with the rl conda environment:

    /Users/jay/miniconda3/envs/rl/bin/python -m IncompleteBaseline.dev.validation_selfimpelmentation.main_sb3_dqn
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

# Allow both package execution and direct execution of this debug entry point.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from IncompleteBaseline.envs.procedual_maze.env import Maze
else:
    from ...envs.procedual_maze.env import Maze


# Defaults mirror IncompleteBaseline/dev/main.py and dev/maze4x4/run_dqn_91.sh.
WIDTH = 5
HEIGHT = 5
TOTAL_TIMESTEPS = 819_200
LEARNING_STARTS = 409_600       # Random warmup, then online learning for the second half.
BUFFER_SIZE = 500_000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005                     # Soft target update, applied every env step.
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
LEARNING_RATE = 1e-3
NET_ARCH = [128, 128, 64, 32]
EXPLORATION_INITIAL_EPS = 1.0   # SB3 anneals epsilon linearly between these two
EXPLORATION_FINAL_EPS = 0.05    # values over EXPLORATION_FRACTION of training. With
EXPLORATION_FRACTION = 1.0      # LEARNING_STARTS at the midpoint, the anneal spans the
MAX_EPISODE_STEPS = 2048        # whole run and stays active during the learning phase.
MAX_EPISODE_STEPS_EVAL = 128    # Reference maze horizon (dev/maze5x5/run_dqn_91.sh).
EVAL_EPISODES = 100
EVAL_FREQ = 10_000
SEED = 42
LOG_DIR = "IncompleteBaseline/dev/validation_selfimpelmentation/logs/env/sb3_dqn/maze5x5/v1"


class MazeSplitEvalEnv(gym.Wrapper):
    """Pin one maze pool and expose the success flag SB3's EvalCallback reads.

    EvalCallback resets without options, which would otherwise let the maze
    sample the training pool, so is_evaluation is forced on every reset. An
    episode is a success when the goal is reached (terminated), which SB3 turns
    into success_rate from info["is_success"].
    """

    def __init__(self, env: Maze, *, is_evaluation: bool) -> None:
        super().__init__(env)
        self.is_evaluation = is_evaluation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        env_options = dict(options or {})
        env_options["is_evaluation"] = self.is_evaluation
        return self.env.reset(seed=seed, options=env_options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["is_success"] = bool(terminated)
        return observation, reward, terminated, truncated, info


class MazeFeatureExtractor(BaseFeaturesExtractor):
    """Match the reference maze Q-network's LayerNorm MLP trunk for SB3."""

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, features_dim=NET_ARCH[-1])
        previous_width = int(np.prod(observation_space.shape))
        layers: list[nn.Module] = []
        for width in NET_ARCH:
            layers.extend((nn.Linear(previous_width, width), nn.LayerNorm(width), nn.ReLU()))
            previous_width = width
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(torch.flatten(observations, start_dim=1))


class _PrefixedLogger:
    """Logger proxy that moves an EvalCallback's eval/ keys under a prefix."""

    def __init__(self, logger, prefix: str) -> None:
        self._logger = logger
        self._prefix = prefix

    def record(self, key, value, exclude=None) -> None:
        if key.startswith("eval/"):
            key = self._prefix + "/" + key[len("eval/"):]
        self._logger.record(key, value, exclude)

    def __getattr__(self, name):
        return getattr(self._logger, name)


class SplitEvalCallback(EvalCallback):
    """EvalCallback that logs under metric_prefix instead of hard-coded eval/.

    SB3 hard-codes eval/, so without a prefix the training-split and held-out
    evaluations would overwrite each other in the same logger. The evaluation
    itself, including success_rate, is entirely SB3's.
    """

    def __init__(self, *args, metric_prefix: str = "eval", **kwargs) -> None:
        self.metric_prefix = metric_prefix
        super().__init__(*args, **kwargs)

    @property
    def logger(self):
        return _PrefixedLogger(self.model.logger, self.metric_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--total-timesteps", "--total-time-steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--exploration-initial-eps", type=float, default=EXPLORATION_INITIAL_EPS)
    parser.add_argument("--exploration-final-eps", type=float, default=EXPLORATION_FINAL_EPS)
    parser.add_argument(
        "--exploration-fraction", type=float, default=EXPLORATION_FRACTION,
        help="Fraction of training over which epsilon anneals to its final value.",
    )
    parser.add_argument("--replay-capacity", type=int, default=BUFFER_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--max-episode-steps-eval", type=int, default=MAX_EPISODE_STEPS_EVAL)
    parser.add_argument("--evaluation-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--evaluation-frequency", type=int, default=EVAL_FREQ)
    parser.add_argument("--tensorboard-log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def create_maze_environments(
    *,
    seed: int,
    width: int,
    height: int,
    train_step_limit: int,
    eval_step_limit: int,
) -> tuple[gym.Env, gym.Env, gym.Env]:
    """Build the training env plus the held-out and training-split eval envs.

    Maze generation is expensive, so the three environments share the dataset
    generated once by the training environment.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating the %sx%s maze dataset (this can take a while).", width, height)

    base = Maze(
        width=width,
        height=height,
        train_step_limitation=train_step_limit,
        eval_step_limitation=eval_step_limit,
    )
    base.generate_maze(seed=seed)
    logger.info("Training mazes: %d", base.training_mazes.shape[0])
    logger.info("Evaluation mazes: %d", base.evaluation_mazes.shape[0])
    training_mazes, evaluation_mazes = base.training_mazes, base.evaluation_mazes

    def eval_split(is_evaluation: bool) -> Monitor:
        env = Maze(
            width=width,
            height=height,
            train_step_limitation=eval_step_limit,
            eval_step_limitation=eval_step_limit,
        )
        env.training_mazes, env.evaluation_mazes = training_mazes, evaluation_mazes
        return Monitor(MazeSplitEvalEnv(env, is_evaluation=is_evaluation))

    return Monitor(base), eval_split(True), eval_split(False)


def build_model(env: gym.Env, args: argparse.Namespace) -> DQN:
    """Create the SB3 DQN with the reference maze hyperparameters."""
    if args.learning_starts < args.batch_size:
        raise ValueError("learning_starts must be at least batch_size")
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.replay_capacity,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        max_grad_norm=10,          # reference implementation does no clipping
        target_update_interval=1,  # reference soft-updates after each step
        # SB3's linear epsilon annealing, replacing fixed epsilon-greedy.
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        # The extractor is the LayerNorm trunk; SB3 adds only its linear Q head.
        policy_kwargs={"features_extractor_class": MazeFeatureExtractor, "net_arch": []},
        tensorboard_log=args.tensorboard_log_dir,
        seed=args.seed,
        device="cpu",
        verbose=1,
    )


def report_success_rate(callback: SplitEvalCallback) -> float:
    """Mean success flag of the callback's most recent SB3 evaluation."""
    successes = callback.evaluations_successes
    return float(np.mean(successes[-1])) if successes else float("nan")


def main() -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_env, eval_env, train_eval_env = create_maze_environments(
        seed=args.seed,
        width=args.width,
        height=args.height,
        train_step_limit=args.max_episode_steps,
        eval_step_limit=args.max_episode_steps_eval,
    )
    log_dir = Path(args.tensorboard_log_dir)

    try:
        model = build_model(train_env, args)
        eval_callback = SplitEvalCallback(
            eval_env,
            n_eval_episodes=args.evaluation_episodes,
            eval_freq=args.evaluation_frequency,
            deterministic=True,
            best_model_save_path=str(log_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            metric_prefix="eval",
        )
        train_eval_callback = SplitEvalCallback(
            train_eval_env,
            n_eval_episodes=args.evaluation_episodes,
            eval_freq=args.evaluation_frequency,
            deterministic=True,
            log_path=str(log_dir / "train_eval"),
            metric_prefix="training",
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList([eval_callback, train_eval_callback]),
        )
    finally:
        train_env.close()
        eval_env.close()
        train_eval_env.close()

    # SB3 already recorded the last evaluation for each split; just surface it.
    metrics = {
        "eval_success_rate": report_success_rate(eval_callback),
        "training_success_rate": report_success_rate(train_eval_callback),
    }
    for name, value in metrics.items():
        model.logger.record(f"final_eval/{name}", value)
    model.logger.dump(model.num_timesteps)
    print(
        "Final SB3 evaluation: "
        f"success={metrics['eval_success_rate']:.1%} (held-out), "
        f"{metrics['training_success_rate']:.1%} (training split)"
    )
    return metrics


if __name__ == "__main__":
    main()
