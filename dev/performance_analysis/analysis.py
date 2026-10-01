"""Evaluate saved 4x4 and 5x5 maze policies on a full maze split."""

import csv
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from IncompleteBaseline.algorithms.common.qnetwork import QNetwork
from IncompleteBaseline.envs.procedual_maze.env import Maze


LOGS = Path(__file__).resolve().parents[1] / "logs"
RESULTS = Path(__file__).with_name("results.csv")
NUM_ACTIONS = 4


class TensorObservation(gym.ObservationWrapper):
    """Flatten a maze observation to the shape expected by QNetwork."""

    def __init__(self, env: Maze):
        super().__init__(env)
        input_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, input_dim), dtype=np.float32
        )

    def observation(self, observation):
        return torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)


def load_model(path: Path | None, input_dim: int) -> QNetwork | None:
    if path is None:
        return None
    model = QNetwork(input_dim, NUM_ACTIONS)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def get_policy(model: QNetwork | None, state: torch.Tensor, policy: str,
               rng: torch.Generator) -> int:
    if policy == "random":
        return int(torch.randint(NUM_ACTIONS, (1,), generator=rng).item())
    with torch.no_grad():
        values = model(state)
        if policy == "actor":
            probabilities = torch.softmax(values, dim=1)
            return int(torch.multinomial(probabilities, 1, generator=rng).item())
        return int(values.argmax(dim=1).item())


def create_eval_environment(width: int, seed: int, max_episode_steps: int,
                            logger) -> TensorObservation:
    env = Maze(
        width=width, height=width,
        train_step_limitation=max_episode_steps,
        eval_step_limitation=max_episode_steps,
    )
    logger.info("Generating %sx%s mazes for seed %s", width, width, seed)
    env.generate_maze(seed=seed)
    return TensorObservation(env)


def final_evaluation(model: QNetwork | None, env: TensorObservation,
                     policy: str, rng: torch.Generator,
                     is_evaluation: bool = True) -> dict:
    mazes = env.unwrapped.evaluation_mazes if is_evaluation else env.unwrapped.training_mazes
    num_mazes = len(mazes)
    solved = 0
    total_steps = 0
    successful_steps = 0

    for maze_index in range(num_mazes):
        state, _ = env.reset(options={
            "is_evaluation": is_evaluation,
            "maze_index": maze_index,
        })
        steps = 0
        while True:
            action = get_policy(model, state, policy, rng)
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        total_steps += steps
        if terminated:
            solved += 1
            successful_steps += steps

    return {
        "num_mazes": num_mazes,
        "success_rate": solved / num_mazes,
        "mean_steps_all": total_steps / num_mazes,
        "mean_steps_success": successful_steps / solved if solved else None,
    }


def performance_analysis(is_evaluation: bool = True, output_path: Path = RESULTS) -> None:
    logger = logging.getLogger(__name__)
    fields = ["maze", "seed", "model", "variant", "policy", "num_mazes",
              "success_rate", "mean_steps_all", "mean_steps_success", "checkpoint"]

    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for size, max_steps in ((4, 160), (5, 250)):
            maze = f"maze{size}x{size}"
            for version in (1, 2, 3):
                seed = 41 + version
                env = create_eval_environment(size, seed, max_steps, logger)
                input_dim = env.observation_space.shape[1]
                dqn_dir = LOGS / "qdn" / maze
                if size == 5:
                    dqn_dir /= "normal"
                actor_dir = LOGS / "actorqdn" / maze
                runs = [
                    ("DQN", "normal", "greedy",
                     dqn_dir / f"v{version}" / "model/q_network.pt"),
                    ("ActorDQN", "normal", "actor",
                     actor_dir / "normal" / f"v{version}" / "model/actor_network.pt"),
                    ("ActorDQN", "random_baseline_checkpoint", "actor",
                     actor_dir / "random_baseline" / f"v{version}" / "model/actor_network.pt"),
                    ("Random", "uniform", "random", None),
                ]

                try:
                    for model_name, variant, policy, checkpoint in runs:
                        model = load_model(checkpoint, input_dim)
                        rng = torch.Generator().manual_seed(seed)
                        metrics = final_evaluation(model, env, policy, rng, is_evaluation)
                        row = {
                            "maze": maze, "seed": seed, "model": model_name,
                            "variant": variant, "policy": policy,
                            **metrics, "checkpoint": str(checkpoint) if checkpoint else "",
                        }
                        writer.writerow(row)
                        output.flush()
                        print(row, flush=True)
                finally:
                    env.close()

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.set_num_threads(1)
    performance_analysis()
