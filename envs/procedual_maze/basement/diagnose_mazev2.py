"""Diagnostic checks for MazeV2 and the module-level maze-generation helpers.

Run from the repository with the rl conda environment:
    /Users/jay/miniconda3/envs/rl/bin/python envs/procedual_maze/basement/diagnose_mazev2.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from env import (  # noqa: E402
    Maze,
    MazeV2,
    generate_maze,
    maze_variation,
    pathfinding,
    train_test_split,
)
from util import GOAL, PLAYER  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        RESULTS.append(("PASS", name, str(detail)))
    except Exception as exc:  # noqa: BLE001 - diagnostic harness
        RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))


TRAIN, EVAL = generate_maze(seed=42, split_ratio=0.2, width=4, height=4)


def c_dataset_dtype() -> str:
    assert TRAIN.dtype == np.int8, TRAIN.dtype
    assert EVAL.dtype == np.int8, EVAL.dtype
    return f"train={TRAIN.shape} {TRAIN.dtype}, eval={EVAL.shape} {EVAL.dtype}"


def c_dataset_wellformed() -> str:
    for pool, label in ((TRAIN, "train"), (EVAL, "eval")):
        for i, maze in enumerate(pool):
            p, g = (maze == PLAYER).sum(), (maze == GOAL).sum()
            assert p == 1 and g == 1, f"{label}[{i}] player={p} goal={g}"
    return "every maze has exactly one player and one goal"


def c_split_ratio() -> str:
    total = len(TRAIN) + len(EVAL)
    return f"total={total}, test_ratio={len(EVAL) / total:.3f}"


def c_pathfinding_count() -> str:
    m = pathfinding(4, 4)
    return f"pathfinding(4,4).shape={m.shape}, endpoint pairs covered={m.shape[0]}"


def c_ordered_pairs() -> str:
    from util import iter_start_goal_pairs

    cells = list(iter_start_goal_pairs(4, 4))
    expected_ordered = 16 * 15
    return f"iter_start_goal_pairs(4,4) yields {len(cells)} (ordered pairs would be {expected_ordered})"


def c_mazev2_construct() -> str:
    env = MazeV2(width=4, height=4, step_limitation=16, dataset=TRAIN)
    return f"obs_space={env.observation_space.shape}, act_space={env.action_space}, step_limitation={env.step_limitation}"


def c_mazev2_reset() -> str:
    env = MazeV2(width=4, height=4, step_limitation=16, dataset=TRAIN)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), obs.shape
    assert obs.dtype == np.float32, obs.dtype
    assert set(info) == {"agent_location", "goal_location", "distance", "invalid_move"}, info
    return f"obs={obs.shape} in space, info keys ok"


def c_mazev2_step() -> str:
    env = MazeV2(width=4, height=4, step_limitation=16, dataset=TRAIN)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    return f"reward={reward} terminated={terminated} truncated={truncated}"


def c_mazev2_empty_dataset() -> str:
    empty = np.empty((0, 4, 4), dtype=np.int8)
    env = MazeV2(width=4, height=4, step_limitation=16, dataset=empty)
    env.reset(seed=0)
    return "reset on an empty dataset returned an episode (unexpected)"


def c_mazev2_options_ignored() -> str:
    env = MazeV2(width=4, height=4, step_limitation=16, dataset=TRAIN)
    obs, info = env.reset(seed=0, options={"maze_index": 0, "is_evaluation": True})
    first = np.array_equal(info["goal_location"], np.argwhere(TRAIN[0] == GOAL)[0])
    return f"reset options accepted silently; maze_index=0 honoured: {first}"


def c_mazev2_no_dataset() -> str:
    env = MazeV2(width=4, height=4, step_limitation=16)
    obs, info = env.reset(seed=0)
    return f"reset without dataset returned obs={obs.shape} (unexpected)"


def c_mazev2_truncation() -> str:
    env = MazeV2(width=4, height=4, step_limitation=3, dataset=TRAIN)
    env.reset(seed=0)
    for i in range(4):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            return f"terminated={terminated} truncated={truncated} after step {i + 1}"
    return "never truncated after 4 steps with step_limitation=3"


def c_mazev2_observation_encoding() -> str:
    maze = np.array([[PLAYER, 0, GOAL]], dtype=np.int8)
    env = MazeV2(width=3, height=1, step_limitation=5, dataset=maze[None, ...])
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 7), obs.shape
    assert obs[0, 0] == 1 and obs[0, 2] == 1, obs
    return f"obs[0,:3]={obs[0, :3].tolist()} padding rows zero={np.all(obs[1:] == 0)}"


def c_mazev2_seed_reproducible() -> str:
    a = MazeV2(width=4, height=4, step_limitation=16, dataset=TRAIN)
    b = MazeV2(width=4, height=4, step_limitation=16, dataset=TRAIN)
    oa, ia = a.reset(seed=123)
    ob, ib = b.reset(seed=123)
    return f"same seed -> same maze: {np.array_equal(oa, ob)}"


def c_maze_still_works() -> str:
    env = Maze(width=3, height=1)
    env.training_mazes = np.array([[[PLAYER, 0, GOAL]]], dtype=np.int8)
    env.evaluation_mazes = env.training_mazes
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(1)
    return f"reward={reward} terminated={terminated} truncated={truncated}"


check("generate_maze dtype/shape", c_dataset_dtype)
check("generate_maze pools well-formed", c_dataset_wellformed)
check("train/eval split ratio", c_split_ratio)
check("pathfinding(4,4) count", c_pathfinding_count)
check("iter_start_goal_pairs ordered coverage", c_ordered_pairs)
check("MazeV2 constructor", c_mazev2_construct)
check("MazeV2.reset", c_mazev2_reset)
check("MazeV2.step", c_mazev2_step)
check("MazeV2 empty-dataset error path", c_mazev2_empty_dataset)
check("MazeV2 reset options handling", c_mazev2_options_ignored)
check("MazeV2 without dataset", c_mazev2_no_dataset)
check("MazeV2 truncation via step_limitation", c_mazev2_truncation)
check("MazeV2 observation encoding", c_mazev2_observation_encoding)
check("MazeV2 seeded reset reproducibility", c_mazev2_seed_reproducible)
check("Maze (v1) still works", c_maze_still_works)

width = max(len(r[1]) for r in RESULTS)
for status, name, detail in RESULTS:
    print(f"[{status}] {name:<{width}}  {detail}")

failed = [r for r in RESULTS if r[0] == "FAIL"]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
