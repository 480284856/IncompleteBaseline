"""Tests for the Gymnasium maze interaction and observation encoding."""

import unittest

import numpy as np

try:
    from ..env import Maze
    from ..util import EMPTY, GOAL, PLAYER, WALL
except ImportError:  # Support discovery with ``-s env``.
    from env import Maze
    from util import EMPTY, GOAL, PLAYER, WALL


DIAGRAM_MAZE = np.array(
    [
        [EMPTY, WALL, WALL, GOAL],
        [WALL, PLAYER, WALL, EMPTY],
        [WALL, EMPTY, WALL, EMPTY],
        [WALL, EMPTY, EMPTY, EMPTY],
    ],
    dtype=np.int8,
)


class ObservationTests(unittest.TestCase):
    def test_observation_matches_the_diagram(self):
        environment = _environment_with_maze(DIAGRAM_MAZE)

        observation, info = environment.reset(seed=7)

        expected = np.array(
            [
                [1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(observation, expected)
        self.assertTrue(environment.observation_space.contains(observation))
        np.testing.assert_array_equal(info["agent_location"], [1, 1])
        np.testing.assert_array_equal(info["goal_location"], [0, 3])

    def test_rectangular_maze_is_padded_to_the_larger_dimension(self):
        maze = np.array(
            [[PLAYER, EMPTY, GOAL], [WALL, WALL, WALL]], dtype=np.int8
        )
        environment = _environment_with_maze(maze)

        observation, _ = environment.reset()

        self.assertEqual(observation.shape, (3, 7))
        np.testing.assert_array_equal(
            observation[:, :3],
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        )
        self.assertTrue(environment.observation_space.contains(observation))


class InteractionTests(unittest.TestCase):
    def test_wall_collision_keeps_the_agent_in_place(self):
        environment = _environment_with_maze(DIAGRAM_MAZE)
        environment.reset()

        _, reward, terminated, truncated, info = environment.step(1)

        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["invalid_move"])
        np.testing.assert_array_equal(info["agent_location"], [1, 1])

    def test_reaching_the_goal_terminates_the_episode(self):
        maze = np.array([[PLAYER, GOAL]], dtype=np.int8)
        environment = _environment_with_maze(maze)
        environment.reset()

        observation, reward, terminated, truncated, info = environment.step(1)

        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["invalid_move"])
        np.testing.assert_array_equal(info["agent_location"], [0, 1])
        self.assertTrue(environment.observation_space.contains(observation))

    def test_reset_can_select_the_evaluation_set(self):
        training_maze = np.array([[PLAYER, GOAL]], dtype=np.int8)
        evaluation_maze = np.array([[GOAL, PLAYER]], dtype=np.int8)
        environment = Maze(width=2, height=1)
        environment.training_mazes = training_maze[None, ...]
        environment.evaluation_mazes = evaluation_maze[None, ...]

        _, info = environment.reset(options={"is_evaluation": True})

        np.testing.assert_array_equal(info["agent_location"], [0, 1])
        np.testing.assert_array_equal(info["goal_location"], [0, 0])

    def test_step_before_reset_is_rejected(self):
        environment = Maze(width=2, height=1)

        with self.assertRaisesRegex(RuntimeError, "reset"):
            environment.step(0)


def _environment_with_maze(maze: np.ndarray) -> Maze:
    environment = Maze(width=maze.shape[1], height=maze.shape[0])
    environment.training_mazes = maze[None, ...]
    environment.evaluation_mazes = maze[None, ...]
    return environment


if __name__ == "__main__":
    unittest.main()
