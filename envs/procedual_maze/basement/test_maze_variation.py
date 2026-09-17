"""Regression tests for solution-preserving maze variation."""

import unittest

import numpy as np

try:
    from ..env import Maze
    from ..util import EMPTY, GOAL, PLAYER, WALL, find_distances, find_shortest_path
except ImportError:  # Support running the test file from the env directory.
    from env import Maze
    from util import EMPTY, GOAL, PLAYER, WALL, find_distances, find_shortest_path


class MazeVariationTests(unittest.TestCase):
    def setUp(self):
        self.maze_environment = Maze.__new__(Maze)
        self.maze_environment.width = 4
        self.maze_environment.height = 4
        self.original = np.array(
            [
                [PLAYER, EMPTY, EMPTY, GOAL],
                [WALL, WALL, WALL, WALL],
                [WALL, WALL, WALL, WALL],
                [WALL, WALL, WALL, WALL],
            ],
            dtype=np.int8,
        )

    def test_preserves_the_original_unique_shortest_solution(self):
        variations = self.maze_environment.maze_variation(
            self.original[np.newaxis, ...],
            random_state=2,
        )

        self.assertEqual(variations.shape, (1, 4, 4))
        self.assertEqual(variations.dtype, np.int8)
        self.assertEqual(
            find_shortest_path(variations[0], (0, 0), (0, 3)),
            ((0, 0), (0, 1), (0, 2), (0, 3)),
        )
        self.assertEqual(_count_shortest_routes(variations[0]), 1)

    def test_is_reproducible_and_does_not_modify_the_source(self):
        source = self.original[np.newaxis, ...]
        unchanged = source.copy()

        first = self.maze_environment.maze_variation(source, random_state=7)
        second = self.maze_environment.maze_variation(source, random_state=7)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(source, unchanged)

    def test_keeps_safe_opened_cells_to_make_the_maze_more_complex(self):
        variation = self.maze_environment.maze_variation(
            self.original[np.newaxis, ...],
            random_state=2,
        )[0]

        self.assertGreater(np.count_nonzero(variation == EMPTY), 2)
        self.assertLess(np.count_nonzero(variation == WALL), 12)

    def test_rejects_a_source_with_passages_outside_its_original_path(self):
        invalid = self.original.copy()
        invalid[3, 3] = EMPTY

        with self.assertRaisesRegex(ValueError, "only its original"):
            self.maze_environment.maze_variation(
                invalid[np.newaxis, ...],
                random_state=1,
            )

    def test_distances_record_but_do_not_cross_the_opposite_endpoint(self):
        maze = np.array(
            [[EMPTY, PLAYER, EMPTY, GOAL, EMPTY]],
            dtype=np.int8,
        )

        distances_from_player = find_distances(maze, (0, 1))
        distances_from_goal = find_distances(maze, (0, 3))

        self.assertEqual(distances_from_player[(0, 3)], 2)
        self.assertNotIn((0, 4), distances_from_player)
        self.assertEqual(distances_from_goal[(0, 1)], 2)
        self.assertNotIn((0, 0), distances_from_goal)

    def test_distances_require_the_player_or_goal_as_the_start(self):
        with self.assertRaisesRegex(ValueError, "player or goal"):
            find_distances(self.original, (0, 1))


def _count_shortest_routes(maze):
    """Count shortest player-goal routes using the BFS distance layers."""
    start = tuple(map(int, np.argwhere(maze == PLAYER)[0]))
    goal = tuple(map(int, np.argwhere(maze == GOAL)[0]))
    distances = find_distances(maze, start)
    routes = {start: 1}

    for position, distance in sorted(distances.items(), key=lambda item: item[1]):
        if position == start:
            continue
        row, column = position
        routes[position] = sum(
            routes.get(neighbor, 0)
            for neighbor in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            )
            if distances.get(neighbor) == distance - 1
        )
    return routes.get(goal, 0)


if __name__ == "__main__":
    unittest.main()
