"""Regression tests for the document-defined path-finding algorithm."""

import unittest
from collections import deque

import numpy as np

try:
    from ..env import Maze
    from ..util import EMPTY, GOAL, PLAYER, WALL
except ImportError:  # Support running the test file from the env directory.
    from env import Maze
    from util import EMPTY, GOAL, PLAYER, WALL


class PathfindingTests(unittest.TestCase):
    def setUp(self):
        # Avoid Maze.__init__, which also invokes the unfinished split and
        # variation modules that are outside this path-finding change.
        self.maze_environment = Maze.__new__(Maze)
        self.maze_environment.width = 3
        self.maze_environment.height = 3

    def test_matches_the_three_paths_in_the_3_by_3_document_example(self):
        mazes = self.maze_environment.pathfinding()

        matching_example = [
            maze
            for maze in mazes
            if maze[0, 0] == PLAYER and maze[0, 2] == GOAL
        ]

        self.assertEqual(len(matching_example), 3)
        actual_passages = {
            frozenset(map(tuple, np.argwhere(maze != WALL)))
            for maze in matching_example
        }
        expected_passages = {
            frozenset(((0, 0), (0, 1), (0, 2))),
            frozenset(((0, 0), (1, 0), (1, 1), (1, 2), (0, 2))),
            frozenset(
                ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2))
            ),
        }
        self.assertEqual(actual_passages, expected_passages)

    def test_every_generated_maze_has_exactly_one_simple_route(self):
        for maze in self.maze_environment.pathfinding():
            self.assertEqual(_count_simple_player_goal_routes(maze), 1)

    def test_result_shape_dtype_and_ordered_endpoint_coverage(self):
        mazes = self.maze_environment.pathfinding()

        self.assertEqual(mazes.shape, (188, 3, 3))
        self.assertEqual(mazes.dtype, np.int8)
        endpoint_pairs = {
            (
                tuple(np.argwhere(maze == PLAYER)[0]),
                tuple(np.argwhere(maze == GOAL)[0]),
            )
            for maze in mazes
        }
        self.assertEqual(len(endpoint_pairs), 3 * 3 * (3 * 3 - 1))


def _count_simple_player_goal_routes(maze):
    """Count simple routes in a tiny test maze, stopping after two routes."""
    start = tuple(np.argwhere(maze == PLAYER)[0])
    goal = tuple(np.argwhere(maze == GOAL)[0])
    passable = {EMPTY, PLAYER, GOAL}
    routes = 0
    pending = deque([(start, frozenset((start,)))])

    while pending and routes < 2:
        position, visited = pending.pop()
        if position == goal:
            routes += 1
            continue

        row, column = position
        for next_position in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            next_row, next_column = next_position
            if (
                0 <= next_row < maze.shape[0]
                and 0 <= next_column < maze.shape[1]
                and next_position not in visited
                and int(maze[next_position]) in passable
            ):
                pending.append((next_position, visited | {next_position}))

    return routes


if __name__ == "__main__":
    unittest.main()
