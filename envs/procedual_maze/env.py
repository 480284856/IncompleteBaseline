from typing import Tuple

import numpy as np
import gymnasium as gym
import logging

try:
    from .util import (
        EMPTY,
        GOAL,
        PLAYER,
        WALL,
        generate_all_single_solution_mazes,
        generate_maze_variations,
    )
except ImportError:  # Support running this file directly.
    from util import (
        EMPTY,
        GOAL,
        PLAYER,
        WALL,
        generate_all_single_solution_mazes,
        generate_maze_variations,
    )


class Maze(gym.Env):
    def __init__(self, 
                 width=5, 
                 height=5,
                 eval_step_limitation:int=10_000,
                 train_step_limitation:int=10_000,):
        '''
        Args:
            eval_step_limitation: In evaluation, The maximum number of steps allowed per episode.
            train_step_limitation: In training, The maximum number of steps allowed per episode.
        '''
        self.logger = logging.getLogger(__name__)

        if not isinstance(width, (int, np.integer)) or not isinstance(
            height, (int, np.integer)
        ):
            raise TypeError("width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if width * height < 2:
            raise ValueError("a maze needs at least two cells")

        self.width = int(width)
        self.height = int(height)

        # The diagram places the walkability grid beside four one-hot vectors:
        # agent row, agent column, goal row, and goal column. 

        # Square mazes use the diagram's exact (size, size + 4) layout. 
        # (Though it won't appear) Rectangular mazes are zero-padded to their larger dimension so all four vectors fit.
        self._observation_size = max(self.width, self.height)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._observation_size, self._observation_size + 4),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([0, 1], dtype=np.int32),   # right
            2: np.array([1, 0], dtype=np.int32),   # down
            3: np.array([0, -1], dtype=np.int32),  # left
        }

        self.current_maze = None
        self._agent_location = None
        self._goal_location = None

        assert eval_step_limitation >= 1
        assert train_step_limitation >= 1
        self.eval_step_limitation = eval_step_limitation
        self.train_step_limitation=train_step_limitation

    def generate_maze(self, seed: int = 42) -> None:
        '''
        Here is the description of the maze generation process:
            https://ljp0vuj4fr1r.jp.larksuite.com/wiki/QyQJwfSRBiuvXZk8kXpjTxKmpEe?from=from_copylink

        Cell meanings:
            0 = empty
            1 = wall
            2 = player
            3 = goal
        '''
        original_set = self.pathfinding()
        original_training_set, original_evaluation_set = self.train_test_split(original_set, test_size=0.2, random_state=seed)

        self.training_mazes = self.maze_variation(original_training_set, random_state=seed)
        self.evaluation_mazes = self.maze_variation(original_evaluation_set, random_state=seed)

    def pathfinding(self) -> np.ndarray:
        """
        Generate every maze whose open cells form one unique player-goal path.

        For detailed description, please refer to the following link:
            https://ljp0vuj4fr1r.jp.larksuite.com/wiki/QyQJwfSRBiuvXZk8kXpjTxKmpEe#share-SDypdpOnto1ZJ1xoNO8jeEj5pIc

        Returns:
            3D numpy array of shape (num_mazes, height, width) containing the generated mazes.
        """
        return generate_all_single_solution_mazes(self.width, self.height)

    def train_test_split(self, mazes: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the generated mazes into training and evaluation sets.

        Args:
            mazes: 3D numpy array of shape (num_mazes, height, width) containing the generated mazes.
            test_size: Proportion of the dataset to include in the evaluation set.
        """
        rng = np.random.default_rng(random_state)
        num_mazes = mazes.shape[0]
        indices = rng.permutation(num_mazes)
        split_idx = int(num_mazes * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        return mazes[train_indices], mazes[test_indices]

    def maze_variation(
        self,
        mazes: np.ndarray,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Generate variations of the given mazes to increase diversity.

        For detailed description, please refer to the following link:
            https://ljp0vuj4fr1r.jp.larksuite.com/wiki/QyQJwfSRBiuvXZk8kXpjTxKmpEe#share-K32KdcnikoxKdoxjQ5wjFpecpJe

        Args:
            mazes: 3D numpy array of shape (num_mazes, height, width) containing the generated mazes.
            random_state: Optional seed for reproducible variations.

        Returns:
            3D numpy array of shape (num_variations, height, width) containing the varied mazes.
        """
        return generate_maze_variations(
            mazes,
            expected_shape=(self.height, self.width),
            random_state=random_state,
        )

    def _get_observation(self) -> np.ndarray:
        """Encode the current maze into a 2D array."""
        if self.current_maze is None:
            raise RuntimeError("reset must be called before requesting an observation")

        # (height, width + 4) layout for square mazes.
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Empty, player, and goal cells are walkable; only walls encode as 0.
        # Copy the walkability grid into the left side of the observation array.
        # All cells that don't equal WALL will be encoded as 1.0 to the corresponding cell in the observation array.
        observation[: self.height, : self.width] = (
            self.current_maze != WALL
        ).astype(np.float32)

        agent_row, agent_column = self._agent_location
        goal_row, goal_column = self._goal_location
        coordinate_offset = self._observation_size
        observation[agent_row, coordinate_offset] = 1.0
        observation[agent_column, coordinate_offset + 1] = 1.0
        observation[goal_row, coordinate_offset + 2] = 1.0
        observation[goal_column, coordinate_offset + 3] = 1.0
        return observation

    # Keep the conventional short helper name available to callers.
    # a method alias: _get_obs becomes another name for _get_observation
    _get_obs = _get_observation

    def _get_info(self, invalid_move: bool = False) -> dict:
        """Return useful episode state information."""
        return {
            "agent_location": self._agent_location.copy(),
            "goal_location": self._goal_location.copy(),
            "distance": int(
                np.abs(self._agent_location - self._goal_location).sum()
            ),
            "invalid_move": invalid_move,
        }

    def reset(self, seed=None, options=None):
        """
        Sample a maze from the training set or evaluation set.
        """
        super().reset(seed=seed)

        # used for truncation
        self.step_elapsed = 0

        self.is_evaluation = False
        if options is not None:
            self.is_evaluation = options.get("is_evaluation", self.is_evaluation)


        # Maze generation can be expensive, so it remains lazy and happens
        # only when a fresh environment is reset for the first time.
        if not hasattr(self, "training_mazes") or not hasattr(
            self, "evaluation_mazes"
        ):
            maze_generator_code = self.generate_maze.__func__.__code__
            self.logger.warning(
                "[INFO] Maze generation is starting and may take a while (%s:%d).",
                maze_generator_code.co_filename,
                maze_generator_code.co_firstlineno,
            )
            self.generate_maze(seed=seed)

        maze_pool = self.evaluation_mazes if self.is_evaluation else self.training_mazes
        if len(maze_pool) == 0:
            split_name = "evaluation" if self.is_evaluation else "training"
            raise RuntimeError(f"the {split_name} maze set is empty")

        # self.np_random is controled by the gymnasium.Env superclass and is seeded by the reset() call.
        # select a random integer from the range [0, len(maze_pool))
        maze_index = int(self.np_random.integers(len(maze_pool)))
        self.current_maze = np.asarray(maze_pool[maze_index], dtype=np.int8).copy()
        if self.current_maze.shape != (self.height, self.width):
            raise ValueError(
                "sampled maze shape must be "
                f"{(self.height, self.width)}, got {self.current_maze.shape}"
            )

        player_positions = np.argwhere(self.current_maze == PLAYER)
        goal_positions = np.argwhere(self.current_maze == GOAL)
        if len(player_positions) != 1 or len(goal_positions) != 1:
            raise ValueError("each maze must contain exactly one player and one goal")

        self._agent_location = player_positions[0].astype(np.int32)
        self._goal_location = goal_positions[0].astype(np.int32)

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Execute an action and return observation, reward, termination flags, and info.

        Actions are 0=up, 1=right, 2=down, and 3=left. Attempting to leave
        the maze or enter a wall keeps the agent in its current cell.
        """
        if self.current_maze is None:
            raise RuntimeError("reset must be called before step")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer from 0 to 3, got {action!r}")

        action = int(action)
        candidate = self._agent_location + self._action_to_direction[action]
        # Check if the candidate position is within the maze bounds.
        # The parentheses are just used for formatting where 'and' statement
        # can be split into multiple lines for better readability.
        in_bounds = (
            0 <= candidate[0] < self.height
            and 0 <= candidate[1] < self.width
        )


        invalid_move = not in_bounds or (
            in_bounds and self.current_maze[tuple(candidate)] == WALL
        )

        if not invalid_move:
            previous_location = tuple(self._agent_location)
            if self.current_maze[previous_location] == PLAYER:
                self.current_maze[previous_location] = EMPTY

            self._agent_location = candidate.astype(np.int32)
            if not np.array_equal(self._agent_location, self._goal_location):
                self.current_maze[tuple(self._agent_location)] = PLAYER

        terminated = bool(
            np.array_equal(self._agent_location, self._goal_location)
        )

        self.step_elapsed += 1
        step_limit = (
            self.eval_step_limitation
            if self.is_evaluation
            else self.train_step_limitation
        )
        truncated = self.step_elapsed >= step_limit

        reward = 1.0 if terminated else -0.01 if not invalid_move else -0.05

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(invalid_move=invalid_move),
        )
    
if __name__ == "__main__":
    from visualization import plot_maze

    #os.mkdir("mazes_pictures-4x4") if not os.path.exists("mazes_pictures-4x4") else None
    #os.chdir("mazes_pictures-4x4")
    #env = Maze(width=6, height=6) # Total number of mazes generated: 669072
    #env = Maze(width=7, height=7) # Total number of mazes generated: 23093748
    #print(f"Total number of mazes generated: {len(env.pathfinding())}")
    #for i, maze in enumerate(env.evaluation_mazes):
    #    plot_maze(maze, show=False, save_path=f"maze_{i}.png")
    env = Maze(width=6, height=6)
    env.reset()
