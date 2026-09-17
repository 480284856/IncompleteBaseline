"""Reusable modules for generating, varying, and inspecting maze grids."""

from collections import deque
from collections.abc import Iterator, Sequence
from typing import TypeAlias

import numpy as np


EMPTY = 0
WALL = 1
PLAYER = 2
GOAL = 3

Position: TypeAlias = tuple[int, int]

CARDINAL_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

PASSABLE_VALUES = frozenset((EMPTY, PLAYER, GOAL))
WALL_OPEN_PROBABILITY = 0.5


def generate_all_single_solution_mazes(width: int, height: int) -> np.ndarray:
    """Generate every maze formed by start-to-goal path.

    Each ordered pair of distinct grid cells is used once as the player and
    goal position. All cells outside the chosen path are walls.

    Args:
        width: Number of columns in each maze.
        height: Number of rows in each maze.

    Returns:
        An array with shape ``(number_of_mazes, height, width)`` and dtype
        ``int8``.
    """
    _validate_grid_size(width, height)

    mazes = []

    # iterate over every ordered pair of distinct cells in row-major order
    # (2,3) is different from (3,2)
    for start, goal in iter_start_goal_pairs(width, height):
        # enumerate every possible path from start to goal
        for path in enumerate_single_solution_paths(width, height, start, goal):
            maze = create_maze_from_path(path, width, height)
            mazes.append(maze)

    if not mazes:
        raise ValueError("No valid mazes generated")
    return np.stack(mazes)


def generate_maze_variations(
    mazes: np.ndarray,
    expected_shape: tuple[int, int] | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Create one randomized, solution-preserving variation of each maze.

    Args:
        mazes: Source mazes with shape ``(count, height, width)``. Each source
            must contain one player, one goal, and only its original path as
            passable cells.
        expected_shape: Optional ``(height, width)`` required for each maze.
        random_state: Optional seed for reproducible random choices.

    Returns:
        An ``int8`` array with the same shape as ``mazes``.
    """
    mazes = validate_maze_batch(mazes, expected_shape)
    if len(mazes) == 0:
        raise ValueError("mazes must contain at least one maze")

    rng = np.random.default_rng(random_state)
    return np.stack(
        [create_solution_preserving_variation(maze, rng) for maze in mazes]
    )


def create_solution_preserving_variation(
    maze: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Open random walls, then remove every changed shortest route."""
    original_maze = validate_original_path_maze(maze)
    start = find_unique_marker(original_maze, PLAYER, "player")
    goal = find_unique_marker(original_maze, GOAL, "goal")

    # Safe checking
    # So far, the maze is original maze, which has only one path from player to goal.
    original_path = find_shortest_path(original_maze, start, goal)
    if original_path is None:
        raise ValueError("the player must be able to reach the goal")
    passages = {
        tuple(position)
        for position in np.argwhere(original_maze != WALL)
    }
    if passages != set(original_path):
        raise ValueError(
            "each original maze must contain only its original player-goal path"
        )

    # Randomly open walls in the maze
    variation = randomly_open_walls(
        original_maze,
        rng,
        probability=WALL_OPEN_PROBABILITY,
    )
    # Remove any changed shortest routes until the original path is the unique shortest route
    return remove_changed_shortest_routes(
        variation,
        original_path=frozenset(original_path),
        start=start,
        goal=goal,
        rng=rng,
    )


def randomly_open_walls(
    maze: np.ndarray,
    rng: np.random.Generator,
    probability: float,
) -> np.ndarray:
    """Independently replace each wall with an empty cell at ``probability``."""
    variation = maze.astype(np.int8, copy=True)
    wall_positions = np.argwhere(variation == WALL)
    cells_to_open = rng.random(len(wall_positions)) < probability

    for position in wall_positions[cells_to_open]:
        variation[tuple(position)] = EMPTY
    return variation


def remove_changed_shortest_routes(
    maze: np.ndarray,
    *,
    original_path: frozenset[Position],
    start: Position,
    goal: Position,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block added cells until the original path is the unique shortest route."""
    repaired = maze.copy()
    original_distance = len(original_path) - 1

    while True:
        distances_from_start = find_distances(repaired, start)
        distances_from_goal = find_distances(repaired, goal)
        # current shortest distance from start to goal in the repaired maze
        current_distance = distances_from_start.get(goal)

        # pure safety checks
        if current_distance is None:
            raise RuntimeError("variation repair disconnected the original path")
        if current_distance > original_distance:
            raise RuntimeError("variation repair damaged the original path")

        # Identify cells that are on a changed shortest route from start to goal
        # or another optimal path having the same length as the original path when current_distance == original_distance
        changed_route_cells = [
            position
            for position, start_distance in distances_from_start.items()
            if position not in original_path
            and position in distances_from_goal # the cell is reachable from both start and goal, which means there is another path from start to goal that goes through this cell
            and start_distance + distances_from_goal[position] == current_distance # the cell is on a shortest path from start to goal
        ]

        original_is_shortest = current_distance == original_distance
        if original_is_shortest and not changed_route_cells:
            return repaired
        if not changed_route_cells:
            raise RuntimeError("cannot repair a changed route without altering the original path")

        cell_to_block = changed_route_cells[
            int(rng.integers(len(changed_route_cells))) # generates a random index from 0 to len(...) - 1
        ]
        repaired[cell_to_block] = WALL


def find_shortest_path(
    maze: np.ndarray,
    start: Position,
    goal: Position,
) -> tuple[Position, ...] | None:
    """Return one shortest passable path, or ``None`` when none exists."""
    parents: dict[Position, Position | None] = {start: None}
    pending = deque((start,))
    height, width = maze.shape

    while pending:
        position = pending.popleft()
        if position == goal:
            return _reconstruct_path(parents, goal)

        for neighbor in cardinal_neighbors(position, width, height):
            # if the neighbor is not already visited              and is not obstacle(path or goal), add it to the queue
            if neighbor not in parents and int(maze[neighbor]) in PASSABLE_VALUES:
                parents[neighbor] = position
                pending.append(neighbor)
    return None


def find_distances(maze: np.ndarray, start: Position) -> dict[Position, int]:
    """Return distances from one endpoint without crossing the other endpoint.

    ``start`` must be either the player or goal position. The opposite endpoint
    is included in the returned distances, but it is terminal: the search does
    not continue through it to cells on its other side.
    """
    height, width = maze.shape
    start_value = int(maze[start])
    if start_value == PLAYER:
        terminal_value = GOAL
    elif start_value == GOAL:
        terminal_value = PLAYER
    else:
        raise ValueError("start must be the player or goal position")

    distances = {start: 0}
    pending = deque((start,))

    while pending:
        position = pending.popleft()
        for neighbor in cardinal_neighbors(position, width, height):
            if neighbor not in distances and int(maze[neighbor]) in PASSABLE_VALUES:
                distances[neighbor] = distances[position] + 1
                if int(maze[neighbor]) != terminal_value:
                    pending.append(neighbor)
    return distances


def validate_maze_batch(
    mazes: np.ndarray,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a maze batch after validating its shape and cell values."""
    result = np.asarray(mazes)
    if result.ndim != 3:
        raise ValueError("mazes must have shape (count, height, width)")
    if result.shape[1] <= 0 or result.shape[2] <= 0:
        raise ValueError("maze width and height must be positive")
    if expected_shape is not None and result.shape[1:] != expected_shape:
        raise ValueError(
            f"maze shape must be {expected_shape}, got {result.shape[1:]}"
        )

    invalid_values = set(np.unique(result)) - {EMPTY, WALL, PLAYER, GOAL}
    if invalid_values:
        raise ValueError(f"mazes contain unsupported values: {sorted(invalid_values)}")
    return result


def validate_original_path_maze(maze: np.ndarray) -> np.ndarray:
    """Validate the shape, values, and endpoint markers of one source maze."""
    result = np.asarray(maze)
    if result.ndim != 2:
        raise ValueError("each maze must be two-dimensional")
    invalid_values = set(np.unique(result)) - {EMPTY, WALL, PLAYER, GOAL}
    if invalid_values:
        raise ValueError(f"maze contains unsupported values: {sorted(invalid_values)}")
    # find the position of the player and goal markers, 
    # which must each occur exactly once, so if they don't, raise an error
    find_unique_marker(result, PLAYER, "player")
    find_unique_marker(result, GOAL, "goal")
    return result


def find_unique_marker(maze: np.ndarray, value: int, name: str) -> Position:
    """Return the position of a marker that must occur exactly once."""
    positions = np.argwhere(maze == value)
    if len(positions) != 1:
        raise ValueError(f"maze must contain exactly one {name} cell")
    return tuple(map(int, positions[0]))


def _reconstruct_path(
    parents: dict[Position, Position | None],
    goal: Position,
) -> tuple[Position, ...]:
    """Reconstruct a start-to-goal path from breadth-first-search parents."""
    path = []
    position: Position | None = goal
    while position is not None:
        path.append(position)
        position = parents[position]
    path.reverse()
    return tuple(path)


def iter_start_goal_pairs(width: int, height: int) -> Iterator[tuple[Position, Position]]:
    """Yield every ordered pair of distinct cells in row-major order."""
    # generate all grid cells in row-major order at once,
    # tuple() will consume the iterator at once
    # the final result looks like this:
    # ((0, 0), (0, 1), (1, 0), (1, 1))
    cells = tuple(iter_grid_cells(width, height))
    for start in cells:
        for goal in cells:
            if start != goal:
                yield start, goal


def enumerate_single_solution_paths(
    width: int,
    height: int,
    start: Position,
    goal: Position,
) -> Iterator[tuple[Position, ...]]:
    """Yield paths whose open cells form exactly one route to ``goal``.
    For more details, please refer to the following link:
        https://ljp0vuj4fr1r.jp.larksuite.com/wiki/QyQJwfSRBiuvXZk8kXpjTxKmpEe#share-TgFvdrewmotuVGx1SBxjqWLKpCb
        
    When listing neighboring cells:

    *  we can't go to a cell that is already on the path or to any neighbors of a cell on the path,
       except for the neighbors of the last cell on the path;
    * when the endpoint touches the goal, the path must enter the goal.
    """
    _validate_grid_size(width, height)
    _validate_position(start, width, height, "start")
    _validate_position(goal, width, height, "goal")
    if start == goal:
        raise ValueError("start and goal must be different cells")

    # generate all paths from start to goal recursively
    yield from _extend_single_solution_path(
        width=width,
        height=height,
        goal=goal,
        path=(start,),
        forbidden=frozenset((start,)),
    )


def create_maze_from_path(
    path: Sequence[Position],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert one start-to-goal path into a wall-filled maze array."""
    if len(path) < 2:
        raise ValueError("path must contain distinct start and goal cells")

    maze = np.full((height, width), WALL, dtype=np.int8)
    for position in path:
        _validate_position(position, width, height, "path cell")
        maze[position] = EMPTY

    maze[path[0]] = PLAYER
    maze[path[-1]] = GOAL
    return maze


def iter_grid_cells(width: int, height: int) -> Iterator[Position]:
    """Yield all cells in a rectangular grid in row-major order."""
    for row in range(height):
        for column in range(width):
            yield row, column


def cardinal_neighbors(
    position: Position,
    width: int,
    height: int,
) -> tuple[Position, ...]:
    """Return the in-bounds cells directly above, below, left, and right."""
    row, column = position
    # based on current position, we get the four neighboring cells:
    # up, down, left, right, and filter out the ones that are out of bounds
    return tuple(
        (row + row_change, column + column_change)
        for row_change, column_change in CARDINAL_DIRECTIONS
        if 0 <= row + row_change < height
        and 0 <= column + column_change < width
    )


def _extend_single_solution_path(
    *,
    width: int,
    height: int,
    goal: Position,
    path: tuple[Position, ...],
    forbidden: frozenset[Position],
) -> Iterator[tuple[Position, ...]]:
    """Recursively extend one path while preventing branches and loops.
    
    Args:
        goal: The cell that the path must reach.
        path: current path from start to the last cell in the path.
        forbidden: cells that the path must not enter, 
                   which are all cells in the path plus all neighbors of those cells except for the last cell.
    """
    endpoint_neighbors = cardinal_neighbors(path[-1], width, height)

    # The goal-adjacency rule(rule 2) makes entering the goal the only legal move.
    if goal in endpoint_neighbors:
        yield path + (goal,)
        # add a return statement here to end the function after
        # come back to the function
        return

    for next_cell in endpoint_neighbors:
        # the first condition checks if the next cell is the goal, which is already handled above
        # so I think the first condition is redundant and can be removed
        if next_cell == goal or next_cell in forbidden:
            continue

        # On the following step, the new endpoint must not return to or touch
        # any part of the path that came before it and the neighbors of the cells
        # on the path except for the last cell.

        # next_forbidden = forbidden U set(endpoint_neighbors) U {next_cell}

        # Here I think (next_cell,) is redundant because next_cell is already in endpoint_neighbors, 
        # so we can just use forbidden.union(endpoint_neighbors)
        next_forbidden = forbidden.union(endpoint_neighbors, (next_cell,))
        yield from _extend_single_solution_path(
            width=width,
            height=height,
            goal=goal,
            path=path + (next_cell,),
            forbidden=next_forbidden,
        )


def _validate_grid_size(width: int, height: int) -> None:
    """Reject dimensions that cannot describe a rectangular grid."""
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)):
        raise TypeError("width must be an integer")
    if isinstance(height, bool) or not isinstance(height, (int, np.integer)):
        raise TypeError("height must be an integer")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")


def _validate_position(
    position: Position,
    width: int,
    height: int,
    name: str,
) -> None:
    """Reject a malformed or out-of-bounds grid coordinate."""
    if len(position) != 2:
        raise ValueError(f"{name} must contain a row and column")
    row, column = position
    if not (0 <= row < height and 0 <= column < width):
        raise ValueError(f"{name} {position} is outside the maze")
