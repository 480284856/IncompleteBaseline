"""Visualization helpers for maze grids."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

try:
    from .util import EMPTY, GOAL, PLAYER, WALL
except ImportError:  # Support importing this module when env.py is run directly.
    from util import EMPTY, GOAL, PLAYER, WALL


MAZE_COLORS = {
    EMPTY: "#E5DFF7",   # Purple passage/path
    WALL: "#000000",    # Black wall
    PLAYER: "#DDF2E5",  # Light-green agent
    GOAL: "#58A06A",    # Green goal
}


def plot_maze(maze, ax=None, *, show=True, save_path=None):
    """Plot a two-dimensional maze array.

    Expected values are 0 for a passage/path, 1 for a wall, 2 for the agent,
    and 3 for the goal. The function returns ``(figure, axes)`` so callers can
    further customize or test the plot. Figures created only for file export
    are closed after saving to keep large batches from exhausting graphics
    memory.

    Args:
        maze: Two-dimensional array-like maze grid.
        ax: Optional Matplotlib axes on which to draw.
        show: Whether to display the figure with ``plt.show()``.
        save_path: Optional path at which to save the rendered figure.
    """
    maze = np.asarray(maze)
    if maze.ndim != 2:
        raise ValueError("maze must be a two-dimensional array")
    if maze.size == 0:
        raise ValueError("maze must not be empty")

    unexpected_values = set(np.unique(maze)) - set(MAZE_COLORS)
    if unexpected_values:
        raise ValueError(
            f"maze contains unsupported values: {sorted(unexpected_values)}"
        )

    owns_figure = ax is None
    if owns_figure:
        height, width = maze.shape
        scale = 0.75
        figure, ax = plt.subplots(
            figsize=(max(width * scale, 2), max(height * scale, 2))
        )
    else:
        figure = ax.figure

    color_map = ListedColormap(
        [MAZE_COLORS[value] for value in (EMPTY, WALL, PLAYER, GOAL)]
    )
    ax.imshow(
        maze,
        cmap=color_map,
        vmin=EMPTY - 0.5,
        vmax=GOAL + 0.5,
        interpolation="none",
        aspect="equal",
    )

    height, width = maze.shape
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    if owns_figure:
        figure.tight_layout(pad=0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
    elif save_path is not None and owns_figure:
        plt.close(figure)

    return figure, ax
