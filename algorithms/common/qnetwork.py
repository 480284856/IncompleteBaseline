from collections.abc import Sequence

from torch import nn
import torch.nn.functional as F

# The architecture every maze experiment in this repository was run with.
DEFAULT_HIDDEN_SIZES = (64, 32)


class QNetwork(nn.Module):
    """A LayerNorm + ReLU multilayer perceptron over observations.

    Args:
        observation_dimensions: Size of the flattened observation.
        number_of_actions: Number of discrete actions; one Q-value is emitted per
            action.
        hidden_sizes: Width of each hidden layer, in order. ``None`` keeps
            ``DEFAULT_HIDDEN_SIZES`` so existing runs are unaffected. An empty
            sequence gives a purely linear map from observations to Q-values.
            Smaller widths are useful on tasks a full-size network solves
            outright, where a saturated success rate hides differences between
            methods.
    """

    def __init__(self, observation_dimensions, number_of_actions, hidden_sizes: Sequence[int] | None = None):
        super(QNetwork, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = DEFAULT_HIDDEN_SIZES
        sizes = tuple(hidden_sizes)
        for width in sizes:
            if isinstance(width, bool) or not isinstance(width, int):
                raise TypeError(f"hidden layer widths must be integers, got {width!r}")
            if width < 1:
                raise ValueError(f"hidden layer widths must be positive, got {width}")

        layers = []
        previous_width = observation_dimensions
        for width in sizes:
            layers.append(nn.Linear(previous_width, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            previous_width = width
        layers.append(nn.Linear(previous_width, number_of_actions))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
