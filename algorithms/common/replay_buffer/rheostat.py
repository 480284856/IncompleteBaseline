import math
import random
from collections import Counter, deque

from .replay_buffer import ReplayBuffer, Transition


class Rheostat(ReplayBuffer):
    """FIFO replay with probabilistic admission based on diversity and reward.

    For a candidate transition, n is its exact-match count in the current
    buffer and d = abs(reward) - abs(mean_reward). Admission probability is
    exp(-m * n) * sigmoid(k * (d - b)). The first transition is always accepted.

    Matches include both tensors' values, shapes and dtypes, action, reward,
    terminated and truncated. Similar but unequal transitions are distinct.(
    Later I will implement that similar transitions are same transitions to be
    applied in continuous situation)
    Statistics count stored occurrences, including accepted duplicates, and
    are evaluated before eviction. Use push()/clear() to change the buffer;
    do not mutate pool or its transitions directly.
    """

    def __init__(self, buffer_size, seed, input_dim, m=0.1, k=5.0, b=-0.1):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if not all(math.isfinite(value) for value in (m, k, b)):
            raise ValueError("m, k and b must be finite.")
        if m < 0 or k <= 0:
            raise ValueError("m must be nonnegative and k must be positive.")
        super().__init__(buffer_size, seed, input_dim)
        self.m = m
        self.k = k
        self.b = b
        self._counts = Counter()
        self._keys = deque()
        self._reward_sum = 0.0
        # Sampling must not change subsequent admission decisions.
        self._push_rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pool)

    @property
    def mean_reward(self) -> float:
        """Mean signed reward of stored occurrences; zero when empty."""
        return self._reward_sum / len(self.pool) if self.pool else 0.0

    def _validate(self, transition: Transition):
        if not isinstance(transition, Transition):
            raise TypeError("Rheostat expects a single Transition object.")
        if (
            transition.state.shape != (1, self.input_dim)
            or transition.next_state.shape != (1, self.input_dim)
        ):
            raise ValueError(f"State tensors must have shape (1, {self.input_dim}).")
        if not math.isfinite(transition.reward):
            raise ValueError("Transition reward must be finite.")

    @staticmethod
    def _transition_key(transition: Transition) -> tuple:
        def tensor_key(tensor):
            return (
                tuple(tensor.shape),
                tensor.dtype,
                tuple(tensor.detach().cpu().reshape(-1).tolist()),
            )

        return (
            tensor_key(transition.state),
            transition.action,
            float(transition.reward),
            tensor_key(transition.next_state),
            transition.terminated,
            transition.truncated,
        )

    def _probability(self, reward, key) -> float:
        # always accept the first transition
        if not self.pool:
            return 1.0
        diversity = math.exp(-self.m * self._counts[key])
        difference = abs(reward) - abs(self.mean_reward)
        z = -1 * self.k * (difference - self.b)
        # Evaluate 1 / (1 + exp(z)) safely for either sign of the exponent.
        if z <= 0:
            rarity = 1.0 / (1.0 + math.exp(z))
        else:
            exp_neg_z = math.exp(-z)
            rarity = exp_neg_z / (1.0 + exp_neg_z)
        return diversity * rarity

    def probability(self, transition: Transition) -> float:
        """Return admission probability without changing statistics or RNG."""
        self._validate(transition)
        return self._probability(transition.reward, self._transition_key(transition))

    def push(self, transition: Transition) -> bool:
        """Return True if inserted; rejection leaves the FIFO unchanged."""
        self._validate(transition)
        key = self._transition_key(transition)
        probability = self._probability(transition.reward, key)
        if self._push_rng.random() >= probability:
            return False

        # Keep the indexed contents independent of later caller-side mutations.
        stored = Transition(
            transition.state.detach().clone(),
            transition.action,
            float(transition.reward),
            transition.next_state.detach().clone(),
            transition.terminated,
            transition.truncated,
        )
        if len(self.pool) == self.buffer_size:
            evicted = self.pool.popleft()
            evicted_key = self._keys.popleft()
            self._counts[evicted_key] -= 1
            if self._counts[evicted_key] == 0:
                del self._counts[evicted_key]
            self._reward_sum -= evicted.reward

        self.pool.append(stored)
        self._keys.append(key)
        self._counts[key] += 1
        self._reward_sum += stored.reward
        return True

    def clear(self):
        """Remove all transitions and statistics, retaining RNG state."""
        self.pool.clear()
        self._keys.clear()
        self._counts.clear()
        self._reward_sum = 0.0
