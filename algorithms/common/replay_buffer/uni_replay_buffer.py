import torch
import random
import math
import numbers
from collections import deque
from dataclasses import dataclass

@dataclass(eq=False)
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    terminated: bool
    truncated: bool

    def __post_init__(self):
        assert isinstance(self.state, torch.Tensor)
        assert isinstance(self.action, int)
        assert isinstance(self.reward, numbers.Real)
        assert isinstance(self.next_state, torch.Tensor)
        assert isinstance(self.terminated, bool)
        assert isinstance(self.truncated, bool)

    def __eq__(self, other: object) -> bool:
        '''
        Check if two Transition objects are equal by checking each term from
        easy to complex.
        '''
        if not isinstance(other, Transition):
            return False
        if (
            self.action != other.action
            or self.reward != other.reward
            or self.terminated != other.terminated
            or self.truncated != other.truncated
        ):
            return False
        if (
            self.state.shape != other.state.shape
            or self.next_state.shape != other.next_state.shape
        ):
            return False
        state_equal = (
            torch.equal(self.state, other.state)
            if self.state.device == other.state.device
            else torch.equal(self.state.cpu(), other.state.cpu())
        )
        if not state_equal:
            return False
        next_state_equal = (
            torch.equal(self.next_state, other.next_state)
            if self.next_state.device == other.next_state.device
            else torch.equal(self.next_state.cpu(), other.next_state.cpu())
        )
        return next_state_equal

    def __hash__(self) -> int:
        '''
        The first time of calling this function will calculate the hash; the next time, it will return the hash directly.
        '''
        if not hasattr(self, "_cached_hash"):
            state_key = tuple(self.state.detach().cpu().reshape(-1).tolist())
            next_state_key = tuple(self.next_state.detach().cpu().reshape(-1).tolist())
            object.__setattr__(
                self,
                "_cached_hash",
                hash((
                    state_key,
                    self.action,
                    float(self.reward),
                    next_state_key,
                    self.terminated,
                    self.truncated,
                )),
            )
        return self._cached_hash

@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.states, torch.Tensor)
        assert isinstance(self.actions, torch.Tensor)
        assert isinstance(self.rewards, torch.Tensor)
        assert isinstance(self.next_states, torch.Tensor)
        assert isinstance(self.terminated, torch.Tensor)
        assert isinstance(self.truncated, torch.Tensor)

class UniReplayBuffer:
    def __init__(self, buffer_size, seed, input_dim):
        '''
        Args:
            input_dim: The dimension of representation of observation. It is used for safety checking.
        '''
        self.buffer_size = buffer_size
        self.pool = deque(maxlen=buffer_size)
        self.pool_set = set()
        self.input_dim = input_dim

        self.seed = seed
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self.pool)

    def __contains__(self, transition: Transition) -> bool:
        return transition in self.pool_set

    def contains(self, transition: Transition) -> bool:
        '''Check whether the transition already exists in the pool.'''
        return transition in self.pool_set

    def clear(self):
        '''
        useless function ...
        but keep for future use
        '''
        self.pool.clear()
        self.pool_set.clear()

    def push(self, transition: Transition) -> bool:
        '''
        Push a transition into the buffer.

        Before pushing, check if the transition already exists in the pool.
        If it already exists, do not push and return False.

        Returns:
            bool: True if the transition was added, False if it was skipped because
                  it already existed in the pool.
        '''
        if not isinstance(transition, Transition):
            raise TypeError(
                "ReplayBuffer.push() expects a single Transition object."
            )
        assert transition.state.shape == (1, self.input_dim)
        assert transition.next_state.shape == (1, self.input_dim)

        if transition in self.pool_set:
            return False

        if len(self.pool) == self.buffer_size:
            evicted = self.pool.popleft()
            self.pool_set.discard(evicted)

        self.pool.append(transition)
        self.pool_set.add(transition)
        return True

    def sample(self, batch_size):
        batch = self._rng.sample(self.pool, batch_size)
        return self._make_batch(batch)

    @staticmethod
    def _make_batch(batch):
        state_batch = torch.concat([ torch.as_tensor(t.state, dtype=torch.float32) for t in batch ], dim=0)
        action_batch = torch.stack([ torch.as_tensor(t.action, dtype=torch.long).reshape(1) for t in batch ])
        reward_batch = torch.stack([ torch.as_tensor(t.reward, dtype=torch.float32).reshape(1) for t in batch ])
        next_state_batch = torch.concat([ torch.as_tensor(t.next_state, dtype=torch.float32) for t in batch ], dim=0)
        terminated_batch = torch.stack([ torch.as_tensor(t.terminated, dtype=torch.bool).reshape(1) for t in batch ])
        truncated_batch = torch.stack([ torch.as_tensor(t.truncated, dtype=torch.bool).reshape(1) for t in batch ])

        return TransitionBatch(
            states=state_batch,
            actions=action_batch,
            rewards=reward_batch,
            next_states=next_state_batch,
            terminated=terminated_batch,
            truncated=truncated_batch
        )

class ContinuousUniReplayBuffer(UniReplayBuffer):
    '''
    Replay buffer for continuous observations that rejects similar transitions.

    ``UniReplayBuffer`` deduplicates by exact equality, which almost never
    happens once observations are real-valued. This buffer instead treats a
    candidate transition as a duplicate when an already stored transition has
    the same action and a sufficiently close state:

        action(candidate) == action(stored)
        and ||state(candidate) - state(stored)||_2 <= tolerance

    ``next_state``, ``reward``, ``terminated`` and ``truncated`` deliberately do
    not take part in the comparison: the goal is to stop writing the same
    (state, action) experience over and over, not to deduplicate whole outcomes.
    With ``tolerance=0`` a transition is rejected only when its state is
    numerically identical to a stored state and the action matches.

    The stored states are mirrored into a preallocated matrix, so the similarity
    test costs one matrix-vector product instead of a Python-level scan. A push
    is therefore O(len(self) * input_dim) rather than the O(1) exact-set lookup
    of ``UniReplayBuffer``; pick ``tolerance`` and ``buffer_size`` accordingly.

    Args:
        buffer_size: Maximum number of stored transitions.
        seed: Seed of the sampling RNG. Kept for interface compatibility.
        input_dim: Dimension of the flattened observation. Used for safety
            checking and to size the state index.
        tolerance: Maximum Euclidean distance between two states for them to be
            treated as the same state. Must be finite and nonnegative. It should
            be chosen relative to the scale of the observation and kept well
            above float32 rounding error.
    '''

    def __init__(self, buffer_size, seed, input_dim, tolerance=1e-3):
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, numbers.Integral):
            raise TypeError("buffer_size must be an integer.")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive.")
        if isinstance(input_dim, bool) or not isinstance(input_dim, numbers.Integral):
            raise TypeError("input_dim must be an integer.")
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real):
            raise TypeError("tolerance must be a real number.")
        if not math.isfinite(float(tolerance)) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative.")

        # UniReplayBuffer.__init__ is intentionally not called: its exact-match
        # set cannot express "close enough". The sampling bookkeeping below
        # mirrors it field by field, so sample() and _make_batch() still apply.
        self.buffer_size = int(buffer_size)
        self.pool = deque(maxlen=self.buffer_size)
        self.input_dim = int(input_dim)

        self.seed = seed
        self._rng = random.Random(self.seed)

        self.tolerance = float(tolerance)
        self._tolerance_sq = self.tolerance ** 2

        # Mirror of the states and actions held by ``pool``, used for the
        # similarity search. Rows are written in a ring: while the buffer is not
        # full the live rows are [0, self._size), and once it is full every row
        # is live, so the live rows are always the first ``self._size`` rows.
        self._states = torch.zeros(self.buffer_size, self.input_dim, dtype=torch.float32)
        self._norms_sq = torch.zeros(self.buffer_size, dtype=torch.float32)
        self._actions = torch.zeros(self.buffer_size, dtype=torch.long)
        self._size = 0
        self._write = 0

    def __contains__(self, transition: object) -> bool:
        if not isinstance(transition, Transition):
            return False
        return self.contains(transition)

    def contains(self, transition: Transition) -> bool:
        '''Check whether a similar transition already exists in the pool.'''
        self._validate(transition)
        return self._has_similar(transition)

    def clear(self):
        '''Remove every transition, retaining the sampling RNG state.'''
        self.pool.clear()
        self._size = 0
        self._write = 0

    def push(self, transition: Transition) -> bool:
        '''
        Push a transition into the buffer.

        Before pushing, check whether a similar transition already exists in the
        pool. If it does, do not push and return False.

        Returns:
            bool: True if the transition was added, False if it was skipped
                  because a similar transition already existed.
        '''
        self._validate(transition)
        if self._has_similar(transition):
            return False

        if len(self.pool) == self.buffer_size:
            self.pool.popleft()

        # Keep the indexed contents independent of later caller-side mutations.
        stored = Transition(
            transition.state.detach().clone(),
            transition.action,
            float(transition.reward),
            transition.next_state.detach().clone(),
            transition.terminated,
            transition.truncated,
        )
        self.pool.append(stored)
        self._index(stored)
        return True

    def _validate(self, transition: Transition):
        if not isinstance(transition, Transition):
            raise TypeError(
                "ContinuousUniReplayBuffer expects a single Transition object."
            )
        if (
            transition.state.shape != (1, self.input_dim)
            or transition.next_state.shape != (1, self.input_dim)
        ):
            raise ValueError(
                f"State tensors must have shape (1, {self.input_dim})."
            )

    def _index(self, transition: Transition):
        '''Write a stored transition into the state index.'''
        flat_state = transition.state.detach().reshape(-1).to(
            device=self._states.device, dtype=self._states.dtype
        )
        self._states[self._write] = flat_state
        self._norms_sq[self._write] = torch.dot(flat_state, flat_state)
        self._actions[self._write] = int(transition.action)
        self._write = (self._write + 1) % self.buffer_size
        if self._size < self.buffer_size:
            self._size += 1

    def _has_similar(self, transition: Transition) -> bool:
        '''Return True when a stored transition has the same action and a close state.'''
        size = self._size
        if size == 0:
            return False

        same_action = self._actions[:size] == int(transition.action)
        if not bool(same_action.any()):
            return False

        candidate = transition.state.detach().reshape(-1).to(
            device=self._states.device, dtype=self._states.dtype
        )
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b, evaluated as one
        # matrix-vector product so no (size, input_dim) temporary is allocated.
        inner_products = self._states[:size].mv(candidate)
        candidate_norm = torch.dot(candidate, candidate)
        squared_distance = torch.clamp(
            self._norms_sq[:size] + candidate_norm - 2.0 * inner_products,
            min=0.0,
        )
        # The identity cancels catastrophically for nearby vectors, so rows in
        # the rounding band around the threshold are rechecked exactly below.
        # The bound only ever over-includes rows, which costs a little compute
        # but never changes the answer.
        eps = torch.finfo(self._states.dtype).eps
        scale = self._norms_sq[:size] + candidate_norm + 2.0 * inner_products.abs()
        rounding_band = (self.input_dim + 8) * eps * scale

        definitely_similar = same_action & (squared_distance <= self._tolerance_sq - rounding_band)
        if bool(definitely_similar.any()):
            return True

        ambiguous = same_action & (squared_distance <= self._tolerance_sq + rounding_band)
        if not bool(ambiguous.any()):
            return False

        ambiguous_states = self._states[:size][ambiguous]
        exact_distance = torch.sum((ambiguous_states - candidate) ** 2, dim=1)
        return bool((exact_distance <= self._tolerance_sq).any())