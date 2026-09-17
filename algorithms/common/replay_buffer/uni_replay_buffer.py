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
    candidate transition as a duplicate when an already stored transition
    matches it on state, action, reward and next state:

        action(candidate) == action(stored)
        and ||state(candidate) - state(stored)||_2 <= tolerance
        and ||next_state(candidate) - next_state(stored)||_2 <= tolerance
        and |reward(candidate) - reward(stored)| <= reward_tolerance

    ``terminated`` and ``truncated`` take no part: for a numerical reward the
    outcome is already carried by ``reward`` and ``next_state``.

    The stored states are mirrored into preallocated matrices, so the similarity
    test costs two matrix-vector products instead of a Python-level scan. The
    cheap exact comparisons (action, reward) run first and the distance tests
    only on the surviving rows, so a push usually costs far less than a full
    scan; the worst case is O(len(self) * input_dim). Pick ``tolerance``,
    ``reward_tolerance`` and ``buffer_size`` accordingly.

    Args:
        buffer_size: Maximum number of stored transitions.
        seed: Seed of the sampling RNG. Kept for interface compatibility.
        input_dim: Dimension of the flattened observation. Used for safety
            checking and to size the state index.
        tolerance: Maximum Euclidean distance between two states, and between
            two next states, for them to count as the same. Must be finite and
            nonnegative. It should be chosen relative to the scale of the
            observation and kept well above float32 rounding error.
        reward_tolerance: Maximum absolute difference between two rewards for
            them to count as the same. Must be finite and nonnegative, and is
            expressed in the environment's reward units.
    '''

    def __init__(self, buffer_size, seed, input_dim, tolerance=1e-3, reward_tolerance=1e-2):
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, numbers.Integral):
            raise TypeError("buffer_size must be an integer.")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive.")
        if isinstance(input_dim, bool) or not isinstance(input_dim, numbers.Integral):
            raise TypeError("input_dim must be an integer.")
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        for name, value in (("tolerance", tolerance), ("reward_tolerance", reward_tolerance)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

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
        self.reward_tolerance = float(reward_tolerance)

        # Mirror of the transitions held by ``pool``, used for the similarity
        # search. Rows are written in a ring: while the buffer is not full the
        # live rows are [0, self._size), and once it is full every row is live,
        # so the live rows are always the first ``self._size`` rows.
        self._states = torch.zeros(self.buffer_size, self.input_dim, dtype=torch.float32)
        self._norms_sq = torch.zeros(self.buffer_size, dtype=torch.float32)
        self._next_states = torch.zeros(self.buffer_size, self.input_dim, dtype=torch.float32)
        self._next_norms_sq = torch.zeros(self.buffer_size, dtype=torch.float32)
        self._rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
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
        '''
        Write a stored transition into the state, next-state and reward indices.
        
        Pre-calculate the norm of the state and the next state, 
        so that we can reuse them all the time in the later training 
        without calculating the norm every time.
        '''
        write = self._write

        flat_state = transition.state.detach().reshape(-1).to(
            device=self._states.device, dtype=self._states.dtype
        )
        self._states[write] = flat_state
        self._norms_sq[write] = torch.dot(flat_state, flat_state)

        flat_next_state = transition.next_state.detach().reshape(-1).to(
            device=self._next_states.device, dtype=self._next_states.dtype
        )
        self._next_states[write] = flat_next_state
        self._next_norms_sq[write] = torch.dot(flat_next_state, flat_next_state)

        self._rewards[write] = float(transition.reward)
        self._actions[write] = int(transition.action)

        self._write = (write + 1) % self.buffer_size
        if self._size < self.buffer_size:
            self._size += 1

    def _close_rows(self, rows, norms_sq, candidate) -> torch.Tensor:
        '''Boolean mask of the ``rows`` lying within ``tolerance`` of ``candidate``.'''
        if rows.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=rows.device)

        candidate = candidate.reshape(-1)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b, evaluated as one
        # matrix-vector product so no (rows, input_dim) temporary is allocated.
        inner_products = rows.mv(candidate)
        candidate_norm = torch.dot(candidate, candidate)
        squared_distance = torch.clamp(
            norms_sq + candidate_norm - 2.0 * inner_products, min=0.0
        )
        # The identity cancels catastrophically for nearby vectors, so rows in
        # the rounding band around the threshold are rechecked exactly. The band
        # only ever over-includes rows, which costs a little compute but never
        # changes the answer.
        eps = torch.finfo(rows.dtype).eps
        # The scale is the maximum of ||a - b||^2
        scale = norms_sq + candidate_norm + 2.0 * inner_products.abs()
        # eps * scale: It is the upper band of the rounding band
        # of a^2+b^2-2ab when a and b are scalers.
        # because a^2+b^2-2ab <= a^2+b^2+2ab and the gap between two large float numbers 
        # is larger than the gap between two small float numbers,
        # which means, the quantization error, which is the distance of the mathematical result of (a - b)^2
        # mapped into the closest representable float number by computer,
        # is smaller than the number multiplying the exponent part of this floating-point number a^2+b^2+2ab
        # by epsilon, and this number is even smaller than eps * scale.
        
        # (self.input_dim + 8): And we know the calculation of the norm of a vector 
        # contains multiple operations described above.
        # So the worst-case scenario is when those quantization errors are in the same direction. 
        # For example, if those mathematical results are all on the right side of the representable 
        # float member, when they do a summation, their quantization errors also do a summation.

        # The number 8 is just an empirical number and 
        # it can be larger or could be smaller theoretically.
        rounding_band = (self.input_dim + 8) * eps * scale

        # If the worst case happened(squared_distance + rounding_band) 
        # and it doesn't go beyond the tolerance, 
        # it can be safely considered as similar.
        close = squared_distance + rounding_band <= self._tolerance_sq 
        # (~close): for those rows which `squared_distance + rounding_band` go beyond tolerance,
        #  close: T T T F F F
        # ~close: F F F T T T
        # The second condition: If the distance is so larger than tolerance 
        # that it can't touch tolerance threshold, even it is subjected by the running band.
        # undecided: F F F T T F
        undecided = (~close) & (squared_distance - rounding_band <= self._tolerance_sq)
        if bool(undecided.any()):
            exact_distance = torch.sum((rows[undecided] - candidate) ** 2, dim=1)
            close[undecided] = exact_distance <= self._tolerance_sq
        return close

    def _has_similar(self, transition: Transition) -> bool:
        '''Return True when a stored transition matches on action, reward, state and next state.'''
        size = self._size
        if size == 0:
            return False

        # Check for the same action.
        # If there are no matches, we can skip the rest of the checks
        # because it implies that it's a brand new transition.
        same_action = self._actions[:size] == int(transition.action)
        if not bool(same_action.any()):
            return False

        close_reward = (
            self._rewards[:size] - float(transition.reward)
        ).abs() <= self.reward_tolerance
        # candidates: the indices of the transitions that are similar to the current transition 
        # on action and reward (x, a, r, x, x, x)
        candidates = (same_action & close_reward).nonzero(as_tuple=True)[0]
        # If there isn't any match, which means it's a new (x, a, r, x, x, x)
        # or the trasition is branch new on pair (a,r)
        if candidates.numel() == 0:
            return False

        candidate_state = transition.state.detach().reshape(-1).to(
            device=self._states.device, dtype=self._states.dtype
        )
        # self._states[candidates]: we only need to check the states
        # that are similar to the current transition on action and reward,
        # those transitions that are not similar to the current transition on action and reward
        # are finally determined as not similar to the current transition
        # even if they are similar to the current transition on state
        state_close = self._close_rows(
            self._states[candidates], self._norms_sq[candidates], candidate_state
        )
        # If it's a brand new transition because of the s in (s,a,r,s',x,x) is brand new.
        # finalists: the index of transitions that are similar to the current transition on (s,a,r,x,x,x)
        finalists = candidates[state_close]
        if finalists.numel() == 0:
            return False

        # If the new transition is similar with at least one transition on (s,a,r,x,x,x),
        # we need to check the next state.
        candidate_next_state = transition.next_state.detach().reshape(-1).to(
            device=self._next_states.device, dtype=self._next_states.dtype
        )
        next_state_close = self._close_rows(
            self._next_states[finalists], self._next_norms_sq[finalists], candidate_next_state
        )
        # If there is any true in next_state_close, 
        # which means there is at least one transition that is similar to the current transition
        # on (s,a,r,s',x,x) and we don't care about the last two elements.
        return bool(next_state_close.any())