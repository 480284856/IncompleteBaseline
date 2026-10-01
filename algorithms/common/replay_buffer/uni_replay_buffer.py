import torch
import random
import math
import numbers
import numpy as np
from collections.abc import Sequence
from .replay_buffer import FastReplayBuffer, Transition, TransitionBatch

class UniReplayBuffer(FastReplayBuffer):
    """Fast replay buffer that stores each exact transition at most once."""

    def __init__(self, buffer_size, seed, input_dim, debug=False, tensorboard_writer=None,
                 debug_log_interval=1000):
        super().__init__(
            buffer_size=buffer_size,
            seed=seed,
            input_dim=input_dim,
            debug=debug,
            tensorboard_writer=tensorboard_writer,
            debug_log_interval=debug_log_interval,
        )
        self._keys = set()
        self._slot_keys = [None] * buffer_size

    def __contains__(self, transition: object) -> bool:
        if not isinstance(transition, Transition):
            return False
        return self.contains(transition)

    def contains(self, transition: Transition) -> bool:
        """Return whether the normalized transition is currently stored."""
        return self._make_key(transition) in self._keys

    def push(self, transition: Transition, time_step: int | None = None) -> bool:
        """Add a transition unless the same normalized transition is present."""
        self._validate(transition)
        key = self._make_key(transition)
        if key in self._keys:
            return False

        slot = self._pos
        evicted_key = self._slot_keys[slot] if self._full else None
        super().push(transition, time_step=time_step)

        if evicted_key is not None:
            self._keys.remove(evicted_key)
        self._slot_keys[slot] = key
        self._keys.add(key)
        return True

    def record_debug_stats(self, time_step):
        """Return duplicate statistics for transitions currently in the buffer."""
        if not self.debug:
            raise RuntimeError("Debug statistics require FastReplayBuffer(debug=True).")

        num_unique_transitions = len(self._transition_counts)

        self.tensorboard_writer.add_scalar("debug/uni-replaybuffer/num_uni_trans", num_unique_transitions, time_step)

    def _validate(self, transition: Transition):
        if not isinstance(transition, Transition):
            raise TypeError(
                "UniReplayBuffer.push() expects a single Transition object."
            )
        assert transition.state.shape == (1, self.input_dim)
        assert transition.next_state.shape == (1, self.input_dim)

    @staticmethod
    def _make_key(transition: Transition):
        """Match FastReplayBuffer's stored dtypes before testing identity."""
        state = transition.state[0].detach().cpu().numpy().astype(np.float32, copy=False)
        next_state = transition.next_state[0].detach().cpu().numpy().astype(np.float32, copy=False)
        reward = np.asarray(transition.reward, dtype=np.float32)
        return (
            state.tobytes(),
            int(transition.action),
            reward.tobytes(),
            next_state.tobytes(),
            transition.terminated,
            transition.truncated,
        )

    @staticmethod
    def _make_batch(batch):
        # The per-transition torch.as_tensor calls dominate the cost of
        # sampling small batches, so gather each field once and cast the
        # stacked result. This is bit-identical to casting every element
        # individually (torch.cat([]) and torch.concat([]) raise alike).
        state_batch = torch.cat([t.state for t in batch], dim=0).to(torch.float32)
        action_batch = torch.tensor([t.action for t in batch], dtype=torch.long).reshape(-1, 1)
        reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32).reshape(-1, 1)
        next_state_batch = torch.cat([t.next_state for t in batch], dim=0).to(torch.float32)
        terminated_batch = torch.tensor([t.terminated for t in batch], dtype=torch.bool).reshape(-1, 1)
        truncated_batch = torch.tensor([t.truncated for t in batch], dtype=torch.bool).reshape(-1, 1)

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

    The stored states are shared with the parent through tensor views, so the similarity
    test costs two matrix-vector products instead of a Python-level scan. The
    cheap exact comparisons (action, reward) run first and the distance tests
    only on the surviving rows, so a push usually costs far less than a full
    scan; the worst case is O(len(self) * input_dim). Pick ``tolerance``,
    ``reward_tolerance`` and ``buffer_size`` accordingly.

    Args:
        buffer_size: Maximum number of stored transitions.
        seed: Seed of the sampling RNG.
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

        super().__init__(int(buffer_size), seed, int(input_dim))

        self.tolerance = float(tolerance)
        self._tolerance_sq = self.tolerance ** 2
        self.reward_tolerance = float(reward_tolerance)

        # Tensor views share the parent's ring storage without copying it.
        self._states_view = torch.from_numpy(self._states)
        self._next_states_view = torch.from_numpy(self._next_states)
        self._actions_view = torch.from_numpy(self._actions[:, 0])
        self._rewards_view = torch.from_numpy(self._rewards[:, 0])
        self._norms_sq = torch.zeros(self.buffer_size, dtype=torch.float32)
        self._next_norms_sq = torch.zeros(self.buffer_size, dtype=torch.float32)

    def __contains__(self, transition: object) -> bool:
        if not isinstance(transition, Transition):
            return False
        return self.contains(transition)

    def contains(self, transition: Transition) -> bool:
        '''Check whether a similar transition already exists in the pool.'''
        self._validate(transition)
        return self._has_similar(transition)

    def push(self, transition: Transition, time_step: int | None = None) -> bool:
        """Reject similar transitions, otherwise insert through the parent."""
        self._validate(transition)
        if self._has_similar(transition):
            return False

        slot = self._pos
        inserted = super().push(transition, time_step=time_step)
        if inserted:
            self._update_norms(slot)
        return inserted

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

    def _update_norms(self, slot):
        """Cache norms of the float32 values actually stored by the parent."""
        state = self._states_view[slot]
        next_state = self._next_states_view[slot]
        self._norms_sq[slot] = torch.dot(state, state)
        self._next_norms_sq[slot] = torch.dot(next_state, next_state)

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
        size = len(self)
        if size == 0:
            return False

        # Check for the same action.
        # If there are no matches, we can skip the rest of the checks
        # because it implies that it's a brand new transition.
        same_action = self._actions_view[:size] == int(transition.action)
        if not bool(same_action.any()):
            return False

        close_reward = (
            self._rewards_view[:size] - float(transition.reward)
        ).abs() <= self.reward_tolerance
        # candidates: the indices of the transitions that are similar to the current transition 
        # on action and reward (x, a, r, x, x, x)
        candidates = (same_action & close_reward).nonzero(as_tuple=True)[0]
        # If there isn't any match, which means it's a new (x, a, r, x, x, x)
        # or the trasition is branch new on pair (a,r)
        if candidates.numel() == 0:
            return False

        candidate_state = transition.state.detach().reshape(-1).to(
            device=self._states_view.device, dtype=self._states_view.dtype
        )
        # self._states_view[candidates]: we only need to check the states
        # that are similar to the current transition on action and reward,
        # those transitions that are not similar to the current transition on action and reward
        # are finally determined as not similar to the current transition
        # even if they are similar to the current transition on state
        state_close = self._close_rows(
            self._states_view[candidates], self._norms_sq[candidates], candidate_state
        )
        # If it's a brand new transition because of the s in (s,a,r,s',x,x) is brand new.
        # finalists: the index of transitions that are similar to the current transition on (s,a,r,x,x,x)
        finalists = candidates[state_close]
        if finalists.numel() == 0:
            return False

        # If the new transition is similar with at least one transition on (s,a,r,x,x,x),
        # we need to check the next state.
        candidate_next_state = transition.next_state.detach().reshape(-1).to(
            device=self._next_states_view.device, dtype=self._next_states_view.dtype
        )
        next_state_close = self._close_rows(
            self._next_states_view[finalists], self._next_norms_sq[finalists], candidate_next_state
        )
        # If there is any true in next_state_close, 
        # which means there is at least one transition that is similar to the current transition
        # on (s,a,r,s',x,x) and we don't care about the last two elements.
        return bool(next_state_close.any())