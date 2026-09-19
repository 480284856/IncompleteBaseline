import torch
import random
import numbers
import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
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

class ReplayBuffer:
    def __init__(self, buffer_size, seed, input_dim):
        '''
        Args:
            input_dim: The dimension of representation of observation. It is used for safety checking.
        '''
        self.buffer_size = buffer_size
        self.pool = deque(maxlen=buffer_size)
        self.input_dim = input_dim

        self.seed = seed
        self._rng = random.Random(self.seed)

    def __len__(self):
        return len(self.pool)

    def push(self, transition: Transition):
        if not isinstance(transition, Transition):
            raise TypeError(
                "ReplayBuffer.push() expects a single Transition object."
            )
        assert transition.state.shape == (1, self.input_dim)
        assert transition.next_state.shape == (1, self.input_dim)
        self.pool.append(transition)

    def sample(self, batch_size):
        batch = self._rng.sample(self.pool, batch_size)

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


class FastReplayBuffer:
    """Uniform replay backed by preallocated numpy arrays.

    Sampling is O(batch_size): indices are drawn from range(len(self))
    (O(1) indexing instead of O(n) deque traversal) and the whole batch is
    gathered with one vectorized numpy fancy-index, then wrapped as torch
    tensors without a copy. sample keeps the original no-replacement
    behaviour; switch to self._rng.choices(range(n), k=batch_size) for
    sampling with replacement (SB3 behaviour).
    """

    def __init__(self, buffer_size, seed, input_dim):
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive.")
        self.buffer_size = buffer_size
        self.input_dim = input_dim
        self._rng = random.Random(seed)

        self._states = np.empty((buffer_size, input_dim), dtype=np.float32)
        self._next_states = np.empty((buffer_size, input_dim), dtype=np.float32)
        self._actions = np.empty((buffer_size, 1), dtype=np.int64)
        self._rewards = np.empty((buffer_size, 1), dtype=np.float32)
        self._terminated = np.empty((buffer_size, 1), dtype=np.bool_)
        self._truncated = np.empty((buffer_size, 1), dtype=np.bool_)
        self._pos = 0
        self._full = False

    def __len__(self):
        return self.buffer_size if self._full else self._pos

    def push(self, transition: Transition):
        if not isinstance(transition, Transition):
            raise TypeError(
                "FastReplayBuffer.push() expects a single Transition object."
            )
        assert transition.state.shape == (1, self.input_dim)
        assert transition.next_state.shape == (1, self.input_dim)

        i = self._pos
        self._states[i] = transition.state[0].detach().cpu().numpy()
        self._next_states[i] = transition.next_state[0].detach().cpu().numpy()
        self._actions[i, 0] = transition.action
        self._rewards[i, 0] = transition.reward
        self._terminated[i, 0] = transition.terminated
        self._truncated[i, 0] = transition.truncated

        self._pos += 1
        if self._pos == self.buffer_size:
            self._pos = 0
            self._full = True

    def sample(self, batch_size):
        n = len(self)
        if batch_size > n:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {n}."
            )
        logical = np.array(self._rng.sample(range(n), batch_size), dtype=np.int64)
        # Once the ring buffer has wrapped, its physical order is rotated by
        # self._pos relative to FIFO order. Map positions back so a sampled
        # batch matches the plain ReplayBuffer deque exactly (reproducibility).
        indices = (logical + self._pos) % self.buffer_size if self._full else logical
        return TransitionBatch(
            states=torch.from_numpy(self._states[indices]),
            actions=torch.from_numpy(self._actions[indices]),
            rewards=torch.from_numpy(self._rewards[indices]),
            next_states=torch.from_numpy(self._next_states[indices]),
            terminated=torch.from_numpy(self._terminated[indices]),
            truncated=torch.from_numpy(self._truncated[indices]),
        )

