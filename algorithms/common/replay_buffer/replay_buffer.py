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
