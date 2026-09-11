import torch
import random
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