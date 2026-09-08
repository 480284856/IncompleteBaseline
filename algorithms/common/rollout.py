from .replay_buffer import Transition
import torch
import gymnasium as gym
import math
import random

class MABRollout:
    def __init__(self, env: gym.Env, c:float=2, lr:float=1e-1, seed:int=42):
        self.env = env
        self.c = c
        self.lr = lr
        self.preferences = {}
        self._rng = random.Random(seed)

    def action_selection(self, state:torch.Tensor):
        key = tuple(state.detach().cpu().reshape(-1).tolist())
        if key not in self.preferences:
            self.preferences[key] = [
                torch.tensor([0 for _ in range(self.env.action_space.n)], dtype=torch.float32),   # Q(a)
                torch.tensor([0 for _ in range(self.env.action_space.n)], dtype=torch.int32)      # N(a)
            ]

        q,n = self.preferences[key]
        untried = torch.where(n == 0)[0].tolist()
        if untried:
            action = self._rng.choice(untried)
        else:
            t = n.sum().item()
            p = q + self.c * torch.sqrt(math.log(t) / n)
            # select action based on that preference using MAB algorithm
            action = torch.argmax(p).item()
        return action
    
    def step(self, state:torch.Tensor):
        '''
        if it's a new state:
            1. build a list of action preference for that state
            2. store that state in self.buffer
        else:
            1. select action based on that preference using MAB algorithm
            2. update preference based on feedback
        '''
        key = tuple(state.detach().cpu().reshape(-1).tolist())
        action = self.action_selection(state=state)

        next_state, reward, terminated, truncated, info = self.env.step(action)

        # update preference based on feedback
        key_nxts = tuple(next_state.detach().cpu().reshape(-1).tolist())
        if key_nxts in self.preferences:
            r = -1 + reward # take real reward as a weak signal if the reward function could indicate the dangerous action.
        else:
            r = 1 + reward

        q,n = self.preferences[key]
        n[action] += 1
        q[action] += self.lr * (r - q[action])
        return action, next_state, reward, terminated, truncated, info

        
