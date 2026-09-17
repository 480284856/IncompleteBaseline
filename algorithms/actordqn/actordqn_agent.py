'''The idea comes from Mr. Griesbach(https://github.com/Sebastian-Griesbach)'''
import torch
import copy
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
from pathlib import Path
from collections.abc import Sequence
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from ..common.exploration_rate_calculation import StepDecay
from ..common.replay_buffer.replay_buffer import ReplayBuffer, Transition, TransitionBatch
from ..common.qnetwork import QNetwork
from ..dqn.dqn_agent import DQNAgent
from tqdm import tqdm

class ActorDQNAgent(DQNAgent):
    def __init__(self,
                 seed,
                 input_dim,
                 output_dim,
                 hidden_sizes:Sequence[int]|None=None,

                 sample_batch_size:int=64,
                 replay_buffer_size:int=500_000,
                 gamma:float=0.9,
                 tau:float=0.005,

                 training_env:gym.Env=None,
                 eval_env:gym.Env=None,
                 loss_fn=torch.nn.SmoothL1Loss(),
                 total_time_steps:int=1_000_000,
                 learning_start:int=10_000,
                 training_freq:int=4,
                 grad_step_per_train:int=4,
                 num_eval_episodes:int=100,
                 eval_freq:int|None=10_000,
                 tensorboard_log_dir:str|None=None,
                 *args, **kwargs):
        '''
        Args:
            input_dim: The dimension of observation.
            output_dim: The dimension of action space.
            hidden_sizes: Width of each Q-network hidden layer. None keeps QNetwork's default
                architecture, so existing runs are unaffected. At most one replay admission
                mode may be enabled.

            sample_batch_size: The size of data sampled at once from the replay buffer.
            gamma: Discount factor for future return.
            tau: The proportion to integrate the weight of the online network to the talking network.
            
            total_time_steps: Total times to call env.step.
            learning_start: Before starting to update the target Q-network, the steps we take to sample data from the environment.
            training_freq: How often do we update the model?
            grad_step_per_train: How many times do we perform supervised learning during each training stage?
            num_eval_episodes: Number of episodes to run per evaluation.
            eval_freq: Evaluate every N training environment steps, including before learning starts. None disables periodic evaluation.
            tensorboard_log_dir: TensorBoard output directory. None creates an automatic run directory under runs/.
        '''
        super().__init__(
            seed,
            input_dim,
            output_dim,
            hidden_sizes,
            sample_batch_size,
            replay_buffer_size,
            gamma,
            tau,
            training_env,
            eval_env,
            loss_fn,
            total_time_steps,
            learning_start,
            training_freq,
            grad_step_per_train,
            num_eval_episodes,
            eval_freq,
            tensorboard_log_dir,
            *args, **kwargs
        )

        self._setup_actor_model(input_dim=input_dim, output_dim=output_dim)
        self._setup_actor_training(seed=seed)

    def update_qnetwork(self,):
        if len(self.replaybuffer.pool) >= self.sample_batch_size:
            batch = self.replaybuffer.sample(batch_size=self.sample_batch_size)

            td_target = self._td_target(batch)

            prediction = self.qnetwork(batch.states)
            estimation = prediction.gather(1, batch.actions)

            loss = self.loss_fn(td_target, estimation)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # update for actor qnetwork
            label = prediction.argmax(dim=1)
            logits = self.actor_dqn_network(batch.states)
            loss_actor = self.loss_fn_actor(logits, label)
            self.optim_actor.zero_grad()
            loss_actor.backward()
            self.optim_actor.step()

            return loss_actor.item()
        return None

    def get_policy(self, state:torch.Tensor, is_training:bool=False, *args, **kwargs) -> int:
        assert state.shape == (1,self.input_dim), "Current implementation is only for single environment, not for vectorized environment."

        with torch.no_grad():
            q_values = self.actor_dqn_network(state)
            action_probs = torch.softmax(q_values, dim=1)
            return int(torch.multinomial(
                action_probs, num_samples=1,
                generator=self.training_rng if is_training else self.eval_rng
            ).item())

    def _setup_actor_model(self, input_dim, output_dim):
        self.actor_dqn_network = QNetwork(
            observation_dimensions=input_dim,
            number_of_actions=output_dim,
        )

    def _setup_actor_training(self, seed):
        self.loss_fn_actor = nn.CrossEntropyLoss()
        self.optim_actor = torch.optim.Adam(self.actor_dqn_network.parameters())
        
        self.training_rng = torch.Generator()
        self.eval_rng = torch.Generator()
        self.training_rng.manual_seed(seed)
        self.eval_rng.manual_seed(seed)
        self.transition_counter=set()

        self.best_solved_rate=-1
        self.best_actor = None
        self.best_model = None
        self.best_target_network = None

    def _save_best_model(self):

        self.best_model = copy.deepcopy(self.qnetwork)
        self.best_target_network = copy.deepcopy(self.q_target_network)
        self.best_actor = copy.deepcopy(self.actor_dqn_network)

        model_dir = Path(self.tensorboard_writer.log_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_model.state_dict(), model_dir / "q_network.pt")
        torch.save(self.best_target_network.state_dict(), model_dir / "q_target_network.pt")
        torch.save(self.best_actor.state_dict(), model_dir / "actor_network.pt")