'''The idea comes from Mr. Griesbach(https://github.com/Sebastian-Griesbach)'''
import torch
import copy
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
from collections.abc import Sequence
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from ..common.exploration_rate_calculation import StepDecay
from ..common.replay_buffer.replay_buffer import ReplayBuffer, Transition, TransitionBatch
from ..common.qnetwork import QNetwork
from .actordqn_agent import ActorDQNAgent
from tqdm import tqdm

class ActorDQNAgent4CartPole(ActorDQNAgent):
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

        self.highest_return = -np.inf

    def train(self, *args, **kwargs):       
        bar = tqdm(range(1,self.total_time_steps+1))
        training_step = 1

        try:
            state, _ = self._reset_env(self.training_env, **kwargs)
            for time_step in bar:
                if time_step >= self.learning_start:
                    next_state, _, terminated, truncated, _ = self.rollout(state=state)
                    if training_step % self.training_freq == 0:
                        for _ in range(self.grad_step_per_train):
                            loss = self.update_qnetwork()
                            if loss is not None:
                                self.tensorboard_writer.add_scalar("training/loss", loss, training_step)
                        self.update_target_network()
                    if self.eval_freq is not None and training_step % self.eval_freq == 0:
                        mean_return, mean_length = self.evaluation(env=self.eval_env, evaluation=True)
                        self.tensorboard_writer.add_scalar("eval/return", mean_return, training_step)
                        self.tensorboard_writer.add_scalar("eval/steps", mean_length, training_step)
                        mean_return, mean_length = self.evaluation(env=self.eval_env, evaluation=False)
                        self.tensorboard_writer.add_scalar("training/return", mean_return, training_step)
                        self.tensorboard_writer.add_scalar("training/steps", mean_length, training_step)
                        self.tensorboard_writer.flush()
                    training_step += 1
                else:
                    next_state, _, terminated, truncated, _ = self.random_rollout(state=state) # random rollout
                if terminated or truncated:
                    state, _ = self._reset_env(self.training_env, **kwargs)                  
                    continue
                else:
                    state = next_state
            self.final_evaluation()
        # The finally block ensures the writer saves buffered logs and closes the progress bar 
        # when training finishes, raises an error, or is interrupted with Ctrl+C
        finally:
            self.tensorboard_writer.close()
            bar.close()

    def evaluation(self,env:gym.Env, evaluation:bool=True, *args, **kwargs):
        returns = []
        lengths = []

        state,_ = self._reset_env(env, **kwargs)
        for episode in range(self.num_eval_episodes):
            rewards = 0.0
            length = 0
            while True:
                action = self.get_policy(state=state)
                next_state, reward, terminated, truncated, info = env.step(action)

                rewards += float(reward)
                length += 1

                if terminated or truncated:
                    state,_ = self._reset_env(env, **kwargs)
                    break
                else:
                    state=next_state

            returns.append(rewards)
            lengths.append(length)

        if evaluation and self.highest_return < np.mean(returns):
            self.highest_return = np.mean(returns)
            self._save_best_model()
        
        return np.mean(returns), np.mean(lengths)
    
    def final_evaluation(self) -> None:
        """
        Evaluate each validation maze once and log final metrics after training.
        """
        if self.best_actor is None:
            return
        
        assert isinstance(self.best_actor, QNetwork)

        try:
            tmp = self.num_eval_episodes
            self.num_eval_episodes *= 10
            tmp_model = self.actor_dqn_network
            self.actor_dqn_network = self.best_actor

            mean_return, mean_length = self.evaluation(self.eval_env, evaluation=True)
            
            self.tensorboard_writer.add_scalar("final_eval/return", mean_return, self.total_time_steps)
            self.tensorboard_writer.add_scalar("final_eval/steps", mean_length, self.total_time_steps)
        finally:
            self.num_eval_episodes = tmp
            self.actor_dqn_network = tmp_model