import torch
import copy
import random
import numpy as np
import gymnasium as gym
from collections.abc import Sequence
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from ..common.exploration_rate_calculation import ClassicalExploration
from ..common.replay_buffer.replay_buffer import ReplayBuffer, Transition, TransitionBatch
from ..common.replay_buffer.rheostat import Rheostat
from ..common.replay_buffer.uni_replay_buffer import (
    UniReplayBuffer,
    ContinuousUniReplayBuffer,
    Transition as UniTransition,
    TransitionBatch as UniTransitionBatch,
)
from ..common.qnetwork import QNetwork
from ..common.rollout import MABRollout
from tqdm import tqdm

class DQNAgent:
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

        self._validate_init_args(
            total_time_steps=total_time_steps,
            learning_start=learning_start,
            training_freq=training_freq,
            grad_step_per_train=grad_step_per_train,
            sample_batch_size=sample_batch_size,
            replay_buffer_size=replay_buffer_size,
            num_eval_episodes=num_eval_episodes,
            training_env=training_env,
            eval_freq=eval_freq,
            eval_env=eval_env,
        )

        self.seed = seed

        self._setup_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
        )
        self._setup_training(
            training_env=training_env,
            eval_env=eval_env,
            sample_batch_size=sample_batch_size,
            loss_fn=loss_fn,
            total_time_steps=total_time_steps,
            learning_start=learning_start,
            training_freq=training_freq,
            grad_step_per_train=grad_step_per_train,
            num_eval_episodes=num_eval_episodes,
            eval_freq=eval_freq,
            tensorboard_log_dir=tensorboard_log_dir,
        )
        self._setup_dqn_parameters(gamma=gamma, tau=tau)
        self._setup_replay_buffer(
            replay_buffer_size=replay_buffer_size,)
        self._setup_rollout_strategies()

    def train(self, *args, **kwargs):
        debug_mode = kwargs.get('debug', False)
        
        bar = tqdm(range(1,self.total_time_steps+1))
        training_step = 1
        if debug_mode:
            solved_episode=0

        try:
            state, _ = self._reset_env(self.training_env, **kwargs)
            for time_step in bar:
                if debug_mode:
                    state_key = tuple(state.detach().cpu().reshape(-1).tolist())
                    self.transition_counter.add(state_key)
                    self.tensorboard_writer.add_scalar(
                        "debug/num_unique_states",
                        len(self.transition_counter),
                        time_step,
                    )

                if time_step >= self.learning_start:
                    next_state, _, terminated, truncated, _ = self.rollout(state=state)
                    if training_step % self.training_freq == 0:
                        for _ in range(self.grad_step_per_train):
                            loss = self.update_qnetwork()
                            if loss is not None:
                                self.tensorboard_writer.add_scalar("training/loss", loss, training_step)
                        self.update_target_network()
                    if self.eval_freq is not None and training_step % self.eval_freq == 0:
                        mean_return, mean_length, success_rate = self.evaluation(env=self.eval_env, evaluation=True)
                        self.tensorboard_writer.add_scalar("eval/return", mean_return, training_step)
                        self.tensorboard_writer.add_scalar("eval/steps", mean_length, training_step)
                        self.tensorboard_writer.add_scalar("eval/success_rate", success_rate, training_step)
                        mean_return, mean_length, success_rate = self.evaluation(env=self.eval_env, evaluation=False)
                        self.tensorboard_writer.add_scalar("training/return", mean_return, training_step)
                        self.tensorboard_writer.add_scalar("training/steps", mean_length, training_step)
                        self.tensorboard_writer.add_scalar("training/success_rate", success_rate, training_step)
                        self.tensorboard_writer.flush()
                    training_step += 1
                else:
                    next_state, _, terminated, truncated, _ = self.random_rollout(state=state) # random rollout
                if terminated or truncated:
                    state, _ = self._reset_env(self.training_env, **kwargs)
                    if debug_mode:
                        if time_step >= self.learning_start:
                            if terminated:
                                solved_episode += 1
                                self.tensorboard_writer.add_scalar("debug/solved_episode", solved_episode, training_step)                    
                    continue
                else:
                    state = next_state
        # The finally block ensures the writer saves buffered logs and closes the progress bar 
        # when training finishes, raises an error, or is interrupted with Ctrl+C
        finally:
            self.tensorboard_writer.close()
            bar.close()

    def evaluation(self,env:gym.Env, evaluation:bool=True, *args, **kwargs):
        returns = []
        lengths = []
        solved = 0

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
                    if terminated:
                        solved += 1
                    break
                else:
                    state=next_state

            returns.append(rewards)
            lengths.append(length)

        if evaluation and self.best_solved_rate < (solved/self.num_eval_episodes):
            self.best_solved_rate = solved/self.num_eval_episodes
            self.best_model = copy.deepcopy(self.qnetwork)
            self.best_target_network = copy.deepcopy(self.q_target_network)
        
        return np.mean(returns), np.mean(lengths), solved/self.num_eval_episodes

    def get_policy(self, state:torch.Tensor, *args, **kwargs) -> int:
        with torch.no_grad():
                return self.qnetwork(state).argmax().item()
        
    def rollout(self, state:torch.Tensor, *args, **kwargs):
        assert isinstance(state, torch.Tensor)
        assert state.shape == (1,self.input_dim), f"Expect shape of (1,{self.input_dim}), got {state.shape}"
        assert state.shape == (1,self.input_dim), "Current implementation is only for single environment, not for vectorized environment."
        
        if self._rng.random() < 0.1:
            action = self._rng.choice(range(0, self.output_dim))
        else:
            with torch.no_grad():
                action = self.qnetwork(state).argmax().item()
        next_state, reward, terminated, truncated, info = self.training_env.step(action)

        t = Transition(
            state,
            action,
            float(reward),
            next_state,
            terminated,
            truncated
        )
        self.replaybuffer.push(t)

        return next_state, reward, terminated, truncated, info

    def random_rollout(self, state:torch.Tensor):
        assert isinstance(state, torch.Tensor)
        assert state.shape == (1,self.input_dim), f"Expect shape of (1,{self.input_dim}), got {state.shape}"

        action = self._rng.choice(range(0, self.output_dim))
        next_state, reward, terminated, truncated, info = self.training_env.step(action)

        t = Transition(
            state,
            action,
            float(reward),
            next_state,
            terminated,
            truncated
        )
        self.replaybuffer.push(t)

        return next_state, reward, terminated, truncated, info
    
    def update_qnetwork(self,) :
        if len(self.replaybuffer.pool) >= self.sample_batch_size:
            batch = self.replaybuffer.sample(batch_size=self.sample_batch_size)

            td_target = self._td_target(batch)

            estimation = self.qnetwork(batch.states).gather(1, batch.actions)

            loss = self.loss_fn(td_target, estimation)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            return loss.item()
        return None

    def update_target_network(self,):
        target_net_state_dict = self.q_target_network.state_dict()
        policy_net_state_dict = self.qnetwork.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.q_target_network.load_state_dict(target_net_state_dict)

    def _td_target(self, transitions:TransitionBatch|UniTransitionBatch):
        if not isinstance(transitions, (TransitionBatch, UniTransitionBatch)):
            raise TypeError()
        assert transitions.states.shape == (self.sample_batch_size, self.input_dim) , "the shape of state should be [bs, self.input_dim]"
        assert transitions.actions.shape == (self.sample_batch_size, 1) , "the shape of actions should be [bs, 1]"
        assert transitions.rewards.shape == (self.sample_batch_size, 1) , "the shape of reward should be [bs, 1]"
        assert transitions.next_states.shape == (self.sample_batch_size, self.input_dim) , "the shape of actions should be [bs, self.input_dim]"
        assert transitions.terminated.shape == (self.sample_batch_size, 1) , "the shape of 'terminated' should be [bs, 1]"
        assert transitions.truncated.shape == (self.sample_batch_size, 1) , "the shape of 'truncated' should be [bs, 1]"

        with torch.no_grad():
            esitmation:torch.Tensor = self.q_target_network(transitions.next_states)
            Q_spa = esitmation.max(dim=1, keepdim=True).values
            td_target = transitions.rewards + self.gamma * (~transitions.terminated) * Q_spa
            return td_target

    def _reset_env(self, env, **kwargs):
        return env.reset(**kwargs)

    def _validate_init_args(self, total_time_steps, learning_start, training_freq,
                            grad_step_per_train, sample_batch_size, replay_buffer_size,
                            num_eval_episodes, training_env,
                            eval_freq, eval_env):
        assert total_time_steps>=1
        assert learning_start>=1 and learning_start<total_time_steps
        assert training_freq>=1 and training_freq<total_time_steps
        assert grad_step_per_train>=1
        # To prevent the network from failing to update even when the buffer is full.
        assert 1 <= sample_batch_size <= replay_buffer_size
        assert num_eval_episodes>=1

        if eval_freq is not None:
            if isinstance(eval_freq, bool) or not isinstance(eval_freq, int) or eval_freq < 1:
                raise ValueError("eval_freq must be a positive integer or None.")
            if eval_env is training_env:
                raise ValueError("Periodic evaluation requires a separate evaluation environment.")

    def _setup_model(self, input_dim, output_dim, hidden_sizes):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden_sizes = None if hidden_sizes is None else tuple(hidden_sizes)
        self.qnetwork = QNetwork(observation_dimensions=input_dim, number_of_actions=output_dim,
                                 hidden_sizes=self.hidden_sizes)
        self.q_target_network = QNetwork(observation_dimensions=input_dim, number_of_actions=output_dim,
                                         hidden_sizes=self.hidden_sizes)
        self.q_target_network.load_state_dict(self.qnetwork.state_dict())
        self.q_target_network.requires_grad_(False)

    def _setup_training(self, 
                        training_env, 
                        eval_env, 
                        sample_batch_size, 
                        
                        loss_fn, 
                        total_time_steps, 
                        learning_start, 
                        training_freq,
                        
                        grad_step_per_train, 
                        num_eval_episodes, 
                        eval_freq,
                        
                        tensorboard_log_dir):
        self.training_env = training_env
        self.eval_env = eval_env
        self.sample_batch_size=sample_batch_size

        self.loss_fn=loss_fn
        self.optim = torch.optim.Adam(self.qnetwork.parameters())

        self._rng = random.Random(self.seed)

        self.total_time_steps=total_time_steps
        self.learning_start=learning_start
        self.training_freq=training_freq
        self.grad_step_per_train=grad_step_per_train
        self.num_eval_episodes=num_eval_episodes
        self.eval_freq=eval_freq

        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.transition_counter=set()

        self.best_solved_rate=0
        self.best_model = None

    def _setup_dqn_parameters(self, gamma, tau):
        self.gamma = gamma
        self.tau = tau

    def _setup_replay_buffer(self, replay_buffer_size, *args, **kwargs):

        self.replaybuffer = ReplayBuffer(buffer_size=replay_buffer_size, seed=self.seed, input_dim=self.input_dim)

    def _setup_rollout_strategies(self, *args, **kwargs):
            self.eps_exp_strategy=None
