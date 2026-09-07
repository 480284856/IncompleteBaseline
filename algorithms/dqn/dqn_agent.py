import torch
import copy
import random
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from ..common.exploration_rate_calculation import ClassicalExploration
from ..common.replay_buffer import ReplayBuffer, Transition, TransitionBatch
from ..common.qnetwork import QNetwork
from ..common.rollout import MABRollout

from tqdm import tqdm

class DQNAgent:
    def __init__(self,
                 input_dim,
                 output_dim,
                 seed,

                 replay_buffer_size:int,
                 training_env:gym.Env,
                 eval_env:gym.Env,
                 sample_batch_size:int,

                 epsilon_strategy:ClassicalExploration=ClassicalExploration(),
                 loss_fn=torch.nn.SmoothL1Loss(),

                 gamma:float=0.9,

                 tau:float=0.005,

                 total_time_steps:int=1_000_000,
                 learning_start:int=10_000,
                 training_freq:int=4,
                 grad_step_per_train:int=4,

                 num_eval_episodes:int=100,
                 eval_freq:int|None=10_000,
                 tensorboard_log_dir:str|None=None,
                 max_episode_steps_warmup:int|None=None):
        '''
        Args:
            input_dim: The dimension of observation.
            output_dim: The dimension of action space.
            epsilon_strategy: How you change the exploration degree with time step going.
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
            max_episode_steps_warmup: Episode step limit while time_step < learning_start.
                None keeps the environment's training limit. Requires an environment with
                train_step_limitation; switching limits preserves the current episode's elapsed steps.
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed

        self.qnetwork = QNetwork(observation_dimensions=input_dim, number_of_actions=output_dim)
        self.q_target_network = QNetwork(observation_dimensions=input_dim, number_of_actions=output_dim)
        self.q_target_network.load_state_dict(self.qnetwork.state_dict())
        self.q_target_network.requires_grad_(False)

        self._rng = random.Random(self.seed)

        self.epsilon_strategy = epsilon_strategy

        self.replaybuffer = ReplayBuffer(buffer_size=replay_buffer_size, seed=self.seed, input_dim=self.input_dim)
        self.training_env = training_env
        self.eval_env = eval_env
        self.sample_batch_size=sample_batch_size

        self.loss_fn=loss_fn
        self.optim = torch.optim.Adam(self.qnetwork.parameters())

        self.gamma = gamma

        self.tau = tau

        self.total_time_steps=total_time_steps
        self.learning_start=learning_start
        self.training_freq=training_freq
        self.grad_step_per_train=grad_step_per_train
        self.num_eval_episodes=num_eval_episodes
        self.eval_freq=eval_freq
        self.max_episode_steps_warmup=max_episode_steps_warmup

        self.rollout_warmup = MABRollout(self.training_env,)

        if self.max_episode_steps_warmup is not None:
            if (isinstance(self.max_episode_steps_warmup, bool)
                    or not isinstance(self.max_episode_steps_warmup, int)
                    or self.max_episode_steps_warmup < 1):
                raise ValueError("max_episode_steps_warmup must be a positive integer or None.")
            if not hasattr(self.training_env.unwrapped, "train_step_limitation"):
                raise ValueError("The warmup limit requires an environment with train_step_limitation.")

        if self.eval_freq is not None:
            if isinstance(self.eval_freq, bool) or not isinstance(self.eval_freq, int) or self.eval_freq < 1:
                raise ValueError("eval_freq must be a positive integer or None.")
            if self.eval_env is self.training_env:
                raise ValueError("Periodic evaluation requires a separate evaluation environment.")

        assert self.total_time_steps>=1
        assert self.learning_start>=1 and self.learning_start<self.total_time_steps
        assert self.training_freq>=1 and self.training_freq<self.total_time_steps
        assert self.grad_step_per_train>=1
        # To prevent the network from failing to update even when the buffer is full.
        assert 1 <= self.sample_batch_size <= replay_buffer_size
        assert self.num_eval_episodes>=1

        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.transition_counter=set()

        self.best_solved_rate=0
        self.best_model = None

    def train(self,):
        bar = tqdm(range(1,self.total_time_steps+1))
        training_step = 1
        solved_episode=0
        training_limit = None
        if self.max_episode_steps_warmup is not None:
            training_limit = self.training_env.unwrapped.train_step_limitation
        try:
            state, _ = self._reset_env(self.training_env, options={"is_evaluation": False})
            for time_step in bar:
                if self.max_episode_steps_warmup is not None:
                    self.training_env.unwrapped.train_step_limitation = (
                        self.max_episode_steps_warmup
                        if time_step < self.learning_start
                        else training_limit
                    )
                state_key = tuple(state.detach().cpu().reshape(-1).tolist())
                self.transition_counter.add(state_key)
                self.tensorboard_writer.add_scalar(
                    "debug/num_unique_states",
                    len(self.transition_counter),
                    time_step,
                )
                if time_step >= self.learning_start:
                    next_state, _, terminated, truncated, _ = self.rollout(current_time_step=training_step, state=state)
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
                    # random rollout
                    next_state, _, terminated, truncated, _ = self.random_rollout(state=state)
                if terminated or truncated:
                    state, _ = self._reset_env(self.training_env, options={"is_evaluation": False})
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
            if self.max_episode_steps_warmup is not None:
                self.training_env.unwrapped.train_step_limitation = training_limit
            self.tensorboard_writer.close()
            bar.close()

    def evaluation(self,env:gym.Env, evaluation:bool=True):
        returns = []
        lengths = []
        solved = 0

        state,_ = self._reset_env(env, options={"is_evaluation": evaluation})
        for episode in range(self.num_eval_episodes):
            rewards = 0.0
            length = 0
            while True:
                action = self.action_selection(state=state, pure_greedy=True)
                next_state, reward, terminated, truncated, info = env.step(action)

                rewards += float(reward)
                length += 1

                if terminated or truncated:
                    state,_ = self._reset_env(env, options={"is_evaluation": evaluation})
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
        
        return np.mean(returns), np.mean(lengths), solved/self.num_eval_episodes

    def action_selection(self, state:torch.Tensor, current_time_step:int|None=None, pure_greedy:bool=False) -> int:
        assert state.shape == (1,self.input_dim), "Current implementation is only for single environment, not for vectorized environment."

        if pure_greedy:
            with torch.no_grad():
                return self.qnetwork(state).argmax().item()
        else:
            exploration_rate = self.epsilon_strategy.step(current_time_step)
            if self._rng.random() < exploration_rate:
                return self.rollout_warmup.action_selection(state=state)
            else:
                with torch.no_grad():
                    return self.qnetwork(state).argmax().item()

    def rollout(self, current_time_step:int, state:torch.Tensor):
        assert isinstance(state, torch.Tensor)
        assert state.shape == (1,self.input_dim), f"Expect shape of (1,{self.input_dim}), got {state.shape}"
        
        action = self.action_selection(state=state, current_time_step=current_time_step)
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

        action, next_state, reward, terminated, truncated, info = self.rollout_warmup.step(state=state)

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
        

    def _td_target(self, transitions:TransitionBatch):
        if not isinstance(transitions, TransitionBatch):
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

    def update_target_network(self,):
        target_net_state_dict = self.q_target_network.state_dict()
        policy_net_state_dict = self.qnetwork.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.q_target_network.load_state_dict(target_net_state_dict)

    def final_evaluation(self) -> Tuple[float, float, float]:
        """
        Evaluate each validation maze once and log final metrics after training.
        """

        env = self.eval_env
        num_mazes = len(env.unwrapped.evaluation_mazes)
        if num_mazes == 0:
            raise RuntimeError("The evaluation maze set is empty.")

        returns, lengths = [], []
        solved = 0
        was_training = self.qnetwork.training
        self.qnetwork.eval()
        try:
            with torch.no_grad():
                for maze_index in range(num_mazes):
                    state, _ = self._reset_env(env, options={
                        "is_evaluation": True,
                        "maze_index": maze_index,
                    })
                    rewards = 0.0
                    length = 0
                    while True:
                        action = self.action_selection(state=state, pure_greedy=True)
                        state, reward, terminated, truncated, _ = env.step(action)
                        rewards += float(reward)
                        length += 1
                        if terminated or truncated:
                            solved += int(terminated)
                            break
                    returns.append(rewards)
                    lengths.append(length)
        finally:
            self.qnetwork.train(was_training)

        mean_return = float(np.mean(returns))
        mean_length = float(np.mean(lengths))
        success_rate = solved / num_mazes
        # train() closes its writer, so reopen the same run directory.
        with SummaryWriter(log_dir=self.tensorboard_writer.log_dir) as writer:
            writer.add_scalar("final_eval/return", mean_return, self.total_time_steps)
            writer.add_scalar("final_eval/steps", mean_length, self.total_time_steps)
            writer.add_scalar("final_eval/success_rate", success_rate, self.total_time_steps)
        return mean_return, mean_length, success_rate
