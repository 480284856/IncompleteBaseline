'''The idea comes from Mr. Griesbach(https://github.com/Sebastian-Griesbach)'''
import torch
import copy
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from ..common.exploration_rate_calculation import StepDecay
from ..common.replay_buffer import ReplayBuffer, Transition, TransitionBatch
from ..common.qnetwork import QNetwork
from ..dqn.dqn_agent import DQNAgent
from tqdm import tqdm

class ActorDQNAgent(DQNAgent):
    def __init__(self,
                 input_dim,
                 output_dim,
                 seed,

                 replay_buffer_size:int,
                 training_env:gym.Env,
                 eval_env:gym.Env,
                 sample_batch_size:int,

                 loss_fn=torch.nn.SmoothL1Loss(),

                 gamma:float=0.9,

                 tau:float=0.005,

                 total_time_steps:int=1_000_000,
                 learning_start:int=10_000,
                 training_freq:int=4,
                 grad_step_per_train:int=4,

                 num_eval_episodes:int=100,
                 eval_freq:int|None=10_000,
                 tensorboard_log_dir:str|None=None):
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
        '''
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,

            replay_buffer_size=replay_buffer_size,
            training_env=training_env,
            eval_env=eval_env,
            sample_batch_size=sample_batch_size,

            loss_fn=loss_fn,

            gamma=gamma,

            tau=tau,

            total_time_steps=total_time_steps,
            learning_start=learning_start,
            training_freq=training_freq,
            grad_step_per_train=grad_step_per_train,

            num_eval_episodes=num_eval_episodes,
            eval_freq=eval_freq,
            tensorboard_log_dir=tensorboard_log_dir
        )

        self.actor_dqn_network = QNetwork(
            observation_dimensions=input_dim,
            number_of_actions=output_dim,
        )
        self.loss_fn_actor = nn.CrossEntropyLoss()
        self.optim_actor = torch.optim.Adam(self.actor_dqn_network.parameters())
        
        self.training_rng = torch.Generator()
        self.eval_rng = torch.Generator()
        self.training_rng.manual_seed(seed)
        self.eval_rng.manual_seed(seed)
        self.transition_counter=set()

        self.best_solved_rate=0
        self.best_model = None

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

    def action_selection(self, state:torch.Tensor, is_training:bool=False) -> int:
        assert state.shape == (1,self.input_dim), "Current implementation is only for single environment, not for vectorized environment."

        with torch.no_grad():
            q_values = self.actor_dqn_network(state)
            action_probs = torch.softmax(q_values, dim=1)
            return int(torch.multinomial(
                action_probs, num_samples=1,
                generator=self.training_rng if is_training else self.eval_rng
            ).item())

    def rollout(self, state:torch.Tensor):
        assert isinstance(state, torch.Tensor)
        assert state.shape == (1,self.input_dim), f"Expect shape of (1,{self.input_dim}), got {state.shape}"
        
        action = self.action_selection(state=state, is_training=True)
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
    
    def train(self,):
        bar = tqdm(range(1,self.total_time_steps+1))
        training_step = 1
        solved_episode=0
        try:
            state, _ = self._reset_env(self.training_env, options={"is_evaluation": False})
            for time_step in bar:
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
            self.tensorboard_writer.close()
            bar.close()

    def final_evaluation(self) -> Tuple[float, float, float]:
        """
        [Warning] Enter function final evaluation will change actor DQN network to the best version, so, keep in mind to call it ONLY after training.

        Evaluate each validation maze once and log final metrics after training.

        Uses the actor's sampled policy, as in periodic evaluation. Returns
        mean episode return, mean episode length, and success rate.
        """
        env = self.eval_env
        if self.best_model != None:
            self.actor_dqn_network = self.best_model
            print("[Warning] Enter function final evaluation will change actor TQN network to the best version, so, keep in mind to call it ONLY after training.")


        num_mazes = len(env.unwrapped.evaluation_mazes)
        if num_mazes == 0:
            raise RuntimeError("The evaluation maze set is empty.")

        returns, lengths = [], []
        solved = 0
        was_training = self.actor_dqn_network.training
        self.actor_dqn_network.eval()
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
                        action = self.action_selection(state=state)
                        state, reward, terminated, truncated, _ = env.step(action)
                        rewards += float(reward)
                        length += 1
                        if terminated or truncated:
                            solved += int(terminated)
                            break
                    returns.append(rewards)
                    lengths.append(length)
        finally:
            self.actor_dqn_network.train(was_training)

        mean_return = float(np.mean(returns))
        mean_length = float(np.mean(lengths))
        success_rate = solved / num_mazes
        # train() closes its writer, so reopen the same run directory.
        with SummaryWriter(log_dir=self.tensorboard_writer.log_dir) as writer:
            writer.add_scalar("final_eval/return", mean_return, self.total_time_steps)
            writer.add_scalar("final_eval/steps", mean_length, self.total_time_steps)
            writer.add_scalar("final_eval/success_rate", success_rate, self.total_time_steps)
        return mean_return, mean_length, success_rate

    def evaluation(self,env:gym.Env, evaluation:bool=True):
        returns = []
        lengths = []
        solved = 0

        state,_ = self._reset_env(env, options={"is_evaluation": evaluation})
        for episode in range(self.num_eval_episodes):
            rewards = 0.0
            length = 0
            while True:
                action = self.action_selection(state=state)
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
            self.best_model = copy.deepcopy(self.actor_dqn_network)
            self.best_target_network = copy.deepcopy(self.q_target_network)
        
        return np.mean(returns), np.mean(lengths), solved/self.num_eval_episodes
