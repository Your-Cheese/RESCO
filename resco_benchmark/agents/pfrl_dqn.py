from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from logging import Logger, getLogger
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

import pfrl
from pfrl import explorers, replay_buffers
from pfrl.explorer import Explorer
from pfrl.agents import DQN
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.utils.contexts import evaluating
from pfrl.utils.batch_states import batch_states
from pfrl.utils.recurrent import recurrent_state_as_numpy, get_recurrent_state_at
from pfrl.agents.dqn import _batch_reset_recurrent_states_when_episodes_end
from pfrl.replay_buffer import batch_experiences
from pfrl.replay_buffers import PrioritizedReplayBuffer


from resco_benchmark.agents.agent import IndependentAgent, Agent

class IDQN(IndependentAgent):
    def __init__(self, env, config, obs_act, map_name, thread_number):
        super().__init__(env, config, obs_act, map_name, thread_number)
        
        self.writer = SummaryWriter()
        config["writer"] = self.writer
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            def conv2d_size_out(size, kernel_size=2, stride=1):
                return (size - (kernel_size - 1) - 1) // stride + 1

            h = conv2d_size_out(obs_space[1])
            w = conv2d_size_out(obs_space[2])

            model = nn.Sequential(
                nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(h * w * 64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, act_space),
                DiscreteActionValueHead()
            )
            config['name'] = key
            self.agents[key] = DQNAgent(config, act_space, model)


class DQNAgent(Agent):
    def __init__(self, config, act_space, model, num_agents=0):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())
        replay_buffer = replay_buffers.ReplayBuffer(10000)

        if num_agents > 0:
            explorer = SharedEpsGreedy(
                config['EPS_START'],
                config['EPS_END'],
                num_agents*config['steps'],
                lambda: np.random.randint(act_space),
            )
        else:
            explorer = explorers.LinearDecayEpsilonGreedy(
                config['EPS_START'],
                config['EPS_END'],
                config['steps'],
                lambda: np.random.randint(act_space),
            )

        if num_agents > 0:
            print('USING SHAREDDQN')
            if config['task_phasing']:
                self.agent = SharedTaskPhasingDQN(self.model, self.optimizer, replay_buffer,
                                                    config['GAMMA'], explorer, gpu=self.device.index,
                                                    minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                                                    phi=lambda x: np.asarray(x, dtype=np.float32),
                                                    target_update_interval=config['TARGET_UPDATE']*num_agents, update_interval=num_agents,
                                                    phase_step=config["phase_step"],
                                                    phasing_update_window=config["phasing_update_window"],
                                                    writer=config["writer"])
            else:
                self.agent = SharedDQN(self.model, self.optimizer, replay_buffer,
                                        config['GAMMA'], explorer, gpu=self.device.index,
                                        minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                                        phi=lambda x: np.asarray(x, dtype=np.float32),
                                        target_update_interval=config['TARGET_UPDATE']*num_agents, update_interval=num_agents)
        else:
            if config['task_phasing']:
                self.agent = TaskPhasingDQN(self.model, self.optimizer, replay_buffer, config['GAMMA'], explorer, 
                                            gpu=self.device.index,
                                            minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                                            phi=lambda x: np.asarray(x, dtype=np.float32),
                                            target_update_interval=config['TARGET_UPDATE'],
                                            phase_step=config["phase_step"],
                                            phasing_update_window=config["phasing_update_window"],
                                            name=config["name"],
                                            writer=config["writer"])
            else:
                self.agent = DQN(self.model, self.optimizer, replay_buffer, config['GAMMA'], explorer,
                                gpu=self.device.index,
                                minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                                phi=lambda x: np.asarray(x, dtype=np.float32),
                                target_update_interval=config['TARGET_UPDATE'])

    def act(self, observation, valid_acts=None, reverse_valid=None):
        if isinstance(self.agent, SharedDQN):
            return self.agent.act(observation, valid_acts=valid_acts, reverse_valid=reverse_valid)
        else:
            return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        if isinstance(self.agent, SharedDQN):
            self.agent.observe(observation, reward, done, info)
        else:
            self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'{path}.pt',
        )


class TaskPhasingDQN(DQN):
    def __init__(
                self,
                q_function: torch.nn.Module,
                optimizer: torch.optim.Optimizer,  # type: ignore  # somehow mypy complains
                replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer,
                gamma: float,
                explorer: Explorer,
                gpu: Optional[int] = None,
                replay_start_size: int = 50000,
                minibatch_size: int = 32,
                update_interval: int = 1,
                target_update_interval: int = 10000,
                clip_delta: bool = True,
                phi: Callable[[Any], Any] = lambda x: x,
                target_update_method: str = "hard",
                soft_update_tau: float = 1e-2,
                n_times_update: int = 1,
                batch_accumulator: str = "mean",
                episodic_update_len: Optional[int] = None,
                logger: Logger = getLogger(__name__),
                batch_states: Callable[
                    [Sequence[Any], torch.device, Callable[[Any], Any]], Any
                ] = batch_states,
                recurrent: bool = False,
                max_grad_norm: Optional[float] = None,
                phase_step: int = 0.015, 
                phasing_update_window: int = 50,
                name: str = "",
                writer: SummaryWriter = None,):
        super().__init__(q_function, optimizer, replay_buffer, gamma, explorer, gpu, replay_start_size, minibatch_size, update_interval, target_update_interval, clip_delta, phi, target_update_method, soft_update_tau, n_times_update, batch_accumulator, episodic_update_len, logger, batch_states, recurrent, max_grad_norm)
        self.AI_used = []
        self.RL_used = 0
        self.maxpressure_act = 0
        self.policy_prob = 0.0
        self.phase_step = phase_step
        self.previous_mean_reward = -999.0
        self.phasing_update_window = phasing_update_window
        self.current_eps = 0
        self.total_eps = 0
        self.cur_ep_rewards = []
        self.ep_mean_rewards = []
        self.name = name
        self.writer = writer
        
    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_argmax = batch_av.greedy_actions.detach().cpu().numpy()
        if self.training:
            exp_policy_select = np.random.rand() > self.policy_prob
            batch_action = []
            for i in range(len(batch_obs)):
                if exp_policy_select:
                    batch_action.append(self.maxpressure_act)
                    self.AI_used.append(1)
                else:
                    batch_action.append(self.explorer.select_action(
                        self.t,
                        lambda: batch_argmax[i],
                        action_value=batch_av[i : i + 1],
                    ))
                    self.RL_used += 1
                    self.AI_used.append(0)
                    # breakpoint()
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax
        return batch_action
    
    def _batch_observe_train(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }
                if self.recurrent:
                    transition["recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states, i, detach=True
                        )
                    )
                    transition["next_recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states, i, detach=True
                        )
                    )
                self.replay_buffer.append(env_id=i, **transition)
                self.cur_ep_rewards.append(batch_reward[i])
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)

                    self.ep_mean_rewards.append(np.mean(self.cur_ep_rewards))
                    self.cur_ep_rewards = []
                    self.current_eps += 1
                    self.total_eps += 1
                    if self.current_eps > self.phasing_update_window:
                        mean_reward = np.mean(self.ep_mean_rewards[-self.phasing_update_window:])
                        self.writer.add_scalar(f'Mean_Reward/train/{self.name}', mean_reward, self.total_eps)
                        if mean_reward >= self.previous_mean_reward:
                            # breakpoint()
                            self.policy_prob = min(self.policy_prob + self.phase_step, 1.0)
                            self.previous_mean_reward = mean_reward
                            self.current_eps = 0

            self.replay_updater.update_if_necessary(self.t)

        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states = None
            self.train_recurrent_states = (
                _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.train_recurrent_states,
                )
            )

    # def update(
    #     self, experiences: List[List[Dict[str, Any]]], errors_out: Optional[list] = None
    # ) -> None:
    #     """Update the model from experiences

    #     Args:
    #         experiences (list): List of lists of dicts.
    #             For DQN, each dict must contains:
    #               - state (object): State
    #               - action (object): Action
    #               - reward (float): Reward
    #               - is_state_terminal (bool): True iff next state is terminal
    #               - next_state (object): Next state
    #               - weight (float, optional): Weight coefficient. It can be
    #                 used for importance sampling.
    #         errors_out (list or None): If set to a list, then TD-errors
    #             computed from the given experiences are appended to the list.

    #     Returns:
    #         None
    #     """
    #     has_weight = "weight" in experiences[0][0]
    #     exp_batch = batch_experiences(
    #         experiences,
    #         device=self.device,
    #         phi=self.phi,
    #         gamma=self.gamma,
    #         batch_states=self.batch_states,
    #     )
    #     if has_weight:
    #         exp_batch["weights"] = torch.tensor(
    #             [elem[0]["weight"] for elem in experiences],
    #             device=self.device,
    #             dtype=torch.float32,
    #         )
    #         if errors_out is None:
    #             errors_out = []
    #     loss = self._compute_loss(exp_batch, errors_out=errors_out)
    #     if has_weight:
    #         assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
    #         self.replay_buffer.update_errors(errors_out)

    #     self.loss_record.append(float(loss.detach().cpu().numpy()))

    #     self.optimizer.zero_grad()
        
    #     # writer.add_scalar(f'Loss/train/{self.name}', loss, self.optim_t)
    #     loss.backward()
    #     if self.max_grad_norm is not None:
    #         pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
    #     self.optimizer.step()
    #     self.optim_t += 1

class SharedDQN(DQN):
    def __init__(self, q_function: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer, gamma: float, explorer: Explorer,
                 gpu, minibatch_size, replay_start_size, phi, target_update_interval, update_interval):

        super().__init__(q_function, optimizer, replay_buffer, gamma, explorer,
                         gpu=gpu, minibatch_size=minibatch_size, replay_start_size=replay_start_size, phi=phi,
                         target_update_interval=target_update_interval, update_interval=update_interval)

    def act(self, obs: Any, valid_acts=None, reverse_valid=None) -> Any:
        return self.batch_act(obs, valid_acts=valid_acts, reverse_valid=reverse_valid)

    def observe(self, obs: Sequence[Any], reward: Sequence[float], done: Sequence[bool], reset: Sequence[bool]) -> None:
        self.batch_observe(obs, reward, done, reset)

    def batch_act(self, batch_obs: Sequence[Any], valid_acts=None, reverse_valid=None) -> Sequence[Any]:
        if valid_acts is None: return super(SharedDQN, self).batch_act(batch_obs)
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)

            batch_qvals = batch_av.params[0].detach().cpu().numpy()
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in valid_acts[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None or batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        if self.training:
            batch_action = []
            for i in range(len(batch_obs)):
                av = batch_av[i : i + 1]
                greed = batch_argmax[i]
                act, greedy = self.explorer.select_action(self.t, lambda: greed, action_value=av, num_acts=len(valid_acts[i]))
                if not greedy:
                    act = reverse_valid[i][act]
                batch_action.append(act)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax

        return [valid_acts[i][batch_action[i]] for i in range(len(batch_action))]


def select_action_epsilon_greedily(epsilon, random_action_func, greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(), False
    else:
        return greedy_action_func(), True


class SharedEpsGreedy(explorers.LinearDecayEpsilonGreedy):

    def select_action(self, t, greedy_action_func, action_value=None, num_acts=None):
        self.epsilon = self.compute_epsilon(t)
        if num_acts is None:
            fn = self.random_action_func
        else:
            fn = lambda: np.random.randint(num_acts)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, fn, greedy_action_func
        )
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        return a if num_acts is None else (a, greedy)
    
class SharedTaskPhasingDQN(SharedDQN):
    def __init__(self, q_function: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer, gamma: float, explorer: Explorer,
                 gpu, minibatch_size, replay_start_size, phi, target_update_interval, update_interval, 
                 phase_step, phasing_update_window, writer):
        super().__init__(q_function, optimizer, replay_buffer, gamma, explorer, gpu, minibatch_size, replay_start_size, phi, target_update_interval, update_interval)
        self.AI_used = []
        self.RL_used = 0
        self.maxpressure_act = {}
        self.policy_prob = 0.0
        self.phase_step = phase_step
        self.previous_mean_reward = -999.0
        self.phasing_update_window = phasing_update_window
        self.current_eps = 0
        self.total_eps = 0
        self.cur_ep_rewards = []
        self.ep_mean_rewards = []
        self.writer = writer

    def batch_act(self, batch_obs: Sequence[Any], valid_acts=None, reverse_valid=None) -> Sequence[Any]:
        if valid_acts is None: return super(SharedDQN, self).batch_act(batch_obs)
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)

            batch_qvals = batch_av.params[0].detach().cpu().numpy()
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in valid_acts[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None or batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        if self.training:
            exp_policy_select = np.random.rand() > self.policy_prob
            if exp_policy_select:
                batch_action = list(self.maxpressure_act.values())
            else:
                batch_action = []
                for i in range(len(batch_obs)):
                    av = batch_av[i : i + 1]
                    greed = batch_argmax[i]
                    act, greedy = self.explorer.select_action(self.t, lambda: greed, action_value=av, num_acts=len(valid_acts[i]))
                    if not greedy:
                        act = reverse_valid[i][act]
                    batch_action.append(act)

            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
            if exp_policy_select:
                return batch_action
        else:
            batch_action = batch_argmax
        
        return [valid_acts[i][batch_action[i]] for i in range(len(batch_action))]
    
    def _batch_observe_train(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }
                if self.recurrent:
                    transition["recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states, i, detach=True
                        )
                    )
                    transition["next_recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states, i, detach=True
                        )
                    )
                self.replay_buffer.append(env_id=i, **transition)
                self.cur_ep_rewards.append(batch_reward[i])
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)

                    self.ep_mean_rewards.append(np.mean(self.cur_ep_rewards))
                    self.cur_ep_rewards = []
                    self.current_eps += 1
                    self.total_eps += 1
                    if self.current_eps > self.phasing_update_window:
                        mean_reward = np.mean(self.ep_mean_rewards[-self.phasing_update_window:])
                        self.writer.add_scalar(f'Mean_Reward/train', mean_reward, self.total_eps)
                        if mean_reward >= self.previous_mean_reward:
                            # breakpoint()
                            self.policy_prob = min(self.policy_prob + self.phase_step, 1.0)
                            self.previous_mean_reward = mean_reward
                            self.current_eps = 0

            self.replay_updater.update_if_necessary(self.t)

        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states = None
            self.train_recurrent_states = (
                _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.train_recurrent_states,
                )
            )
            