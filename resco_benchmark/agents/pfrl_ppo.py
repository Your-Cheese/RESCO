from typing import Any, List, Mapping, Optional

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from pfrl.nn import Branched
import pfrl.initializers
from pfrl.agents import PPO
from pfrl.agents.ppo import _yield_minibatches
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.utils.recurrent import one_step_forward, get_recurrent_state_at, mask_recurrent_state_at
from pfrl.utils.batch_states import batch_states

from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.agents.maxpressure import MAXPRESSURE

writer = SummaryWriter()

def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


class IPPO(IndependentAgent):
    def __init__(self, env, config, obs_act, map_name, thread_number):
        super().__init__(env, config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            config["name"] = key
            self.agents[key] = PFRLPPOAgent(config, obs_space, act_space)


class PFRLPPOAgent(Agent):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        
        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        h = conv2d_size_out(obs_space[1])
        w = conv2d_size_out(obs_space[2])

        self.model = nn.Sequential(
            lecun_init(nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            lecun_init(nn.Linear(h*w*64, 64)),
            nn.ReLU(),
            lecun_init(nn.Linear(64, 64)),
            nn.ReLU(),
            Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(64, act_space), 1e-2),
                    SoftmaxCategoricalHead()
                ),
                lecun_init(nn.Linear(64, 1))
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        if config["task_phasing"]:
            self.agent = TaskPhasingPPO(self.model, self.optimizer, gpu=self.device.index,
                         phi=lambda x: np.asarray(x, dtype=np.float32),
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         update_interval=1024,
                         minibatch_size=256,
                         epochs=4,
                         standardize_advantages=True,
                         entropy_coef=0.001,
                         max_grad_norm=0.5, 
                         phase_step=config["phase_step"],
                         phasing_update_window=config["phasing_update_window"],
                         name=config["name"])
        else:
            self.agent = PPO(self.model, self.optimizer, gpu=self.device.index,
                         phi=lambda x: np.asarray(x, dtype=np.float32),
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         update_interval=1024,
                         minibatch_size=256,
                         epochs=4,
                         standardize_advantages=True,
                         entropy_coef=0.001,
                         max_grad_norm=0.5)

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'{path}.pt',
        )

class TaskPhasingPPO(PPO):
    
    def __init__(self,
                model,
                optimizer,
                obs_normalizer=None,
                gpu=None,
                gamma=0.99,
                lambd=0.95,
                phi=lambda x: x,
                value_func_coef=1.0,
                entropy_coef=0.01,
                update_interval=2048,
                minibatch_size=64,
                epochs=10,
                clip_eps=0.2,
                clip_eps_vf=None,
                standardize_advantages=True,
                batch_states=batch_states,
                recurrent=False,
                max_recurrent_sequence_len=None,
                act_deterministically=False,
                max_grad_norm=None,
                value_stats_window=1000,
                entropy_stats_window=1000,
                value_loss_stats_window=100,
                policy_loss_stats_window=100,
                phase_step: int = 0.015, 
                phasing_update_window: int = 50,
                name: str = ""):
        super().__init__(model, optimizer, obs_normalizer, gpu, gamma, lambd, phi, value_func_coef, entropy_coef, update_interval, minibatch_size, epochs, clip_eps, clip_eps_vf, standardize_advantages, batch_states, recurrent, max_recurrent_sequence_len, act_deterministically, max_grad_norm, value_stats_window, entropy_stats_window, value_loss_stats_window, policy_loss_stats_window)
        self.AI_used = []
        self.RL_used = 0
        self.maxpressure_act = 0
        self.policy_prob = 0.0
        self.phase_step = phase_step
        self.previous_mean_reward = -999.0
        self.phasing_update_window = phasing_update_window
        self.current_eps = 0
        self.ep_mean_rewards = []
        self.name = name
        
    
    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs
        
        exp_policy_select = np.random.rand() > self.policy_prob
        
        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_distrib, batch_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, b_state, self.train_prev_recurrent_states
                )
            else:
                action_distrib, batch_value = self.model(b_state)
            #Choose between current policy or other policy and find likelihood update
            if exp_policy_select:
                batch_action = np.array([self.maxpressure_act])
                self.AI_used.append(1)
            else:
                batch_action = action_distrib.sample().cpu().numpy()
                self.RL_used += 1
                self.AI_used.append(0)
                
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action
    
    
    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        
        assert self.training

        for i, (state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
            )
        ):
            if state is not None:
                assert action is not None
                transition = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                }
                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                self.batch_last_episode[i].append(transition)
                
            if done or reset:
                assert self.batch_last_episode[i]
                self.ep_mean_rewards.append(np.mean([t["reward"] for t in self.batch_last_episode[i]]))
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
                self.current_eps += 1
                if self.current_eps > self.phasing_update_window:
                    mean_reward = np.mean(self.ep_mean_rewards[-self.phasing_update_window:])
                    if mean_reward >= self.previous_mean_reward:
                        self.policy_prob = min(self.policy_prob + self.phase_step, 1.0)
                        self.previous_mean_reward = mean_reward
                        self.current_eps = 0
                
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None

        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )
                
        self._update_if_dataset_is_ready()

    def _update(self, dataset):
        """Update both the policy and the value function."""

        device = self.device

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        assert "state" in dataset[0]
        assert "v_teacher" in dataset[0]

        if self.standardize_advantages:
            all_advs = torch.tensor([b["adv"] for b in dataset], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)

        for batch in _yield_minibatches(
            dataset, minibatch_size=self.minibatch_size, num_epochs=self.epochs
        ):
            states = self.batch_states(
                [b["state"] for b in batch], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            actions = torch.tensor([b["action"] for b in batch], device=device)
            distribs, vs_pred = self.model(states)

            advs = torch.tensor(
                [b["adv"] for b in batch], dtype=torch.float32, device=device
            )
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = torch.tensor(
                [b["log_prob"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_pred_old = torch.tensor(
                [b["v_pred"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_teacher = torch.tensor(
                [b["v_teacher"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.model.zero_grad()
            loss = self._lossfun(
                distribs.entropy(),
                vs_pred,
                distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                log_probs_old=log_probs_old,
                advs=advs,
                vs_teacher=vs_teacher,
            )
            writer.add_scalar(f'Loss/train/{self.name}', loss, self.n_updates)
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.n_updates += 1