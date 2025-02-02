import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from resco_benchmark.agents.agent import SharedAgent
from resco_benchmark.agents.pfrl_dqn import DQNAgent
from resco_benchmark.config.signal_config import signal_configs
from pfrl.q_functions import DiscreteActionValueHead


class MPLight(SharedAgent):
    def __init__(self, env, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        self.env = env
        self.writer = SummaryWriter()
        config["writer"] = self.writer
        phase_pairs = np.array(signal_configs[map_name]['phase_pairs'])
        num_actions = len(phase_pairs)

        comp_mask = []
        for i in range(len(phase_pairs)):
            zeros = np.zeros(len(phase_pairs) - 1, dtype=np.int)
            cnt = 0
            for j in range(len(phase_pairs)):
                if i == j: continue
                pair_a = phase_pairs[i]
                pair_b = phase_pairs[j]
                if len(set(pair_a + pair_b)) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = np.asarray(comp_mask)
        print(comp_mask)

        comp_mask = torch.from_numpy(comp_mask).to(self.device)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        model = FRAP(config, num_actions, phase_pairs, comp_mask, self.device)
        self.agent = DQNAgent(config, num_actions, model, num_agents=config['num_lights'])

    def act(self, observation):
        if self.reverse_valid is None and self.valid_acts is not None:
            self.reverse_valid = {}
            for signal_id in self.valid_acts:
                self.reverse_valid[signal_id] = dict(zip(self.valid_acts[signal_id].values(), self.valid_acts[signal_id].keys()))

        batch_obs = list(observation.values())
        if self.valid_acts is None:
            batch_valid = None
            batch_reverse = None
        else:
            batch_valid = [self.valid_acts.get(agent_id) for agent_id in
                           observation.keys()]
            batch_reverse = [self.reverse_valid.get(agent_id) for agent_id in
                          observation.keys()]
            
        if self.config["task_phasing"]:
            self.agent.agent.maxpressure_act = self.env.maxpressure_act

        batch_acts = self.agent.act(batch_obs,
                                valid_acts=batch_valid,
                                reverse_valid=batch_reverse)
        return {
            agent_id: batch_acts[i]
            for i, agent_id in enumerate(observation.keys())
        }


class FRAP(nn.Module):
    def __init__(self, config, output_shape, phase_pairs, competition_mask, device):
        super(FRAP, self).__init__()
        self.oshape = output_shape
        self.phase_pairs = torch.from_numpy(phase_pairs).to(device)
        self.comp_mask = competition_mask
        self.device = device
        self.demand_shape = config['demand_shape']      # Allows more than just queue to be used

        self.d_out = 4      # units in demand input layer
        self.p_out = 4      # size of phase embedding
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)

        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)

        self.lane_conv = nn.Conv2d(2*self.lane_embed_units, 20, kernel_size=(1, 1))

        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))

        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

        self.head = DiscreteActionValueHead()

    def forward(self, states):
        states = states.to(self.device)
        num_movements = int((states.size()[1]-1)/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, 0].to(torch.int64)
        states = states[:, 1:]
        states = states.float()

        # Expand action index to mark demand input indices
        act_pairs = self.phase_pairs[acts]
        extended_acts = torch.zeros(batch_size, num_movements, dtype=torch.int64, device=self.device).scatter_(1, act_pairs, 1)
        phase_embeds = torch.sigmoid(self.p(extended_acts))
            
        phase_demands = []
        for i in range(num_movements):
            phase = phase_embeds[:, i]  # size 4
            demand = states[:, i:i+self.demand_shape]
            demand = torch.sigmoid(self.d(demand))    # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)

        pairs = torch.transpose(phase_demands[:, self.phase_pairs[:,0]] + phase_demands[:, self.phase_pairs[:,1]], 1, 0)
        
        # Create permutations of phase pairs
        phase_idx = torch.arange(0, self.oshape, device=self.device)
        phases = torch.dstack(torch.meshgrid(phase_idx, phase_idx, indexing='ij'))
        mask = torch.ones_like(torch.arange(0, self.oshape*self.oshape)).scatter_(0, torch.arange(0, self.oshape*self.oshape, self.oshape+1), 0)
        phases = phases.view(self.oshape*self.oshape, 2)[mask.bool()]
        pair_permutations = pairs[phases]
        
        rotated_phases = pair_permutations.view(batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units)
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.tile((batch_size, 1, 1))
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)  # Move channels up
        relations = F.relu(self.relation_conv(relations))  # Pair demand representation

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # Pairwise competition result

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))
        q_values = torch.sum(combine_features, dim=-1)
        return self.head(q_values)
