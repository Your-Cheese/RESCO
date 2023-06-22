import numpy as np

from resco_benchmark.config.signal_config import signal_configs
from resco_benchmark.agents.agent import Agent

try:
    import tensorflow as tf
    from resco_benchmark.agents.ma2c import MA2CAgent
except ImportError:
    tf = None

if tf is None:
    class FMA2C(Agent):
        def __init__(self, config, obs_act, map_name, thread_number):
            super().__init__()
            raise EnvironmentError("Install optional tensorflow requirement for FMA2C")

else:

    class FMA2C(Agent):
        def __init__(self, config, obs_act, map_name, thread_number):
            super().__init__()
            self.config = config

            tf.reset_default_graph()
            cfg_proto = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=cfg_proto)

            self.signal_config = signal_configs[map_name]
            self.supervisors = config['mdp']['supervisors']  # reverse of management
            self.management_neighbors = config['mdp']['management_neighbors']
            management = config['mdp']['management']

            self.state = None
            self.acts = None

            self.managers = {}
            self.workers = {}

            for manager in management:
                worker_ids = management[manager]
                mgr_act_size = self.config['management_acts']
                mgr_fingerprint_size = len(self.management_neighbors[manager]) * mgr_act_size
                self.managers[manager] = MA2CAgent(config, obs_act[manager][0], mgr_act_size, mgr_fingerprint_size, 0,
                                                   manager + str(thread_number), self.sess)

                for worker_id in worker_ids:
                    # Get fingerprint size
                    downstream = self.signal_config[worker_id]['downstream']
                    neighbors = [downstream[direction] for direction in downstream]
                    fp_size = sum(
                        obs_act[neighbor][1]  # neighbor's action size
                        for neighbor in neighbors
                        if neighbor is not None
                        and self.supervisors[neighbor] == self.supervisors[worker_id]
                    )
                    # Get waiting size
                    lane_sets = self.signal_config[worker_id]['lane_sets']
                    lanes = []
                    for direction in lane_sets:
                        for lane in lane_sets[direction]:
                            if lane not in lanes: lanes.append(lane)
                    waits_len = len(lanes)

                    management_size = len(self.management_neighbors[manager])+1

                    observation_shape = (obs_act[worker_id][0][0] + management_size,)
                    num_actions = obs_act[worker_id][1]
                    self.workers[worker_id] = MA2CAgent(config, observation_shape, num_actions, fp_size, waits_len,
                                                        worker_id + str(thread_number), self.sess)

            self.saver = tf.train.Saver(max_to_keep=1)
            self.sess.run(tf.global_variables_initializer())

        def fingerprints(self, observation):
            agent_fingerprint = {}
            for agent_id in observation.keys():
                fingerprints = []
                if agent_id in self.managers:
                    fingerprints.extend(
                        self.managers[neighbor].fingerprint
                        for neighbor in self.management_neighbors[agent_id]
                    )
                else:
                    downstream = self.signal_config[agent_id]['downstream']
                    neighbors = [downstream[direction] for direction in downstream]
                    fingerprints.extend(
                        self.workers[neighbor].fingerprint
                        for neighbor in neighbors
                        if neighbor is not None
                        and self.supervisors[neighbor] == self.supervisors[agent_id]
                    )
                fp = np.concatenate(fingerprints) if fingerprints else np.asarray([])
                agent_fingerprint[agent_id] = fp
            return agent_fingerprint

        def act(self, observation):
            acts = {}
            full_state = {}    # Includes fingerprints, but not manager acts
            fingerprints = self.fingerprints(observation)
            # First get management's acts, they're part of the state for workers
            for agent_id in self.managers:
                env_obs = observation[agent_id]
                neighbor_fingerprints = fingerprints[agent_id]
                combine = np.concatenate([env_obs, neighbor_fingerprints])
                acts[agent_id] = self.managers[agent_id].act(combine)

            for agent_id in self.workers:
                env_obs = observation[agent_id]
                neighbor_fingerprints = fingerprints[agent_id]

                combine = np.concatenate([env_obs, neighbor_fingerprints])
                full_state[agent_id] = combine

                # Get management goals
                managing_agent = self.supervisors[agent_id]
                managing_agents_acts = [acts[managing_agent]]
                managing_agents_acts.extend(
                    acts[mgr_neighbor]
                    for mgr_neighbor in self.management_neighbors[managing_agent]
                )
                managing_agents_acts = np.asarray(managing_agents_acts)
                combine = np.concatenate([managing_agents_acts, combine])

                acts[agent_id] = self.workers[agent_id].act(combine)
            self.state = full_state
            self.acts = acts
            return acts

        def observe(self, observation, reward, done, info):
            fingerprints = self.fingerprints(observation)

            for agent_id in observation.keys():
                env_obs = observation[agent_id]
                neighbor_fingerprints = fingerprints[agent_id]
                combine = np.concatenate([env_obs, neighbor_fingerprints])

                if agent_id in self.managers:
                    rw = reward[agent_id]
                    self.managers[agent_id].observe(combine, rw, done, info)
                else:
                    managing_agent = self.supervisors[agent_id]
                    managing_agents_acts = [self.acts[managing_agent]]
                    managing_agents_acts.extend(
                        self.acts[mgr_neighbor]
                        for mgr_neighbor in self.management_neighbors[managing_agent]
                    )
                    managing_agents_acts = np.asarray(managing_agents_acts)
                    combine = np.concatenate([managing_agents_acts, combine])
                    self.workers[agent_id].observe(combine, reward[agent_id], done, info)

                if done and info['eps'] % 100 == 0 and self.saver is not None:
                    self.saver.save(self.sess, self.config['log_dir'] + 'agent_' + 'checkpoint',
                                    global_step=info['eps'])
