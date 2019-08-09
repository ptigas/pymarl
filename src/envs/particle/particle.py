from math import sqrt
from multiagent.environment import MultiAgentEnv as OpenAIMultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np

from envs.multiagentenv import MultiAgentEnv

class Particle(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)

        # load scenario from script
        self.episode_limit = self.args.episode_limit
        self.scenario = scenarios.load(self.args.scenario_name + ".py").Scenario()
        self.world = self.scenario.make_world()
        self.n_agents = len(self.world.policy_agents)
        self.steps = 0
        self.truncate_episodes = getattr(self.args, "truncate_episodes", True) #by default

        if self.args.benchmark:
            self.env = OpenAIMultiAgentEnv(self.world,
                                            self.scenario.reset_world,
                                            self.scenario.reward,
                                            self.scenario.observation,
                                            self.scenario.benchmark_data)
        else:
            self.env = OpenAIMultiAgentEnv(self.world,
                                            self.scenario.reset_world,
                                            self.scenario.reward,
                                            self.scenario.observation)

        self.glob_args = kwargs.get("args")

        self.env.discrete_action_input = False
        # entity_pos = []
        # for entity in self.world.landmarks:
        #     entity_pos.append(entity.state.p_pos)

        # # communication of all other agents
        # # TODO: is it sensible to include all communication into central state??
        # if self.args.state_mode == "all":
        #     comm = []
        #     agent_pos = []
        #     for agent in self.world.agents:
        #         comm.append(agent.state.c)
        #         agent_pos.append(agent.state.p_pos)
        pass

    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        # Terminate if episode_limit is reached
        self.steps += 1
        if self.steps >= self.episode_limit:
            done_n = [True for _ in done_n]
            info_n["episode_limit"] = self.truncate_episodes  # by default True
        else:
            info_n["episode_limit"] = False
        reward_scaling_factor = getattr(self, "reward_scaling_factor", 1.0)
        reward_n = [r*reward_scaling_factor for r in reward_n]

        # test minimum distance to a landmark
        min_dists = []
        for agent in self.world.agents:
            min_dists.append(float("inf"))
            for landmark in self.world.landmarks:
                dist = sqrt(sum((apos-lpos)**2 for apos, lpos in zip(agent.state.p_pos, landmark.state.p_pos)))
                if dist < min_dists[-1]:
                    min_dists[-1] = dist

        info_n["min_dists_mean"] = np.mean(min_dists)
        if hasattr(self.scenario, "n_last_collisions"):
            info_n["n_last_collisions"] = self.scenario.n_last_collisions

        if "n" in info_n:
            del info_n["n"]

        for i, min_dist in enumerate(min_dists):
            info_n["mind_dist__agent{}".format(i)] = min_dist

        return reward_n, done_n, info_n

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for i, _ in enumerate(self.world.policy_agents):
            obs = self.get_obs_agent(i)
            obs_n.append(obs)
        return obs_n

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs = self.env._get_obs(self.world.policy_agents[agent_id])
        if len(obs) < self.get_obs_size():
            obs = np.concatenate([obs, np.zeros((self.get_obs_size() - len(obs)))],
                                 axis=0)  # pad all obs to same length
        return obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return max([o.shape[0] for o in self.env.observation_space])

    def get_state(self, team=None):

        #if self.args.scenario_name == "simple_spread":
        state = np.concatenate(self.get_obs())
        return state
        #else:
        #    raise Exception("not implemented for this scenario!")

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = len(self.get_state())
        return state_size

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.get_total_actions()))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return max([x.n for x in self.env.action_space]) # not sure what happens if inhomogeneous action spaces!

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        if not getattr(self.glob_args, "continuous_episode", False):
            self.env.reset()
        pass

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def get_env_info(self):
        from gym import spaces
        #translate list of discrete action spaces into continuous action spaces (which need to be between 0..1 and should sum to 1)

        action_spaces = [spaces.Box(low=0.0, high=1.0, shape=(acts.n,), dtype=np.float32) for i, acts in enumerate(self.env.action_space)] # number of continuous actions

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": action_spaces,
                    "actions_dtype": np.float32,
                    "normalise_actions": True} # actions should be normalised like a probability
        return env_info