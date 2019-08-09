from envs.multiagentenv import MultiAgentEnv
import numpy as np
import pygame
from utils.dict2namedtuple import convert

'''
A simple grid-world game for N agents trying to capture M prey. No two entities can occupy the same position. 
The world can be either toroidal or bounded.

MOVEMENTS
Both predators and prey can move to the 4 adjacent states or remain in the current one. Movement is executed 
sequentially: first the predators move in a random order, then the prey chooses a random available action 
(i.e. an action that would not lead to a collision with another entity). 
A prey is captured if it cannot move (i.e. if 4 agents block all 4 adjacent fields).

REWARDS 
A captured prey is removed and yields a collaborative reward of +50. 
Forcing the a prey to move (scaring it off), by moving into the same field yields no additional reward. 
Collisions between agents is not punished, and each movement costs additional -0.1 reward. 
An episode ends if all prey have been captured.  

OBSERVATIONS
Prey only react to blocked movements (after the predators have moved), but predator agents observe all positions 
in a square of obs_size=(2*agent_obs+1) centered around the agent's current position. The observations are reshaped 
into a 1d vector of size (2*obs_size), including all predators and prey the agent can observe.

State output is a list of length 2, giving location of all agents and all targets.

TODO: Fine tune the reward to allow fast learning. 
'''

int_type = np.int16
float_type = np.float32


class PredatorPreyCapture(MultiAgentEnv):

    # ---------- CONSTRUCTOR -------------------------------------------------------------------------------------------
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Downwards compatibility of batch_mode
        self.batch_mode = batch_size is not None
        self.batch_size = batch_size if self.batch_mode else 1

        # Define the environment grid
        self.toroidal = args.predator_prey_toroidal
        shape = args.predator_prey_shape
        self.x_max, self.y_max = shape
        self.state_size = self.x_max * self.y_max * 2
        self.env_max = np.asarray(shape, dtype=int_type)
        self.grid_shape = np.asarray(shape, dtype=int_type)
        self.grid = np.zeros((self.batch_size, self.x_max, self.y_max, 2), dtype=float_type) # 0=agents, 1=prey

        # Define the agents
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=int_type)
        self.action_names = ["right", "down", "left", "up", "stay"]
        self.n_actions = self.actions.shape[0]
        self.n_agents = args.n_agents
        self.n_prey = args.n_prey
        self.agent_obs = args.agent_obs
        self.obs_size = 2*(2*args.agent_obs[0]+1)*(2*args.agent_obs[1]+1)

        # Define the episode and rewards
        self.episode_limit = args.episode_limit
        self.time_reward = -0.1
        self.collision_reward = 0.0 #-0.1
        self.scare_off_reward = 0.0
        #self.capture_rewards = [20, 1]
        self.capture_rewards = [50, 1]
        self.capture_terminal = [True, False, False, False, False]

        # Define the internal state
        self.agents = np.zeros((self.n_agents, self.batch_size, 2), dtype=int_type)
        self.prey = np.zeros((self.n_prey, self.batch_size, 2), dtype=int_type)
        self.prey_alive = np.zeros((self.n_prey, self.batch_size), dtype=int_type)
        self.steps = 0
        self.reset()

        self.made_screen = False
        self.scaling = 5

    # ---------- PRIVATE METHODS ---------------------------------------------------------------------------------------
    def _place_actors(self, actors: np.ndarray, type_id: int):
        for b in range(self.batch_size):
            for a in range(actors.shape[0]):
                is_free = False
                while not is_free:
                    # Draw actors's position randomly
                    actors[a, b, 0] = np.random.randint(self.env_max[0])
                    actors[a, b, 1] = np.random.randint(self.env_max[1])
                    # Check if position is valid
                    is_free = np.sum(self.grid[b, actors[a, b, 0], actors[a, b, 1], :]) == 0
                self.grid[b, actors[a, b, 0], actors[a, b, 1], type_id] = 1

    def print_grid(self, batch=0, grid=None):
        if grid is None:
            grid = self.grid
        grid = grid[batch, :, :, :].squeeze().copy()
        for i in range(grid.shape[2]):
            grid[:, :, i] *= i + 1
        grid = np.sum(grid, axis=2)
        print(grid)

    def _env_bounds(self, positions: np.ndarray):
        # positions is 2 dimensional
        if self.toroidal:
            positions = positions % self.env_max
        else:
            positions = np.min(positions, self.env_max - 1)
            positions = np.max(positions, 0)
        return positions

    def _move_actor(self, pos: np.ndarray, action: int, batch: int, collision_mask: np.ndarray, move_type=None):
        # compute hypothetical next position
        new_pos = self._env_bounds(pos + self.actions[action])
        # check for a collision with anything in the collision_mask
        found_at_new_pos = self.grid[batch, new_pos[0], new_pos[1], :]
        collision = np.sum(found_at_new_pos[collision_mask]) > 0
        if collision:
            # No change in position
            new_pos = pos
        elif move_type is not None:
            # change the agent's state and position on the grid
            self.grid[batch, pos[0], pos[1], move_type] = 0
            self.grid[batch, new_pos[0], new_pos[1], move_type] = 1
        return new_pos, collision

    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------
    def reset(self):
        # Reset old episode
        self.prey_alive.fill(1)
        self.steps = 0

        # Clear the grid
        self.grid.fill(0.0)

        # Place n_agents and n_preys on the grid
        self._place_actors(self.agents, 0)
        self._place_actors(self.prey, 1)

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Execute a*bs actions in the environment. """
        if not self.batch_mode:
            actions = np.expand_dims(np.asarray(actions, dtype=int_type), axis=1)
        assert len(actions.shape) == 2 and actions.shape[0] == self.n_agents and actions.shape[1] == self.batch_size, \
            "improper number of agents and/or parallel environments!"
        actions = actions.astype(dtype=int_type)

        # Initialise returned values and grid
        reward = np.ones(self.batch_size, dtype=float_type) * self.time_reward
        terminated = [False for _ in range(self.batch_size)]

        # Move the agents sequentially in random order
        for b in range(self.batch_size):
            for a in np.random.permutation(self.n_agents):
                self.agents[a, b, :], collide = self._move_actor(self.agents[a, b, :], actions[a, b], b,
                                                                 np.asarray([0], dtype=int_type), 0)
                if collide:
                    reward[b] += self.collision_reward

        # Move the prey
        for b in range(self.batch_size):
            for p in np.random.permutation(self.n_prey):
                if self.prey_alive[p, b] > 0:
                    # Collect all allowed actions for the prey
                    possible = []
                    # Run through all potential actions (without actually moving), except stay(4)
                    for u in range(self.n_actions-1):
                        _, c = self._move_actor(self.prey[p, b, :], u, b, np.asarray([0, 1], dtype=int_type))
                        if not c:
                            possible.append(u)
                    # Reward captures of differing magnitude (i.e. involving different numbers of predators)
                    escaped = True
                    for _i in range(len(self.capture_rewards)):
                        if len(possible) == _i:
                            reward[b] += self.capture_rewards[_i]
                            escaped = not self.capture_terminal[_i]
                            break
                    # Check if stay(4) is an option for the prey
                    if self.grid[b, self.prey[p, b, 0], self.prey[p, b, 1], 0] == 0:
                        possible.append(4)
                    else:   # if not, an agent will scare off the prey (i.e. force it to move)
                        reward[b] += self.scare_off_reward
                    # TODO: What should happen if hunter and prey switch positions?
                    # If the prey has escaped the predators...
                    if escaped:
                        # ... the prey chooses one of the allowed actions uniformly...
                        u = possible[np.random.randint(len(possible))]
                        self.prey[p, b, :], _ = self._move_actor(self.prey[p, b, :], u, b,
                                                                 np.asarray([0, 1], dtype=int_type), 1)
                    else:
                        # ... otherwise it is caught and removed from the game
                        self.prey_alive[p, b] = 0
                        self.grid[b, self.prey[p, b, 0], self.prey[p, b, 1], 1] = 0
            terminated[b] = sum(self.prey_alive[:, b]) == 0

        # Terminate if episode_limit is reached
        info = {}
        self.steps += 1
        if self.steps >= self.episode_limit:
            terminated = [True for _ in range(self.batch_size)]
            info["episode_limit"] = True
        else:
            info["episode_limit"] = False

        if self.batch_mode:
            return reward, terminated, info
        else:
            return np.asscalar(reward[0]), int(terminated[0]), info

    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        """ Return a wrapped observation for other agents' locations and targets, the size specified by observation
            shape, centered on the agent. """
        # TODO: implement observations for self.toroidal=False
        a_x, a_y = self.agents[agent_id, batch, :]
        o_x, o_y = self.agent_obs
        x_range = range((a_x - o_x), (a_x + o_x + 1))
        y_range = range((a_y - o_y), (a_y + o_y + 1))
        ex_grid = self.grid[batch, :, :, :].astype(dtype=float_type)
        agent_obs = ex_grid.take(x_range, 0, mode='wrap').take(y_range, 1, mode='wrap')
        return np.resize(agent_obs, self.obs_size)

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        if self.batch_mode:
            return self.grid.copy().reshape(self.state_size)
        else:
            return self.grid[0, :, :, :].reshape(self.state_size)

    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self, agent_id):
        return [1 for _ in range(self.n_actions)]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_stats(self):
        pass

    # --------- RENDER METHODS -----------------------------------------------------------------------------------------
    def close(self):
        if self.made_screen:
            pygame.quit()
        print("Closing Multi-Agent Navigation")

    def render_array(self):
        # Return an rgb array of the frame
        return None

    def render(self):
        # TODO!
        pass

    def seed(self):
        raise NotImplementedError


# ######################################################################################################################
if __name__ == "__main__":
    env_args = {
        # SC2
        'map_name': '5m_5m',
        'n_enemies': 5,
        'move_amount': 2,
        'step_mul': 8,  # lets you skip observations and actions
        'difficulty': "3",
        'reward_only_positive': True,
        'reward_negative_scale': 0.5,
        'reward_death_value': 10,
        'reward_win': 200,
        'reward_scale': True,
        'reward_scale_rate': 2,
        'state_last_action': True,
        'heuristic_function': False,
        'measure_fps': True,
        # Payoff
        'payoff_game': "monotonic",
        # Predator Prey
        'prey_movement': "escape",
        'predator_prey_shape': (4, 4),
        'predator_prey_toroidal': True,
        'nagent_capture_enabled': False,
        # Stag Hunt
        'stag_hunt_shape': (3, 3),
        'stochastic_reward_shift_optim': None,  # e.g. (1.0, 4) = (p, Delta_r)
        'stochastic_reward_shift_mul': None,  # e.g. (0.5, 2) = (p, Factor_r)
        'global_reward_scale_factor': 1.0,
        'state_variant': "grid",  # comma-separated string
        'n_prey': 1,
        'agent_obs': (1, 1),
        'episode_limit': 20,
        'n_agents': 4,
    }

    env = PredatorPreyCapture(env_args=env_args)
    print("Env is ", "batched" if env.batch_mode else "not batched")

    [all_obs, state] = env.reset()
    print(state)
    for i in range(env.n_agents):
        print(all_obs[i])

    acts = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]]).transpose()
    env.step(acts[:, 0])

    env.print_grid()
    obs = []
    for i in range(4):
        obs.append(np.expand_dims(env.get_obs_agent(i), axis=1))
    print(np.concatenate(obs, axis=1))