import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete

# update bounds to center around agent
cam_range = 2

# Luke: cam_range is used to define the camera's range for rendering, typically in a visual representation of the environment.

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    # Luke: metadata defines the rendering modes available for this environment, such as human-viewable or RGB array.

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=True):
        # Luke: world is an object representing the environment, containing all entities (agents, obstacles, etc.).
        # Luke: reset_callback, reward_callback, observation_callback, info_callback, done_callback, post_step_callback 
        # are functions passed to customize the behavior of the environment.
        # shared_viewer indicates whether all agents share the same viewer during rendering.
        # discrete_action indicates whether the action space is discrete.

        # Luke: self.world is the environment instance, self.world_length is the length of the world (e.g., number of steps).
        self.world = world
        # Luke: self.world_length stores the total length or duration of an episode in the environment.
        # It is set to the value of world.world_length, which defines how many steps or time units 
        # the environment runs before an episode ends. This ensures the environment's behavior 
        # follows the predefined length set in the world configuration.
        self.world_length = self.world.world_length
        # Luke: self.current_step tracks the number of steps in the current episode.
        self.current_step = 0
        # Luke: self.agents stores all agents in the environment.
        self.agents = self.world.policy_agents
        
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # Luke: self.n is the number of agents in the environment.

        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        
        self.post_step_callback = post_step_callback
        # Luke: These callbacks allow customization of how the environment is reset, how rewards are calculated, 
        # how observations are generated, and how termination conditions are handled.

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = discrete_action
        # Luke: self.discrete_action_space determines whether the actions are discrete.

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # Luke: self.discrete_action_input determines the format of the input action (either an integer or a one-hot vector).

        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # in this env, force_discrete_action == False , because world do not have discrete_action
        # Luke: self.force_discrete_action ensures that continuous actions are executed as discrete actions if necessary. 
        # It checks if the world object has a discrete_action attribute.

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        #self.shared_reward = False
        # Luke: self.shared_reward indicates whether all agents share the same reward, promoting collaboration.

        self.time = 0
        # Luke: self.time is used to track the elapsed time or steps in the environment.

        # configure spaces
        # Luke: These lists will hold the action spaces, observation spaces, and shared observation spaces for each agent.
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        # Luke: share_obs_dim accumulates the total dimension of observations across all agents.
        # This value is used to define the shared observation space, allowing agents to 
        # potentially access combined observational data from all agents, which can be 
        # useful for cooperative tasks where global awareness is beneficial.
        

        for agent in self.agents:
            # Luke: total_action_space will hold the combined action space for an agent, including physical and communication actions.
            total_action_space = []
            # Luke: Physical actions refer to the movement-related actions that an agent can take,
            # such as changing position within the environment. These are typically represented
            # by vectors or discrete values depending on the action space configuration.
            # Communication actions involve information exchange between agents, allowing them
            # to coordinate or share knowledge. These actions are also defined by either 
            # discrete or continuous spaces and are essential in multi-agent environments where
            # cooperation or strategy sharing is required.

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(
                    low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)  # [-1,1]
            # Luke: u_action_space defines the action space for physical movement. 
            # If the action space is discrete, it ranges over possible movements; 
            # if continuous, it uses a box to define ranges.

            if agent.movable:
                total_action_space.append(u_action_space)
            # Luke: Only append the physical action space if the agent is movable.

            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(
                    world.dim_c,), dtype=np.float32) # [0,1]
            # Luke: c_action_space defines the action space for communication. Like physical actions, it can be discrete or continuous.

            if not agent.silent:
                total_action_space.append(c_action_space)
            # Luke: Only append the communication action space if the agent is not silent (i.e., it can communicate).

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # Luke: If there is more than one action space, combine them into a single action space 
            # (either MultiDiscrete or Tuple). Otherwise, use the single action space directly.

            # observation space
            # Luke: Define the observation space for the agent based on the observation dimension returned by the callback.
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)) # [-inf, inf]
            
            # Luke: Initialize the agent's communication action to zero.
            agent.action.c = np.zeros(self.world.dim_c)

        # Luke: Define the shared observation space, which accumulates the dimensions of all agents' observations.
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

        # rendering
        # Luke: Set up the viewer(s) for rendering. If shared_viewer is True, all agents share the same viewer; otherwise, each agent has its own.
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        
        # Luke: Initialize the rendering assets by calling _reset_render().
        self._reset_render()      

    # Luke: Set the random seed for reproducibility. If no seed is provided, default to 1.
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
        

    # step  this is  env.step()
    def step(self, action_n):
        # Luke: Update the environment by one step. Track the current step, and prepare to collect observations, rewards, done flags, and info for each agent.
        # Luke: Track the number of steps in the current episode.
        self.current_step += 1
        # Luke: Store observations for each agent.
        obs_n = []
        # Luke: Store rewards for each agent.
        reward_n = []
        # Luke: Store done flags for each agent.
        done_n = []
        # Luke: Store additional information for each agent.
        info_n = []
        # Luke: Update the list of active agents.
        self.agents = self.world.policy_agents

        # set action for each agent
        # Luke: For each agent, set the action based on the provided action space.
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        
        # advance world state
        # Luke: Advance the state of the world by one step.
        self.world.step()  # core.step()
        
        # record observation for each agent
        # Luke: For each agent, collect the observation, reward, done flag, and additional info. If the agent fails, record it in the info.
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        # Luke: If a post-step callback is defined, call it.
        # The post-step callback is executed after the environment's state
        # has been updated and rewards, observations, and done flags have been calculated.
        # It allows for any additional processing or custom logic that should occur at the end
        # of each step, such as logging, modifying the environment, or applying global effects.
        if self.post_step_callback is not None:
            self.post_step_callback(self.world)
        
        # Luke: Return the collected observations, rewards, done flags, and info for all agents
        return obs_n, reward_n, done_n, info_n
    
    # Luke: Reset the environment to its initial state at the beginning of a new episode.
    def reset(self):
        # Luke: Reset the current step counter to 0 at the beginning of a new episode.
        self.current_step = 0

        # reset world
        # Luke: Call the reset callback to reinitialize the world state.
        self.reset_callback(self.world)
        

        # reset renderer
        # Luke: Reset the rendering assets to their initial state.
        self._reset_render()
        
        # record observations for each agent
        # Luke: Prepare to collect observations from all agents after resetting the world.
        obs_n = []
        self.agents = self.world.policy_agents
        
        # Luke: Collect the initial observations for each agent after the environment has been reset.
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        # Luke: Return the list of initial observations.
        return obs_n 

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)
        # Luke: If the info_callback is provided, call it to get additional information about the agent's state for benchmarking purposes.

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)
        # Luke: If the observation_callback is provided, call it to generate the agent's observation. Otherwise, return an empty observation.

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)
        # Luke: Determine if an agent is done (i.e., if the episode should end for this agent). If no done_callback is provided, the episode ends when the maximum number of steps is reached.

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)
        # Luke: Calculate the reward for the agent using the reward_callback if provided. Otherwise, return a default reward of 0.0.

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
         # Luke: Initialize the agent's physical and communication actions to zero.
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        
        # process action
        # Luke: If the action space is MultiDiscrete, split the action into its components. Otherwise, treat the action as a single item.
        if isinstance(action_space, MultiDiscrete):
            act = []
            # Luke: Calculate the size of each sub-action by finding the difference 
            # between the high and low bounds of the action space, and adding 1. (Both sides include)
            size = action_space.high - action_space.low + 1
            index = 0
            # Luke: Iterate over each size value, splitting the action array into parts corresponding to each sub-action's size.
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        # Luke: Check if the agent is movable, meaning it can physically move in the environment.
        if agent.movable:
            
            # Luke: If the action input is discrete (e.g., a single integer), initialize the movement action vector to zeros.
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                
                # Luke: Process the discrete action to determine movement direction.
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0  # Move left
                if action[0] == 2:
                    agent.action.u[0] = +1.0  # Move right
                if action[0] == 3:
                    agent.action.u[1] = -1.0  # Move down
                if action[0] == 4:
                    agent.action.u[1] = +1.0  # Move up
                
                # Luke: Set d to the number of physical dimensions (e.g., 2D space).
                d = self.world.dim_p
            
            # Luke: If the action input is not a simple discrete action.
            else:
                # Luke: If the action space is discrete, calculate the movement by subtracting left (action[0][1]) / right (action[0][2])
                # and down (action[0][3])/up (action[0][4]) action components.
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]  # Horizontal movement
                    agent.action.u[1] += action[0][3] - action[0][4]  # Vertical movement
                    d = 5  # Luke: d is set to 5 because there are 5 possible discrete actions in the space.
                
                # Luke: If the action space is continuous, process the action as a vector.
                else:
                    if self.force_discrete_action:
                        p = np.argmax(action[0][0:self.world.dim_p])
                        action[0][:] = 0.0
                        action[0][p] = 1.0  # Set the most likely action to 1 (one-hot vector).
                    agent.action.u = action[0][0:self.world.dim_p]
                    # Luke: Set d to the number of physical dimensions.
                    d = self.world.dim_p  

            # Luke: Apply the agent's sensitivity (acceleration) to scale the movement. default is 5.0.
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

            # Luke: If the agent is not silent and the action space is not MultiDiscrete, remove the used portion of the action for further processing.
            if (not agent.silent) and (not isinstance(action_space, MultiDiscrete)):
                action[0] = action[0][d:]
            else:
                action = action[1:]


        if not agent.silent:
            # communication action
            # Luke: Set the agent's communication action based on whether the input is discrete or continuous.
            # Initialize communication action as a one-hot vector based on the discrete action input.
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]

            # Luke: Remove the used portion of the action for further processing.
            action = action[1:]

        # make sure we used all elements of action
        # Luke: Ensure that all parts of the action have been processed and no action elements remain.
        assert len(action) == 0

    # reset rendering assets
    # Luke: Clear the rendering geometries and transformations to prepare for a new rendering cycle.
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None   

    def render(self, mode='human', close=False):
        if close:
            # close any existing renderers
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []
        # Luke: Close any active renderers if the close flag is set. This is useful for cleaning up when the environment is no longer in use.
        
        # Luke: In human mode, generate and print a communication message between agents based on their communication states.
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            
            for agent in self.world.agents:
                # Luke: Initialize a list to hold the communication signals for the current agent.
                comm = []  
                
                for other in self.world.agents:
                    if other is agent:
                        continue
                    # Luke: If the other agent's communication state is all zeros, use '_' to indicate no communication.
                    # Otherwise, find the index of the highest communication signal and map it to a letter.
                    if np.all(other.state.c == 0):
                        word = '_'  
                    else:
                        word = alphabet[np.argmax(other.state.c)]  
                    
                    message += (other.name + ' to ' +
                                agent.name + ': ' + word + '   ')  
            
            print(message)

        # Luke: Initialize viewers if they haven't been created yet. This uses a 700x700 pixel window for each viewer.
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from . import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from . import rendering
            # Luke: If rendering geometries haven't been created, initialize the lists to store them.
            self.render_geoms = []
            self.render_geoms_xform = []
            self.comm_geoms = []
            
            # Luke: For each entity in the world, create its physical and communication geometries, and add them to the respective lists.
            for entity in self.world.entities:
                # Luke: Create a circular geometry for each entity in the world, and initialize a transformation object for it.
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                entity_comm_geoms = []

                if 'agent' in entity.name:
                    # Luke: Set the color of the agent's geometry with a semi-transparent alpha value.
                    geom.set_color(*entity.color, alpha=0.5)
                    # Luke: If the agent is not silent (i.e., it can communicate), create visual indicators for communication.
                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        # Luke: Create small circles representing communication channels around the agent.
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                else:
                    # Luke: For non-agent entities, set their color without transparency.
                    geom.set_color(*entity.color)
                    if entity.channel is not None:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        # Luke: Create small circles representing communication channels around the entity.
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                # Luke: Add the transformation attribute to the main geometry.
                geom.add_attr(xform)
                # Luke: Add the main geometry and its transformation to the rendering lists.
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                # Luke: Add the communication geometries to the list of communication visuals.
                self.comm_geoms.append(entity_comm_geoms)
            
            # Luke: For each wall in the world, create its geometric representation and add it to the list of render geometries.
            for wall in self.world.walls:
                # Luke: Calculate the four corners of the wall based on its position (axis_pos), width, and endpoints.
                corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                           (wall.axis_pos - 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                # Luke: If the wall is horizontally oriented ('H'), reverse the corner coordinates to match the orientation.
                if wall.orient == 'H':
                    corners = tuple(c[::-1] for c in corners)
                # Luke: Create a polygon geometry for the wall using the calculated corners.
                geom = rendering.make_polygon(corners)
                # Luke: Set the color of the wall; if it's a hard wall, use full opacity, otherwise use semi-transparency.
                if wall.hard:
                    geom.set_color(*wall.color)
                else:
                    geom.set_color(*wall.color, alpha=0.5)
                # Luke: Add the wall's geometry to the list of objects to be rendered.
                self.render_geoms.append(geom)
            

            # add geoms to viewer
            # Luke: Add all created geometries to each viewer for rendering.
            # for viewer in self.viewers:
            #     viewer.geoms = []
            #     for geom in self.render_geoms:
            #         viewer.add_geom(geom)
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)
            

        results = []
        for i in range(len(self.viewers)):
            from . import rendering
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            # Luke: Set the bounds of the viewer based on the agent's position or the shared viewer's position.
            self.viewers[i].set_bounds(
                pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            
            # update geometry positions
            # Luke: Update the position and color of each entity's geometries in the viewer, based on the current state of the world.
            for e, entity in enumerate(self.world.entities):
                # Luke: Set the translation (position) of the entity's geometry based on its current position in the environment.
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                # Luke: Check if the entity is an agent by looking for 'agent' in its name.
                if 'agent' in entity.name:
                    # Luke: Set the agent's color with semi-transparency (alpha = 0.5).
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)
                    # Luke: If the agent is not silent, update the color of its communication channels.
                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            # Luke: Calculate the color for the communication channel based on its state.
                            color = 1 - entity.state.c[ci]
                            # Luke: Set the color of the corresponding communication geometry.
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
                else:
                     # Luke: Set the entity's color with full opacity
                    self.render_geoms[e].set_color(*entity.color)
                    # Luke: If the entity has communication channels, update their colors.
                    if entity.channel is not None:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.channel[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
            

            # render to display or array
            results.append(self.viewers[i].render(
                return_rgb_array=mode == 'rgb_array'))
            # Luke: Render the environment to the display or return an RGB array if specified.

        return results
        # Luke: Return the rendering results for each viewer.

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        # Luke: Initialize receptor locations based on the specified type (polar or grid), and set the range of distances for the receptors.
        receptor_type = 'polar'
        # Luke: Define the minimum and maximum ranges for the receptors, determining how close and far they can detect.
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = [] 

        # circular receptive field
        # Luke: In polar mode, create receptor locations arranged in a circular pattern around the agent, including the origin.
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin point
            dx.append(np.array([0.0, 0.0]))

        # grid receptive field
        # Luke: In grid mode, create receptor locations arranged in a grid around the agent.
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        
        # Luke: Return the list of receptor locations.
        return dx
