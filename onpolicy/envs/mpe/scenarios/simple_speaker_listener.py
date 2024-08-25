import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args):
        # Luke: Create a new instance of the world
        world = World()

        # Luke: Set the length of the world, taken from the `episode_length` argument in `args`
        world.world_length = args.episode_length

        # set any world properties first
        # Luke: Set the communication dimension `dim_c` to 3
        world.dim_c = 3

        # Luke: Set the number of landmarks in the world, taken from `args.num_landmarks`
        world.num_landmarks = args.num_landmarks  # 3

        # Luke: Set the world to be in collaborative mode
        world.collaborative = True

        # add agents
        # Luke: Add agents to the world, the number is taken from `args.num_agents`
        world.num_agents = args.num_agents  # 2

        # Luke: Ensure the number of agents is 2, as only 2 agents are supported
        assert world.num_agents == 2, (
            "only 2 agents is supported, check the config.py.")

        # Luke: Create an Agent instance for each agent and store them in the `agents` list of the world
        world.agents = [Agent() for i in range(world.num_agents)]

        # Luke: Initialize properties for each agent
        for i, agent in enumerate(world.agents):
            # Luke: Set the agent's name to 'agent i'
            agent.name = 'agent %d' % i
            # Luke: Disable the agent's collision behavior
            agent.collide = False
            # Luke: Set the size of the agent
            agent.size = 0.075

        # speaker
        # Luke: The first agent is the speaker and is not movable
        world.agents[0].movable = False

        # listener
        # Luke: The second agent is the listener and cannot communicate
        world.agents[1].silent = True

        # add landmarks
        # Luke: Add landmarks to the world, the number is based on `world.num_landmarks`
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]

        # Luke: Initialize properties for each landmark
        for i, landmark in enumerate(world.landmarks):
            # Luke: Set the landmark's name to 'landmark i'
            landmark.name = 'landmark %d' % i
            # Luke: Disable the landmark's collision behavior
            landmark.collide = False
            # Luke: Make the landmark immovable
            landmark.movable = False
            # Luke: Set the size of the landmark
            landmark.size = 0.04

        # make initial conditions
        # Luke: Initialize all conditions in the world
        self.reset_world(world)

        # Luke: Return the constructed world instance
        return world

    def reset_world(self, world):
        # assign goals to agents
        # Luke: Assign goals to each agent, initially set to None
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None

        # want listener to go to the goal landmark
        # Luke: Set the speaker's goal to the listener and randomly choose a landmark as another goal
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)

        # random properties for agents
        # Luke: Assign random color properties to each agent
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        # random properties for landmarks
        # Luke: Assign random color properties to each landmark
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])

        # special colors for goals
        # Luke: Assign special colors for the goals, which are the landmark colors plus an offset
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + \
            np.array([0.45, 0.45, 0.45])

        # set random initial states
        # Luke: Set random initial states for the agents, including position (`p_pos`), velocity (`p_vel`), and communication state (`c`)
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Luke: Set random initial states for the landmarks, including position (`p_pos`) and velocity (`p_vel`)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        # Luke: Return data for benchmarking purposes, this calls the reward function
        return reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        # Luke: Calculate the squared distance from the listener to the landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        # Luke: Return the negative squared distance as the reward value
        return -dist2

    def observation(self, agent, world):
        # goal color
        # Luke: Initialize the goal color as a zero vector
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            # Luke: If the goal exists, get the color of the goal
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        # Luke: Get the positions of all entities (landmarks) relative to the agent
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        # Luke: Get the communication states of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)

        # speaker
        # Luke: If the agent is the speaker, return the goal color
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        # Luke: If the agent is the listener, return the agent's velocity, all entity positions, and communication states
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
