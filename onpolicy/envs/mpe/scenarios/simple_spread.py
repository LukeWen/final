import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        # Luke: Create a new instance of the world
        world = World()

        # Luke: Set the length of each episode in the world, taken from `args.episode_length`
        world.world_length = args.episode_length

        # set any world properties first
        # Luke: Set the communication dimension (`dim_c`) to 2
        world.dim_c = 2

        # Luke: Set the number of agents in the world based on `args.num_agents`
        world.num_agents = args.num_agents

        # Luke: Set the number of landmarks in the world based on `args.num_landmarks`
        world.num_landmarks = args.num_landmarks  # 3

        # Luke: Enable collaborative mode where agents share rewards
        world.collaborative = True

        # add agents
        # Luke: Create agent instances and add them to the world
        world.agents = [Agent() for i in range(world.num_agents)]

        # Luke: Initialize properties for each agent
        for i, agent in enumerate(world.agents):
            # Luke: Assign a unique name to each agent
            agent.name = 'agent %d' % i
            # Luke: Enable collision detection for the agents
            agent.collide = True
            # Luke: Make agents silent (no communication)
            agent.silent = True
            # Luke: Set the size of the agents
            agent.size = 0.15

        # add landmarks
        # Luke: Create landmark instances and add them to the world
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]

        # Luke: Initialize properties for each landmark
        for i, landmark in enumerate(world.landmarks):
            # Luke: Assign a unique name to each landmark
            landmark.name = 'landmark %d' % i
            # Luke: Disable collision detection for the landmarks
            landmark.collide = False
            # Luke: Make landmarks immovable
            landmark.movable = False

        # make initial conditions
        # Luke: Initialize the world with the initial conditions
        self.reset_world(world)

        # Luke: Return the fully constructed world instance
        return world

    def reset_world(self, world):
        # random properties for agents
        # Luke: Assign random colors to agents using a utility function
        world.assign_agent_colors()

        # Luke: Assign random colors to landmarks using a utility function
        world.assign_landmark_colors()

        # set random initial states
        # Luke: Set random initial states for agents, including position (`p_pos`), velocity (`p_vel`), and communication state (`c`)
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Luke: Set random initial positions for landmarks, scaled by 0.8
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # Luke: Initialize variables for benchmarking data
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0

        # Luke: Calculate distances from agents to landmarks and determine rewards
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1

        # Luke: Check for collisions between agents and penalize if any occur
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1

        # Luke: Return the benchmarking data as a tuple
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        # Luke: Calculate the distance between two agents
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        # Luke: Determine the minimum distance for a collision based on agent sizes
        dist_min = agent1.size + agent2.size

        # Luke: Return True if the distance is less than the minimum required for a collision
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # Luke: Initialize the reward variable
        rew = 0

        # Luke: Calculate and accumulate rewards based on the minimum distance from any agent to each landmark
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)

        # Luke: Penalize the agent if it collides with other agents
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        # Luke: Return the calculated reward
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # Luke: Gather the positions of all landmarks relative to the agent's position
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        # Luke: Gather the colors of all landmarks
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        # Luke: Gather the communication states and positions of all other agents relative to this agent
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # Luke: Return a concatenated array of the agent's velocity, position, relative positions of entities, and communication states
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

