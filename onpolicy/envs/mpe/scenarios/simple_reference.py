import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        # Luke: Create a new world instance
        world = World()

        # set any world properties first
        # Luke: Set the length of each episode in the world, taken from `args.episode_length`
        world.world_length = args.episode_length

        # Luke: Set the communication dimension (`dim_c`) to 10
        # Communication dimension (`dim_c`) defines the size of the communication vector for each agent.
        # It determines how much information agents can exchange with each other.
        world.dim_c = 10

        # Luke: Set the world to collaborative mode where agents share rewards
        world.collaborative = True  # whether agents share rewards

        # add agents
        # Luke: Set the number of agents in the world, based on `args.num_agents`
        world.num_agents = args.num_agents  # 2

        # Luke: Ensure that the number of agents is 2, as this scenario only supports 2 agents
        assert world.num_agents == 2, (
            "only 2 agents is supported, check the config.py.")

        # Luke: Create agent instances and store them in the world's `agents` list
        world.agents = [Agent() for i in range(world.num_agents)]

        # Luke: Initialize properties for each agent
        for i, agent in enumerate(world.agents):
            # Luke: Assign a unique name to each agent
            agent.name = 'agent %d' % i
            # Luke: Disable collision for the agents
            agent.collide = False
            # agent.u_noise = 1e-1  # Optional: Uncomment to add action noise
            # agent.c_noise = 1e-1  # Optional: Uncomment to add communication noise

        # add landmarks
        # Luke: Set the number of landmarks in the world, based on `args.num_landmarks`
        world.num_landmarks = args.num_landmarks  # 3

        # Luke: Create landmark instances and store them in the world's `landmarks` list
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]

        # Luke: Initialize properties for each landmark
        for i, landmark in enumerate(world.landmarks):
            # Luke: Assign a unique name to each landmark
            landmark.name = 'landmark %d' % i
            # Luke: Disable collision for the landmarks
            landmark.collide = False
            # Luke: Make landmarks immovable
            landmark.movable = False

        # make initial conditions
        # Luke: Initialize the world with the initial conditions
        self.reset_world(world)

        # Luke: Return the fully constructed world instance
        return world

    def reset_world(self, world):
        # assign goals to agents
        # Luke: Reset goals for all agents to None initially
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None

        # want other agent to go to the goal landmark
        # Luke: Assign goals to the agents, each agent's goal is the other agent and a randomly chosen landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)

        # random properties for agents
        # Luke: Assign random colors to agents using a utility function
        world.assign_agent_colors()

        # random properties for landmarks
        # Luke: Assign specific colors to the landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])

        # special colors for goals
        # Luke: Assign the color of the goal landmark to the goal agent
        world.agents[0].goal_a.color = world.agents[0].goal_b.color
        world.agents[1].goal_a.color = world.agents[1].goal_b.color

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

    def reward(self, agent, world):
        # Luke: If the agent has no goals assigned, return a reward of 0.0
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        # Luke: Calculate the squared distance between the agent's goal_a and goal_b
        dist2 = np.sum(
            np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        # Luke: Return the negative squared distance as the reward
        return -dist2  # np.exp(-dist2) # Optionally, you could use an exponential of the negative distance

    def observation(self, agent, world):
        # goal positions
        # goal_pos = [np.zeros(world.dim_p), np.zeros(world.dim_p)]
        # if agent.goal_a is not None:
        #     goal_pos[0] = agent.goal_a.state.p_pos - agent.state.p_pos
        # if agent.goal_b is not None:
        #     goal_pos[1] = agent.goal_b.state.p_pos - agent.state.p_pos
        # goal color
        # Luke: Initialize the goal color as a zero vector
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
         # if agent.goal_a is not None:
        #     goal_color[0] = agent.goal_a.color
        # Luke: If the agent has a goal landmark, set the goal color to that of the landmark
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

         # get positions of all entities in this agent's reference frame
        # Luke: Get the positions of all landmarks relative to the agent's position
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        # Luke: Get the colors of all landmarks
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        # Luke: Gather the communication states of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)

        # Luke: Return a concatenated array of the agent's velocity, relative positions of entities, goal color, and communication states
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
