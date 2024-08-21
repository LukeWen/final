from .environment import MultiAgentEnv
from .scenarios import load


def MPEEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # Luke: Load the scenario module using the provided scenario name (without the .py extension).
    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    
    # Luke: Create the world for the scenario. The world contains all the entities, agents, and dynamics
    # specific to the scenario.
    # create world
    world = scenario.make_world(args)
    
    # Luke: Instantiate the MultiAgentEnv object, passing in the world, and the scenario-specific functions 
    # for resetting the world, computing rewards, generating observations, and providing additional info.
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)

    return env
