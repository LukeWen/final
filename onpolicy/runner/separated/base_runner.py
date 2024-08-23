    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):
        # Luke: Store the entire configuration dictionary for easy access later.
        self.all_args = config['all_args']
        # Luke: Set up the environments used for training.
        self.envs = config['envs']
        # Luke: Set up the environments used for evaluation.
        self.eval_envs = config['eval_envs']
        # Luke: Define the device (e.g., CPU or GPU) where the computation will occur.
        self.device = config['device']
        # Luke: Set the number of agents that will be trained in the environment.
        self.num_agents = config['num_agents']

        # parameters
        # Luke: Extract and store various parameters from the configuration for easy access.
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        # Luke: These parameters define intervals for saving models, evaluation, and logging during training.
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        # Luke: Directory where the models are saved. If a previous model exists, it will be loaded.
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            # Luke: If rendering is enabled, set up directories for saving GIFs.
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            # Luke: Set up directories for logging and saving models.
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        # Luke: Import the training algorithm and policy classes.
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        # Luke: Initialize the policy network for each agent.
        self.policy = []
        for agent_id in range(self.num_agents):
            # Luke: Depending on whether centralized value functions are used, choose the appropriate observation space.
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # Luke: Initialize the policy for the current agent.
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        # Luke: If a previous model exists, restore the model from the saved directory.
        if self.model_dir is not None:
            self.restore()

        # Luke: Initialize the trainer and buffer for each agent.
        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # Luke: Initialize the training algorithm for the current agent.
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            # Luke: Initialize the buffer to store experiences for training.
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
    def run(self):
        # Luke: Placeholder method to be implemented for running the training process.
        raise NotImplementedError

    def warmup(self):
        # Luke: Placeholder method to be implemented for warming up the agents before training.
        raise NotImplementedError

    def collect(self, step):
        # Luke: Placeholder method to be implemented for collecting experiences during training.
        raise NotImplementedError

    def insert(self, data):
        # Luke: Placeholder method to be implemented for inserting experiences into the buffer.
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        # Luke: Calculate the value of the next state for each agent to estimate the returns.
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            # Luke: Compute the returns based on the next value estimated by the critic.
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        # Luke: Train the agents using the collected experiences and return the training information.
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        # Luke: Save the state of the policy (actor and critic) for each agent.
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            # Luke: Save the value normalizer state if it is being used.
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")

    def restore(self):
        # Luke: Load the saved state of the policy (actor and critic) for each agent.
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            # Luke: Load the saved state of the value normalizer if it is being used.
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(agent_id) + '.pt')
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        # Luke: Log the training information for each agent at the specified steps.
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        # Luke: Log the environment information at the specified steps.
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
