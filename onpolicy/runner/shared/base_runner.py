import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):
        # Luke: Store the configuration dictionary for easy access to parameters.
        self.all_args = config['all_args']
        # Luke: Set up the environments used for training.
        self.envs = config['envs']
        # Luke: Set up the environments used for evaluation.
        self.eval_envs = config['eval_envs']
        # Luke: Define the device (e.g., CPU or GPU) where the computation will occur.
        self.device = config['device']
        # Luke: Set the number of agents that will be trained in the environment.
        self.num_agents = config['num_agents']
        # Luke: Check if the config contains rendering environments and set it up if available.
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

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
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
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

        if self.use_wandb:
            # Luke: Set up directories for saving models and logging if using Weights & Biases.
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            # Luke: Set up directories for saving models and logging if not using Weights & Biases.
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

        # Luke: Depending on whether centralized value functions are used, choose the appropriate observation space.
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        # Luke: Initialize the policy network.
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        # Luke: If a previous model exists, restore the model from the saved directory.
        if self.model_dir is not None:
            self.restore()

        # algorithm
        # Luke: Initialize the training algorithm for the policy.
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
       
        # buffer 
        # Luke: Initialize the shared replay buffer for storing experiences.
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        # Luke: Placeholder method to be implemented for running the training process.
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        # Luke: Placeholder method to be implemented for warming up the agents before training.
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        # Luke: Placeholder method to be implemented for collecting experiences during training.
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        # Luke: Placeholder method to be implemented for inserting experiences into the buffer.
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        # Luke: Prepare the trainer for rollout (e.g., set the model to evaluation mode).
        self.trainer.prep_rollout()
        # Luke: Get the estimated value of the next state from the policy's critic network.
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        # Luke: Split the concatenated next values into separate arrays for each rollout thread.
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        # Luke: Compute the returns based on the next values and update the buffer.
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer."""
        # Luke: Prepare the trainer for training (e.g., set the model to training mode).
        self.trainer.prep_training()
        # Luke: Train the policy using the experiences in the buffer and get training information.
        train_infos = self.trainer.train(self.buffer)      
        # Luke: Perform any necessary updates to the buffer after training.
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        # Luke: Save the state of the policy's actor network.
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        # Luke: Save the state of the policy's critic network.
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        # Luke: Save the state of the value normalizer if it is being used.
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        # Luke: Load the saved state of the policy's actor network.
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            # Luke: Load the saved state of the policy's critic network if not in render mode.
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            # Luke: Load the saved state of the value normalizer if it is being used.
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        # Luke: Log training information for each key in train_infos, using either wandb or tensorboard.
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        # Luke: Log environment information, averaging values if necessary, using either wandb or tensorboard.
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
