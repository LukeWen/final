This project is a personal learning endeavor for understanding MAPPO-related concepts. The comments in the source code that start with "Luke" represent my own interpretations and explanations. Some parts of the source code are accompanied by supplementary Markdown files for detailed explanations.
For demonstrating on laptop,the trainning runs on cpu and the treads and trainning times are lower than usual. 
# 1. How to run it
## 1.1. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd on-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 1.1.1. MPE

``` Bash
# install this package first
pip install seaborn
```

There are 3 Cooperative scenarios in MPE:

* simple_spread
* simple_speaker_listener, which is 'Comm' scenario in paper
* simple_reference

## 1.2. Train
Here we use train_mpe_spread.sh as an example:
```
cd onpolicy/scripts/train_mpe_scripts
chmod +x ./train_mpe_spread.sh
./train_mpe_spread.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

# 2. Progress

scripts/train_mpe.py ✅


envs/MPE_env.py ✅


envs/environment.py ✅


envs/mpe/core.py ✅


runner/shared/base_runner.py ✅


runner/separated/base_runner.py ✅


runner/separated/mpe_runner.py ✅


runner/shared/mpe_runner.py ✅


envs/mpe/scenarios/simple_speaker_listener.py ✅


envs/mpe/scenarios/simple_reference.py ✅


envs/mpe/scenarios/simple_spread ✅


algorithms/r_mappo.py ✅


algorithms/algorithm/rMAPPOPolicy.py ✅


algorithms/algorithm/r_actor_critic.py ✅

# 3. Workflow 
train_mpe_xxx.sh with specified parameters 

calls 

train_mpe.py

After all initiating works,
```
    if all_args.share_policy:
        from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()
```
get one MPERunner to run the process.

In the MPERunner, we use these parameters, including all environment settings, to initiate policy network. (Take centralized version as an example.)
```
    self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space, 
                            self.envs.action_space[0],
                            device = self.device)
```
Then, with the specified algorithme code (i.e. PPO), we get trainer


(TODO: add something about policy network)


```
  self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
```
and buffer.
```
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
```
Then, it is the core part of trainning. (TODO: add detailed explaination)
Prepation part
```
    def run(self):
        # Luke: Start the warmup process to initialize the environment and buffer.
        self.warmup()   

        # Luke: Record the start time of the training process.
        start = time.time()
        # Luke: Calculate the total number of episodes based on the number of environment steps, episode length, and the number of rollout threads.
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            # Luke: Apply linear learning rate decay if it's enabled in the configuration.
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
```

Collect,step(executive), and insert 
```

            for step in range(self.episode_length):
                # Sample actions
                # Luke: Collect data by sampling actions from the policy.
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                # Luke: Execute the actions in the environment and observe the results (next observations, rewards, done flags, and info).
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # Luke: Package the collected data into a tuple for easy insertion into the buffer.
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                # Luke: Insert the collected data into the buffer.
                self.insert(data)
```
Compute, post process, save model, log, and evaluate
```
            # compute return and update network
            # Luke: After the episode, compute the returns and update the network.
            self.compute()
            train_infos = self.train()
            
            # post process
            # Luke: Calculate the total number of steps taken so far.
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            # Luke: Save the model at specified intervals or at the last episode.
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            # Luke: Log information at specified intervals.
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                # Luke: Log individual rewards for each agent in the MPE environment.
                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                # Luke: Log the average episode rewards.
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            # Luke: Evaluate the policy at specified intervals if evaluation is enabled.
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
```
Detailes of collectation
```
    @torch.no_grad()
    def collect(self, step):
        # Luke: Prepare the trainer for rollout (e.g., set the model to evaluation mode).
        self.trainer.prep_rollout()
        # Luke: Get the actions, values, and log probabilities from the policy's action distribution.
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        # Luke: Convert the collected values, actions, and log probabilities to numpy arrays and split them by rollout threads.
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        # Luke: Rearrange the actions to match the environment's action space format.
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env
```
Detailes of insertation
```
    def insert(self, data):
        # Luke: Unpack the data tuple into individual components.
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # Luke: Reset RNN states for agents that are done.
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        # Luke: Create masks to indicate whether an agent is done or not.
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # Luke: Prepare the shared observations based on whether centralized value functions are used.
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # Luke: Insert the collected data into the buffer.
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)
```
Detailes of evaluation
```
    @torch.no_grad()
    def eval(self, total_num_steps):
        # Luke: Initialize a list to store rewards for evaluation episodes.
        eval_episode_rewards = []
        # Luke: Reset the evaluation environments to get initial observations.
        eval_obs = self.eval_envs.reset()

        # Luke: Initialize RNN states and masks for the evaluation episodes.
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            # Luke: Prepare the trainer for rollout and perform actions deterministically during evaluation.
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Luke: Rearrange the evaluation actions to match the environment's action space format.
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            # Luke: Execute the evaluation actions in the environment and observe the results.
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # Luke: Reset RNN states for agents that are done during evaluation.
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        # Luke: Calculate the total rewards for evaluation episodes.
        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        # Luke: Log the evaluation results.
        self.log_env(eval_env_infos, total_num_steps)
```


# 4. TBD 

Original repo address is https://github.com/marlbenchmark/on-policy. Following is the original Readme.md
# MAPPO

Chao Yu*, Akash Velu*, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. 

This repository implements MAPPO, a multi-agent variant of PPO. The implementation in this repositorory is used in the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955). This repository is heavily based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. We also make the off-policy repo public, please feel free to try that. [off-policy link](https://github.com/marlbenchmark/off-policy)

<font color="red"> All hyperparameters and training curves are reported in appendix, we would strongly suggest to double check the important factors before runing the code, such as the rollout threads, episode length, ppo epoch, mini-batches, clip term and so on. <font color='red'>Besides, we have updated the newest results on google football testbed and suggestions about the episode length and parameter-sharing in appendix, welcome to check that. </font>

<font color="red"> We have recently noticed that a lot of papers do not reproduce the mappo results correctly, probably due to the rough hyper-parameters description. We have updated training scripts for each map or scenario in /train/train_xxx_scripts/*.sh. Feel free to try that.</font>
 

## Environments supported:

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
- [Hanabi](https://github.com/deepmind/hanabi-learning-environment)
- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)
- [Google Research Football (GRF)](https://github.com/google-research/football)

## 1. Usage
**WARNING: by default all experiments assume a shared policy by all agents i.e. there is one neural network shared by all agents**

All core code is located within the onpolicy folder. The algorithms/ subfolder contains algorithm-specific code
for MAPPO. 

* The envs/ subfolder contains environment wrapper implementations for the MPEs, SMAC, and Hanabi. 

* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 

* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name (in the case of SMAC and the MPEs) can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 


## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd on-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.


### 2.2 Hanabi
Environment code for Hanabi is developed from the open-source environment code, but has been slightly modified to fit the algorithms used here.  
To install, execute the following:
``` Bash
pip install cffi
cd envs/hanabi
mkdir build & cd build
cmake ..
make -j
```
Here are all hanabi [models](https://drive.google.com/drive/folders/1RIcP_rG9NY9UzaWfFsIncDcjASk5h4Nx?usp=sharing).

### 2.3 MPE

``` Bash
# install this package first
pip install seaborn
```

There are 3 Cooperative scenarios in MPE:

* simple_spread
* simple_speaker_listener, which is 'Comm' scenario in paper
* simple_reference

### 2.4 GRF

Please see the [football](https://github.com/google-research/football/blob/master/README.md) repository to install the football environment.

## 3.Train
Here we use train_mpe.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

We additionally provide `./eval_hanabi_forward.sh` for evaluating the hanabi score over 100k trials. 

## 4. Publication

If you find this repository useful, please cite our [paper](https://arxiv.org/abs/2103.01955):
```
@misc{yu2021surprising,
      title={The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games}, 
      author={Chao Yu and Akash Velu and Eugene Vinitsky and Jiaxuan Gao and Yu Wang and Alexandre Bayen and Yi Wu},
      year={2021},
      eprint={2103.01955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

