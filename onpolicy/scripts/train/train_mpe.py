#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""

def make_train_env(all_args):
    # Luke: This function creates the environment for training.
    # Luke: It determines whether to use a single-threaded or multi-threaded environment setup.
    def get_env_fn(rank):
        def init_env():
            # Luke: Initialize the MPE environment based on the provided arguments.
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                # Luke: Raise an error if the environment name is not supported.
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            # Luke: Set the seed for the environment, ensuring reproducibility.
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    # Luke: Return either a single-threaded (DummyVecEnv) or multi-threaded (SubprocVecEnv) environment based on the number of rollout threads.
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    # Luke: This function creates the environment for evaluation.
    # Luke: Similar to make_train_env, but with different seeding to ensure evaluation is independent of training.
    def get_env_fn(rank):
        def init_env():
            # Luke: Initialize the MPE environment based on the provided arguments.
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                # Luke: Raise an error if the environment name is not supported.
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            # Luke: Set the seed for the evaluation environment.
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    # Luke: Return either a single-threaded (DummyVecEnv) or multi-threaded (SubprocVecEnv) environment based on the number of evaluation rollout threads.
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    # Luke: This function parses the command-line arguments and returns the parsed arguments.
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    # Luke: Parse the arguments using the provided parser and return them.
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # Luke: Main function that orchestrates the training process.
    parser = get_config()
    all_args = parse_args(args, parser)

    # Luke: Set specific flags based on the chosen algorithm.
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # Luke: Ensure that shared policy is not used with the simple_speaker_listener scenario.
    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    # if all_args.cuda and torch.cuda.is_available():
    #     print("choose to use gpu...")
    #     device = torch.device("cuda:0")
    #     torch.set_num_threads(all_args.n_training_threads)
    #     if all_args.cuda_deterministic:
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True
    # else:
    # Luke: For demonstration on a laptop, the training runs on CPU.
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    # Luke: Create the directory structure for storing results.
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        # Luke: Initialize a new Weights and Biases run for logging and tracking experiments.
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        # Luke: Create a new directory for storing results if wandb is not used.
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            # Luke: Determine the current run number by checking existing run directories.
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # Luke: Set the process title for easier identification in process monitors.
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    # Luke: Set the random seeds for torch and numpy to ensure reproducibility.
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    # Luke: Initialize the training and evaluation environments.
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    # Luke: Create a configuration dictionary that will be passed to the Runner.
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    # Luke: Import and instantiate the appropriate runner based on whether policies are shared or separated.
    if all_args.share_policy:
        from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    # Luke: Start the training process by calling the run method of the runner.
    runner.run()
    
    # post process
    # Luke: Close the environments after training is complete.
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # Luke: Finalize the wandb run or export the results to a JSON file.
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    # Luke: Entry point of the script. Pass command-line arguments to the main function.
    main(sys.argv[1:])
