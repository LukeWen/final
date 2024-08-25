import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        # Luke: Initializes the R_MAPPO class with arguments for the model and device settings.
        # Luke: self.device stores the device (CPU/GPU) to be used for computations.
        self.device = device
        
        # Luke: tpdv is a dictionary holding tensor properties like data type and device for use later.
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Luke: The policy to be updated is stored in self.policy.
        self.policy = policy
        
        # Luke: Various hyperparameters are initialized from the provided args.
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        # Luke: Flags for using different model variants and loss functions are set.
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        # Luke: Ensure that both popart and valuenorm are not activated simultaneously.
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        # Luke: Set up value normalization based on the flags.
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timestep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        # Luke: Compute the clipped value predictions by constraining the change from old predictions.
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                   self.clip_param)
        # Luke: Normalize the return batch if popart or valuenorm is used, then compute errors.
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        # Luke: Calculate the value loss using either Huber loss or MSE loss.
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        # Luke: Use the maximum of original and clipped value losses if specified.
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # Luke: Apply active masks if specified to calculate the mean value loss.
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :param update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        :return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        # Luke: Unpack the data sample.
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        # Luke: Prepare the tensors for computation by moving them to the appropriate device.
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        # Luke: Perform a forward pass to get the values, action log probabilities, and entropy for all steps.
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        # Luke: Update the actor by calculating the importance sampling weights.
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        # Luke: Calculate the surrogate loss functions for the policy.
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # Luke: Calculate the policy action loss, considering active masks if specified.
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # Luke: Zero the gradients for the actor optimizer before updating.
        self.policy.actor_optimizer.zero_grad()

        # Luke: Perform backpropagation for the policy loss if actor update is enabled.
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        # Luke: Clip the gradients if max grad norm is enabled, otherwise calculate the gradient norm.
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        # Luke: Step the actor optimizer to apply the gradient update.
        self.policy.actor_optimizer.step()

        # critic update
        # Luke: Update the critic by calculating the value loss.
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        # Luke: Zero the gradients for the critic optimizer before updating.
        self.policy.critic_optimizer.zero_grad()

        # Luke: Backpropagate the value loss scaled by the value loss coefficient.
        (value_loss * self.value_loss_coef).backward()

        # Luke: Clip the gradients for the critic if max grad norm is enabled, otherwise calculate the gradient norm.
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        # Luke: Step the critic optimizer to apply the gradient update.
        self.policy.critic_optimizer.step()

        # Luke: Return the losses, gradient norms, entropy, and importance weights for logging.
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # Luke: Compute the advantages for the training data.
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        # Luke: Copy the advantages and mask inactive agents.
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        
        # Luke: Normalize the advantages using mean and standard deviation.
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        # Luke: Initialize a dictionary to store the training information.
        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        # Luke: Perform training updates for the number of PPO epochs specified.
        for _ in range(self.ppo_epoch):
            # Luke: Use the appropriate data generator based on the policy type.
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            # Luke: Iterate over the samples generated by the data generator.
            for sample in data_generator:
                # Luke: Perform a PPO update with the current sample.
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                # Luke: Accumulate the losses and gradient norms for logging.
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        # Luke: Calculate the average of the accumulated values by dividing by the number of updates.
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        # Luke: Return the training information after the update.
        return train_info

    def prep_training(self):
        # Luke: Set the actor and critic to training mode.
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        # Luke: Set the actor and critic to evaluation mode for rollout.
        self.policy.actor.eval()
        self.policy.critic.eval()
