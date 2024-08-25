import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        # Luke: Initialize the parent class nn.Module.
        super(R_Actor, self).__init__()
        # Luke: Store the hidden layer size from the arguments.
        self.hidden_size = args.hidden_size

        # Luke: Store the gain factor used for initializing weights.
        self._gain = args.gain
        # Luke: Store whether orthogonal initialization should be used.
        self._use_orthogonal = args.use_orthogonal
        # Luke: Store whether to use active masks for policy computation.
        self._use_policy_active_masks = args.use_policy_active_masks
        # Luke: Store whether to use a naive recurrent policy.
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        # Luke: Store whether to use a recurrent policy.
        self._use_recurrent_policy = args.use_recurrent_policy
        # Luke: Store the number of recurrent layers to be used.
        self._recurrent_N = args.recurrent_N
        # Luke: Set the dtype and device for tensors.
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Luke: Determine the observation shape based on the observation space.
        obs_shape = get_shape_from_obs_space(obs_space)
        # Luke: Choose the base network architecture (CNNBase for images, MLPBase for others) based on observation shape.
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        # Luke: Initialize the base network with the selected architecture and observation shape.
        self.base = base(args, obs_shape)

        # Luke: Initialize the RNN layer if using a recurrent policy.
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # Luke: Initialize the action layer with the action space, hidden size, orthogonal initialization flag, and gain.
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        # Luke: Move the model to the specified device (CPU/GPU).
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # Luke: Convert observations, RNN states, and masks to tensors and move them to the correct device.
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            # Luke: Convert available actions to tensors and move them to the correct device if not None.
            available_actions = check(available_actions).to(**self.tpdv)

        # Luke: Extract features from the base network using the observation input.
        actor_features = self.base(obs)

        # Luke: Pass the features through the RNN layer if using a recurrent policy.
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Luke: Compute actions and their log probabilities from the action layer.
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # Luke: Convert observations, RNN states, actions, and masks to tensors and move them to the correct device.
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            # Luke: Convert available actions to tensors and move them to the correct device if not None.
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            # Luke: Convert active masks to tensors and move them to the correct device if not None.
            active_masks = check(active_masks).to(**self.tpdv)

        # Luke: Extract features from the base network using the observation input.
        actor_features = self.base(obs)

        # Luke: Pass the features through the RNN layer if using a recurrent policy.
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Luke: Compute log probabilities and entropy of the given actions using the action layer.
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        # Luke: Initialize the parent class nn.Module.
        super(R_Critic, self).__init__()
        # Luke: Store the hidden layer size from the arguments.
        self.hidden_size = args.hidden_size
        # Luke: Store whether orthogonal initialization should be used.
        self._use_orthogonal = args.use_orthogonal
        # Luke: Store whether to use a naive recurrent policy.
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        # Luke: Store whether to use a recurrent policy.
        self._use_recurrent_policy = args.use_recurrent_policy
        # Luke: Store the number of recurrent layers to be used.
        self._recurrent_N = args.recurrent_N
        # Luke: Store whether to use PopArt normalization for value output.
        self._use_popart = args.use_popart
        # Luke: Set the dtype and device for tensors.
        self.tpdv = dict(dtype=torch.float32, device=device)
        # Luke: Choose initialization method based on orthogonal initialization flag.
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        # Luke: Determine the shape of the centralized observation based on the observation space.
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        # Luke: Choose the base network architecture (CNNBase for images, MLPBase for others) based on observation shape.
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        # Luke: Initialize the base network with the selected architecture and centralized observation shape.
        self.base = base(args, cent_obs_shape)

        # Luke: Initialize the RNN layer if using a recurrent policy.
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            # Luke: Initialize the given module with the selected initialization method.
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # Luke: Initialize the output layer for value prediction, using PopArt if specified.
        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        # Luke: Move the model to the specified device (CPU/GPU).
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # Luke: Convert centralized observations, RNN states, and masks to tensors and move them to the correct device.
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # Luke: Extract features from the base network using the centralized observation input.
        critic_features = self.base(cent_obs)
        # Luke: Pass the features through the RNN layer if using a recurrent policy.
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        # Luke: Compute value predictions using the output layer.
        values = self.v_out(critic_features)

        return values, rnn_states
