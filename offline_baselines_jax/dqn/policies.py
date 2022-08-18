from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gym
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import functools

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.jax_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from offline_baselines_jax.common.type_aliases import Schedule, Params


@functools.partial(jax.jit, static_argnames=('q_fn', ))
def sample_actions(q_fn: Callable[..., Any], q_params: Params, observations: np.ndarray,) -> jnp.ndarray:
    q_value = q_fn({'params': q_params}, observations)
    return q_value


class QNetwork(nn.Module):
    """
    Action-Value (Q-Value) network for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    """

    features_extractor: nn.Module
    action_space: gym.spaces.Discrete
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, **kwargs) -> jnp.ndarray:
        features = self.features_extractor(observations, **kwargs)
        action_dim = self.action_space.n
        q_net = create_mlp(action_dim, self.net_arch, self.activation_fn,)(features)
        return q_net


class DQNPolicy(object):
    """
    Policy class with Q-Value Net and target net for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        key: Any,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 10,
    ):

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        if features_extractor_kwargs is None:
            self.features_extractor_kwargs = {}
        self.observation_space = observation_space
        self.action_space = action_space
        self.activation_fn = activation_fn
        self.rng = key
        self.max_grad_norm = max_grad_norm

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        Put the target network into evaluation mode.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.rng, q_key, feature_key = jax.random.split(self.rng, 3)
        features_extractor_def = self.features_extractor_class(_observation_space=self.observation_space, **self.features_extractor_kwargs)
        qnet_def = QNetwork(features_extractor=features_extractor_def, net_arch=self.net_arch,
                            activation_fn=self.activation_fn, action_space=self.action_space)

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=lr_schedule)
        )

        if isinstance(self.observation_space, gym.spaces.Dict):
            observation = self.observation_space.sample()
            for key, _ in self.observation_space.spaces.items():
                observation[key] = np.expand_dims(observation[key], axis=0)
        else:
            observation = np.expand_dims(self.observation_space.sample(), axis=0)

        self.q_net = Model.create(qnet_def, inputs=[q_key, observation], tx=optimizer)
        self.q_net_target = Model.create(qnet_def, inputs=[q_key, observation])

    def forward(self, obs: np.array, deterministic: bool = True) -> np.array:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: np.array, deterministic: bool = True, mask: np.ndarray = None, **kwargs) -> np.array:
        q_value = sample_actions(self.q_net.apply_fn, self.q_net.params, obs)
        if mask is not None:
            q_value = jnp.clip(q_value, a_max=1e+5) - mask * 1e+6
        return np.array(q_value.argmax(axis=1).reshape(-1))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        mask: np.ndarray = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self._predict(observation, deterministic, mask), None

MlpPolicy = DQNPolicy


class CnnPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        key: Any,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 10,
    ):
        super().__init__(
            key,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            max_grad_norm,
        )


class MultiInputPolicy(DQNPolicy):
    """
    Policy class for DQN when using dict observations as input.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        key: Any,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 10,
    ):
        super().__init__(
            key,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            max_grad_norm,
        )