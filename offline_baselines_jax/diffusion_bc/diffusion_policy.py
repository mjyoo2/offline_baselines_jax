from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import jax
import gym
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import optax
import functools

from offline_baselines_jax.common.jax_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
    default_init,
)

from offline_baselines_jax.diffusion_bc.ddpm_schedule import DiffusionBetaScheduler, DDPMCoefficients
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import Schedule


@functools.partial(jax.jit, static_argnames=('noise_dim', 'total_denoise_steps'))
def diffusion_infer(_rng: Any, _actor: Model, _y_t: jnp.ndarray, _observation: jnp.ndarray, noise_dim: int, total_denoise_steps: int, oneover_sqrta:jnp.ndarray,
                    ma_over_sqrtmab_inv: jnp.ndarray, sqrt_beta_t:jnp.ndarray):
    broadcast_shape = _observation.shape[: -1]

    # denoising chain
    for t in range(total_denoise_steps, 0, -1):
        rng, _ = jax.random.split(_rng)
        denoise_step = t + jnp.zeros(shape=broadcast_shape, dtype="i4")[..., jnp.newaxis]
        z = jax.random.normal(rng, shape=(*_observation.shape[: -1], noise_dim)) if t > 0 else 0

        # z = jax.random.normal(self.rng, shape=(*observation.shape[: -1], self.noise_dim))
        pred = _actor(
            x=_observation,
            y=_y_t,
            t=denoise_step,
        )
        eps = pred['pred']

        _y_t = oneover_sqrta[t] * (_y_t - ma_over_sqrtmab_inv[t] * eps) + (sqrt_beta_t[t] * z)
        _y_t = jnp.clip(_y_t, -2.0, 2.0)
    return _y_t

class Actor(nn.Module):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    features_extractor: nn.Module
    action_space: gym.spaces.Space
    net_arch: List[int]
    total_denoise_steps: int
    activation_fn: Type[nn.Module] = nn.relu
    embed_dim: int = 128
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        :param x: observation   [b, l, d]
        :param y: action    [b, l, d]
        :param t: denoising step    [b, l]
        :param deterministic
        :return:
        """
        t = t / self.total_denoise_steps

        emb_x = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.embed_dim, ],
            activation_fn=self.activation_fn,
            layer_norm=True)(x)
        emb_y = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.embed_dim, ],
            activation_fn=self.activation_fn,
            layer_norm=True)(y)
        emb_t = nn.Sequential([
            nn.Dense(self.embed_dim),
            jnp.sin,
            nn.Dense(self.embed_dim)
        ])(t)

        in1 = jnp.concatenate((emb_y, emb_x, emb_t), axis=-1)
        out1 =create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.embed_dim, ],
            activation_fn=nn.gelu,
            layer_norm=True)(in1)

        in2 = jnp.concatenate((out1 / 1.414, emb_y, emb_t), axis=-1)
        out2 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.embed_dim, ],
            activation_fn=nn.gelu,
            layer_norm=True)(in2)
        out2 = out2 + out1 / 1.414

        in3 = jnp.concatenate((out2 / 1.414, emb_y, emb_t), axis=-1)
        out3 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.embed_dim, ],
            activation_fn=nn.gelu,
            layer_norm=True)(in3)
        out3 = out3 + out2 / 1.414

        in4 = jnp.concatenate((out3, emb_y, emb_t), axis=-1)
        out4 = create_mlp(
            output_dim=self.action_space.shape[0],
            net_arch=[self.embed_dim, ],
            activation_fn=nn.gelu,
            layer_norm=True)(in4)

        return {"pred": out4, "emb_x": emb_x, "emb_y": emb_y, "emb_t": emb_t}

class DiffusionPolicy(object):
    def __init__(
            self,
            key,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.relu,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            ddpm_schedule_kwargs: Optional[Dict[str, Any]] = None,
            total_denoise_steps: int = 8
    ):

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        if ddpm_schedule_kwargs is None:
            ddpm_schedule_kwargs = {}

        self.rng, actor_key, critic_key, features_key = jax.random.split(key, 4)
        self.observation_space = observation_space
        self.action_space = action_space
        self.total_denoise_steps = total_denoise_steps
        self.noise_dim = self.action_space.shape[0]
        self.ddpm_scheduler = DiffusionBetaScheduler(**ddpm_schedule_kwargs)
        self.ddpm_schedule = self.ddpm_scheduler.schedule()

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        features_extractor_def = features_extractor_class(_observation_space=observation_space, **features_extractor_kwargs)
        actor_def = Actor(features_extractor=features_extractor_def, action_space=action_space,
                          net_arch=actor_arch, activation_fn=activation_fn, total_denoise_steps=self.total_denoise_steps)

        if isinstance(observation_space, gym.spaces.Dict):
            observation = observation_space.sample()
            for key, _ in observation_space.spaces.items():
                observation[key] = np.expand_dims(observation[key], axis=0)
        else:
            observation = np.expand_dims(observation_space.sample(), axis=0)

        action = np.expand_dims(action_space.sample(), axis=0)
        denoise_step = np.zeros(shape=(1, 1))

        actor = Model.create(actor_def, inputs=[actor_key, observation, action, denoise_step], tx=optax.adam(learning_rate=lr_schedule))
        self.actor = actor

    def _predict(self, observation: jnp.ndarray):
        batch_size = observation.shape[0]
        y_t = jax.random.normal(self.rng, shape=(batch_size, self.noise_dim))
        oneover_sqrta = self.ddpm_schedule.oneover_sqrta
        ma_over_sqrtmab_inv = self.ddpm_schedule.ma_over_sqrtmab_inv
        sqrt_beta_t = self.ddpm_schedule.sqrt_beta_t
        y_t = diffusion_infer(self.rng, self.actor, y_t, observation, self.noise_dim, self.total_denoise_steps, oneover_sqrta, ma_over_sqrtmab_inv, sqrt_beta_t)
        return y_t

    def predict(self, observation: jnp.ndarray, deterministic: bool = False) -> np.ndarray:
        actions = self._predict(observation)
        if isinstance(self.action_space, gym.spaces.Box):
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            actions = self.unscale_action(actions)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            actions = np.argmax(actions, axis=-1)
        return actions, None

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
