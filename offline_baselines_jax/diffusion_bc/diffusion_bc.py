from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax
import functools
import copy
from dataclasses import asdict

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.buffers import BCReplayBuffer
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule, InfoDict, ReplayBufferSamples, Params
from offline_baselines_jax.diffusion_bc.diffusion_policy import DiffusionPolicy
from offline_baselines_jax.diffusion_bc.ddpm_schedule import DDPMCoefficients

@functools.partial(jax.jit, static_argnames=("noise_dim", "total_denoise_steps"))
def update_diffusion(
        rng: Any,
        policy: Model,
        observations: jnp.ndarray,  # [b, l, d]
        actions: jnp.ndarray,  # [b, l, d]

        ddpm_schedule: Dict,
        noise_dim: int,
        total_denoise_steps: int,
) -> Tuple[Model, Dict]:
    rng_list = jax.random.split(rng, 2)
    batch_size = observations.shape[0]
    # subseq_len = observations.shape[1]

    # sample noise
    noise = jax.random.normal(rng_list[0], shape=(batch_size, noise_dim))  # [b, l, d]
    # noise = jnp.repeat(noise[:, jnp.newaxis, ...], repeats=subseq_len, axis=1)  # [b, l, d]

    # add noise to clean target actions
    _ts = jax.random.randint(rng_list[1], shape=(batch_size, 1), minval=1, maxval=total_denoise_steps + 1)

    sqrtab = ddpm_schedule["sqrtab"][_ts]  # [b, 1]
    sqrtmab = ddpm_schedule["sqrtmab"][_ts]  # [b, 1]

    y_t = sqrtab * actions + sqrtmab * noise

    # use diffusion model to predict noise
    def policy_loss(actor_params: Params) -> Tuple[jnp.ndarray, Dict]:
        pred = policy.apply_fn(
            {'params': actor_params},
            x=observations,
            y=y_t,
            t=_ts,  # [b, l]
        )
        noise_pred = pred["pred"]

        noise_pred = noise_pred.reshape(-1, noise_dim)
        noise_targ = noise.reshape(-1, noise_dim)

        mse_loss = jnp.sum(jnp.mean((noise_pred - noise_targ) ** 2, axis=-1))

        # If key startswith __ (double underbar), then it is not printed.
        _info = {
            "actor_loss": mse_loss,
            "__noise_pred": noise_pred,
            "__noise_targ": noise_targ,
            "__ts": _ts,
            "__sqrtab": sqrtab,
            "__sqrtmab": sqrtmab,
        }

        return mse_loss, _info

    new_policy, info = policy.apply_gradient(policy_loss)
    return new_policy, info

def _update_jit(
        rng: int,
        actor: Model,
        replay_data: ReplayBufferSamples,
        ddpm_schedule:DDPMCoefficients,
        total_denoise_steps: int,
        noise_dim: int,
) -> Tuple[int, Model, InfoDict]:
    rng, key = jax.random.split(rng, 2)
    new_actor, actor_info = update_diffusion(
        rng=key,
        policy=actor,
        observations=replay_data.observations,
        actions=replay_data.actions,
        ddpm_schedule=asdict(ddpm_schedule),
        noise_dim=noise_dim,
        total_denoise_steps=total_denoise_steps
    )
    return rng, new_actor, {**actor_info}

class DiffusionBC(OffPolicyAlgorithm):
    """
    Behavior Cloning (BC) Algorithm.
    It is not reinforcement learning algorithm.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[DiffusionPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, 'episode'),
        gradient_steps: int = -1,
        replay_buffer_class: Optional[BCReplayBuffer] = BCReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 0,
        _init_setup_model: bool = True,
        without_exploration: bool = False,
    ):
        super(DiffusionBC, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box, gym.spaces.Discrete),
            support_multi_env=True,
            without_exploration=without_exploration,
        )
        if _init_setup_model:
            self._setup_model()
        self.total_denoise_steps = self.policy.total_denoise_steps
        self.noise_dim = self.policy.noise_dim

    def _setup_model(self) -> None:
        super(DiffusionBC, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor

    def _load_policy(self, only_actor=False) -> None:
        self.policy.actor = self.actor

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        actor_losses= []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.key, new_actor, info = _update_jit(
                self.key,
                self.actor,
                replay_data,
                self.policy.ddpm_schedule,
                self.total_denoise_steps,
                self.noise_dim,
            )
            self.policy.actor = new_actor

            self._create_aliases()
            actor_losses.append(copy.deepcopy(info['actor_loss']))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "BC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(DiffusionBC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(DiffusionBC, self)._excluded_save_params() + ["actor"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        params_dict['actor'] = self.actor.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['actor']