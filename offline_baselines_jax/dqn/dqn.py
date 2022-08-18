import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import copy
import jax
import optax
import functools

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.preprocessing import maybe_transpose
from offline_baselines_jax.common.type_aliases import GymEnv, MaybeCallback, Schedule, ReplayBufferSamples, Params, InfoDict
from offline_baselines_jax.common.utils import get_linear_fn, is_vectorized_observation
from offline_baselines_jax.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise


def sample(action_space: gym.spaces.Discrete, mask: Optional[np.ndarray] = None) -> int:
    """Generates a single random sample from this space.
    A sample will be chosen uniformly at random with the mask if provided
    Args:
        mask: An optional mask for if an action can be selected.
            Expected `np.ndarray` of shape `(n,)` and dtype `np.int8` where `1` represents valid actions and `0` invalid / infeasible actions.
            If there are no possible actions (i.e. `np.all(mask == 0)`) then `space.start` will be returned.
    Returns:
        A sampled integer from the space
    """
    if mask is not None:
        assert isinstance(
            mask, np.ndarray
        ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
        assert (
                mask.dtype == np.int8
        ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
        assert mask.shape == (
            action_space.n,
        ), f"The expected shape of the mask is {(action_space.n,)}, actual shape: {mask.shape}"
        valid_action_mask = mask == 1
        assert np.all(
            np.logical_or(mask == 0, valid_action_mask)
        ), f"All values of a mask should be 0 or 1, actual values: {mask}"
        if np.any(valid_action_mask):
            return int(
                action_space.np_random.choice(np.where(valid_action_mask)[0])
            )
        else:
            return 0

    return int(action_space.np_random.integers(action_space.n))


@functools.partial(jax.jit, static_argnames=('tau',))
def polyak_update(agent: Model, target: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), agent.params, target.params)
    return target.replace(params=new_target_params)


@functools.partial(jax.jit, static_argnames=('gamma',))
def _update_jit(key: Any, q_net:Model, q_net_target:Model, replay_data: ReplayBufferSamples, gamma:float, sm_critic: Model, sm_actor:Model, sm_obs):
    next_q_values = q_net_target(replay_data.next_observations) * (1 - replay_data.observations['mask']) - replay_data.observations['mask'] * 1e+12
    next_q_values = next_q_values.max(axis=1)
    next_q_values = next_q_values.reshape(-1, 1)

    real_action = sm_actor(sm_obs).sample(seed=key)
    action_q_values = sm_critic(sm_obs, real_action)
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values + replay_data.dones * action_q_values

    def q_loss_fn(q_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates
        current_q_values = q_net.apply_fn({'params': q_params}, replay_data.observations)
        action_selection = jax.nn.one_hot(replay_data.actions, current_q_values.shape[-1]).squeeze(-2)
        # Retrieve the q-values for the actions from the replay buffer
        current_q_values = jnp.sum(current_q_values * action_selection, axis=-1, keepdims=True)
        # Compute Huber loss (less sensitive to outliers)
        loss = optax.huber_loss(current_q_values, target_q_values).mean()
        return loss, {'current_q_value': current_q_values.mean(), 'loss': loss}

    new_q_fn, info = q_net.apply_gradient(q_loss_fn)
    return new_q_fn, info


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)
    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, object] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.,
        exploration_final_eps: float = 0.1,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = 777,
        _init_setup_model: bool = True,
        sm_model = None,
    ):
        if policy_kwargs is None:
            policy_kwargs = {'max_grad_norm': max_grad_norm}
        else:
            policy_kwargs['max_grad_norm'] = max_grad_norm

        super().__init__(
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
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
            support_multi_env=True,
            without_exploration=False
        )
        self.soft_modularization_agent = None
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None
        self.sm_model = sm_model
        self.rng = jax.random.PRNGKey(seed)
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            self.policy.q_net_target = polyak_update(self.q_net, self.q_net_target, self.tau)

        self._create_aliases()
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            sm_obs = copy.deepcopy(replay_data.observations)
            del sm_obs['mask']
            del sm_obs['num_resource']
            self.rng, key = jax.random.split(self.rng, 2)
            new_q_fn, info = _update_jit(key, self.q_net, self.q_net_target, replay_data, self.gamma, self.sm_model.critic, self.sm_model.actor, sm_obs)
            self.policy.q_net = new_q_fn

            self._create_aliases()
            losses.append(info['loss'])

        # Increase update counter
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def set_model(self, model):
        self.soft_modularization_agent = model

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if observation['mask'].shape[0] != 1 and len(observation['mask'].shape) != 1:
            raise NotImplementedError()
        else:
            mask = np.copy(observation['mask']).flatten()

        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([sample(self.action_space, mask=np.asarray(1-mask,dtype=np.int8)) for _ in range(n_batch)])
            else:
                action = np.array(sample(self.action_space, mask=np.asarray(1-mask,dtype=np.int8)))
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic, mask=mask)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:
        print(id(callback.sm_model))
        print(id(self))
        return super().learn(
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
        return super(DQN, self)._excluded_save_params() + ['q_net', 'q_net_target']

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        params_dict['q_net'] = self.q_net.params
        params_dict['q_net_target'] = self.q_net_target.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['q_net', 'q_net_target']

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy

        unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        buffer_action = unscaled_action
        action = buffer_action
        return action, buffer_action