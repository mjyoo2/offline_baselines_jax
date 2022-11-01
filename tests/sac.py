import numpy as np
from offline_baselines_jax import SAC, TD3
from offline_baselines_jax.sac.policies import SACPolicy
from offline_baselines_jax.td3.policies import TD3Policy

import gym

train_env = gym.make('LunarLanderContinuous-v2')
model = SAC(SACPolicy, train_env, seed=777, verbose=1, batch_size=256, buffer_size=1000000)
model.learn(total_timesteps=3000000, log_interval=10)