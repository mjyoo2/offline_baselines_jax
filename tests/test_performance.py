from offline_baselines_jax import SAC, TD3
from offline_baselines_jax.sac.policies import SACPolicy
from offline_baselines_jax.td3.policies import TD3Policy

import gym

train_env = gym.make('HalfCheetah-v2')
model = TD3(TD3Policy, train_env, seed=777, verbose=1, batch_size=1024, buffer_size=50000)
model.learn(total_timesteps=100000, log_interval=10)
model = SAC(SACPolicy, train_env, seed=777, verbose=1, batch_size=1024, buffer_size=50000)
model.learn(total_timesteps=100000, log_interval=10)

from stable_baselines3 import SAC, TD3

model = SAC('MlpPolicy', train_env, verbose=1, batch_size=1024, buffer_size=50000)
model.learn(total_timesteps=100000, log_interval=10)

model = TD3('MlpPolicy', train_env, verbose=1, batch_size=1024, buffer_size=50000)
model.learn(total_timesteps=100000, log_interval=10)