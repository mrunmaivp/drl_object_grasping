import gymnasium as gym
import sys
from stable_baselines3 import SAC, TD3
import datetime
import gym_panda
from stable_baselines3.common.buffers import ReplayBuffer

env = gym.make('PandaGraspingEnv-v0')

loaded_model = SAC.load("panda_grasping_model_sac_1711", env=env)
loaded_model.load_replay_buffer("panda_grasping_replay_buffer_sac_1711")
print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

print("Loaded ReplayBuffer")
loaded_model.ent_coef = 'auto_0.2'
loaded_model.seed = 12345
loaded_model.tensorboard_log = "./sac_panda_grasping_18112023_12:15"

loaded_model.learn(total_timesteps=30000, progress_bar=True, tb_log_name="first_run")

loaded_model.save("panda_grasping_model_sac_1811_object_position")

loaded_model.save_replay_buffer("panda_grasping_model_sac_1811_object_position")