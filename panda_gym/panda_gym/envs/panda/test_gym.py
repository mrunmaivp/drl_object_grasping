from panda_grasping_env import PandaGraspingEnv
import gym
import numpy as np
# from stable_baselines3 import SAC
import pdb

n_episodes = 20

# env = gym.make('PandaGraspingEnv-v0')

env = PandaGraspingEnv()

obs = env.reset()
done = False


for i in range(n_episodes):
    # pdb.set_trace()
    print("==============EPISODE==============", i)
    action_sample = env.action_space.sample()
    print("Action Sample", action_sample)
    observation, reward, done = env.step(action_sample)
    print("=================== STEP COMPLETE ========================")
    print("OBSERVATION", observation)
    print("REWARD", reward)
    print("DONE", done)
    
