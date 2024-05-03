import datetime
import gym
import numpy as np
import itertools
import torch
import sys
from soft_actor_critic import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import multiprocessing


sys.path.append('/home/ros2/mt_panda_grasping_ws/src/panda_grasping')
from panda_gym.panda_gym.envs.panda.panda_env import PandaEnv
from panda_gym.panda_gym.envs.panda.env_object_grasp import ObjectGraspEnv

GAMMA = 0.99
TAU = 0.005
LR = 0.0003
ALPHA = 0.2
POLICY = "Gaussian"
AUTOMATIC_ENTROPY_TUNING = False
TARGET_UPDATE_INTERVAL = 1
HIDDEN_SIZE = 256
SEED = None
REPLAY_SIZE = 28000
START_STEPS = 0
BATCH_SIZE = 256
UPDATES_PER_STEP = 1
NUM_STEPS = 30000
EVAL = False
REWARD_THRESHOLD = 50
MAX_CONSECUTIVE_SUCCESS = 10

class Args():
    def __init__(self):
        self.gamma = GAMMA 
        self.tau = TAU 
        self.alpha = ALPHA 
        self.policy = POLICY 
        self.target_update_interval = TARGET_UPDATE_INTERVAL 
        self.automatic_entropy_tuning = AUTOMATIC_ENTROPY_TUNING
        self.hidden_size = HIDDEN_SIZE 
        self.lr = LR 
        self.replay_size = REPLAY_SIZE 
        self.seed = SEED
        self.start_steps = START_STEPS
        self.batch_size = BATCH_SIZE
        self.updates_per_step = UPDATES_PER_STEP
        self.num_steps = NUM_STEPS
        self.eval = EVAL

def state_to_tensor(state):
    state = np.concatenate(state)
    state_tensor = torch.tensor(state, dtype=torch.float32)
    return state_tensor

def train():
    args = Args()
    env_pregrasp = PandaEnv()
    env_grasp = ObjectGraspEnv()
    # env.seed(123456)
    # env.action_space.seed(123456)
    # torch.manual_seed(123456)
    # np.random.seed(123456)
    print("Set Seed")

    # Agent
    agent_pregrasp = SAC(env_pregrasp.observation_space.shape[0], env_pregrasp.action_space, args)
    agent_grasp = SAC(env_grasp.observation_space.shape[0], env_grasp.action_space, args)

    agent_pregrasp.load_checkpoint(ckpt_path="/home/ros2/mt_panda_grasping_ws/src/panda_grasping/panda_gym/panda_gym/rl/checkpoints/sac_checkpoint_PandaEnv_2023-12-17_11-40-23", evaluate=True)
    agent_grasp.load_checkpoint(ckpt_path="/home/ros2/mt_panda_grasping_ws/src/panda_grasping/panda_gym/panda_gym/rl/checkpoints/sac_checkpoint_PandaGrasping_2023-12-04_14-25-41", evaluate=True)

    # #Tesnorboard
    writer = SummaryWriter('runs/evaluate{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))


    print(" ===================== EVALUATION =========================")
    avg_overall_reward = 0
    avg_reward_pregrasp = 0
    avg_reward_grasp = 0
    episodes = 100
    for i_episode in range(episodes):
        max_timesteps_per_episode_pregrasp = 0
        state_pregrasp = env_pregrasp.reset()
        episode_reward_pregrasp = 0
        done_pregrasp = False
        while not done_pregrasp and max_timesteps_per_episode_pregrasp < 200:
            action_pregrasp = agent_pregrasp.select_action(state_pregrasp, evaluate=True)
            print("Evalution", action_pregrasp)
            next_state_pregrasp, reward_pregrasp, done_pregrasp = env_pregrasp.step(action_pregrasp)
            episode_reward_pregrasp += reward_pregrasp
            max_timesteps_per_episode_pregrasp += 1

            state_pregrasp = next_state_pregrasp
        writer.add_scalar('reward_pregrasp/evaluate', episode_reward_pregrasp, i_episode)
        avg_reward_pregrasp += episode_reward_pregrasp

        if episode_reward_pregrasp > 50:

            print(" ----------------- GRASPING ---------------------------")
            max_timesteps_per_episode_grasp = 0
            state_grasp = env_grasp.reset()
            episode_reward_grasp = 0
            done_grasp = False
            while not done_grasp and max_timesteps_per_episode_grasp < 100:
                action_grasp = agent_grasp.select_action(state_grasp, evaluate=True)
                next_state_grasp, reward_grasp, done_grasp = env_grasp.step(action_grasp)
                episode_reward_grasp += reward_grasp
                max_timesteps_per_episode_grasp += 1

                state_grasp = next_state_grasp

        else:
            episode_reward_grasp = 0
        writer.add_scalar('reward_grasp/evaluate', episode_reward_grasp, i_episode)    
        
        avg_reward_pregrasp += episode_reward_pregrasp
        avg_reward = episode_reward_pregrasp + episode_reward_grasp
        writer.add_scalar('reward/evaluate', avg_reward, i_episode)   
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
    
    env.close()

if __name__ == "__main__":
    train()