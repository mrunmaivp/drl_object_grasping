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


sys.path.append('/home/ros2/panda_grasping_ws/src/panda_grasping')
from panda_gym.panda_gym.envs.panda.panda_env import PandaEnv
from panda_gym.panda_gym.envs.panda.object_grasping_env import ObjectGraspingEnv

GAMMA = 0.99
TAU = 0.005
LR = 0.0003
ALPHA = 0.2
POLICY = "Gaussian"
AUTOMATIC_ENTROPY_TUNING = False
TARGET_UPDATE_INTERVAL = 1
HIDDEN_SIZE = 256
SEED = 123456
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
    env1 = PandaEnv()
    env2 = ObjectGraspingEnv()
    torch.manual_seed(123456)
    np.random.seed(123456)
    print("Set Seed")

    # Agent
    agent1 = SAC(env1.observation_space.shape[0], env1.action_space, args)
    agent2 = SAC(env2.observation_space.shape[0], env2.action_space, args)

    agent1.load_checkpoint(ckpt_path="/home/ros2/panda_grasping_ws/src/panda_grasping/panda_gym/panda_gym/rl/checkpoints/sac_checkpoint_PandaGraspingEnv_2023-10-22_10-05-58", evaluate=True)
    agent2.load_checkpoint(ckpt_path="/home/ros2/panda_grasping_ws/src/panda_grasping/panda_gym/panda_gym/rl/checkpoints/sac_checkpoint_GraspingEnv_2023-11-03_15-43-14", evaluate=True)

    # #Tesnorboard
    writer = SummaryWriter('runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory2 = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    consecutive_success = 0
    total_numsteps2 = 0
    updates2 = 0
    consecutive_success2 = 0

    for i_episode in itertools.count(1):
        print(f"====Episode {i_episode}====")
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env1.reset()
        max_timesteps_per_episode = 0

        while not done and max_timesteps_per_episode < 200:
            print(f"==== Timestep {total_numsteps} and Episode {i_episode} ====")
            if args.start_steps > total_numsteps:
                action = env1.action_space.sample()  # Sample random action
                print(" +++++++ Random Action Selected ++++++ ", action)
            else:
                action = agent1.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent1.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            max_timesteps_per_episode += 1
            next_state, reward, done = env1.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == 10 else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        print(f"====Episode {i_episode}====")
        episode_reward2 = 0
        episode_steps2 = 0
        done2 = False
        state2 = env2.reset()
        max_timesteps_per_episode2 = 0

        while not done2 and max_timesteps_per_episode2 < 200:
            print(f"==== Timestep {total_numsteps2} and Episode {i_episode} ====")
            if args.start_steps > total_numsteps2:
                action2 = env2.action_space.sample()  # Sample random action
                print(" +++++++ Random Action Selected ++++++ ", action2)
            else:
                action2 = agent2.select_action(state2)  # Sample action from policy

            if len(memory2) > args.batch_size:
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent2.update_parameters(memory, args.batch_size, updates2)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates2)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates2)
                    writer.add_scalar('loss/policy', policy_loss, updates2)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates2)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates2)
                    updates2 += 1
            max_timesteps_per_episode2 += 1
            next_state2, reward2, done2 = env2.step(action2) # Step
            episode_steps2 += 1
            total_numsteps2 += 1
            episode_reward2 += reward

            mask2 = 1 if episode_steps2 == 10 else float(not done2)

            memory2.push(state2, action2, reward2, next_state2, mask2) # Append transition to memory

            state2 = next_state2

        if episode_reward >= REWARD_THRESHOLD:
            consecutive_success += 1
        else:
            consecutive_success = 0
            
        if consecutive_success >= MAX_CONSECUTIVE_SUCCESS:
            print("Starting training with next domain randomization")

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        print(" ===================== EVALUATION =========================")
        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                max_timesteps_per_episode = 0
                state = env.reset()
                episode_reward = 0
                done = False
                while not done and max_timesteps_per_episode < 200:
                    action = agent.select_action(state, evaluate=True)
                    print("Evalution", action)
                    next_state, reward, done = env.step(action)
                    episode_reward += reward
                    max_timesteps_per_episode += 1

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
        
    agent.save_checkpoint("PandaGraspingEnv", suffix=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    env.close()

if __name__ == "__main__":
    train()