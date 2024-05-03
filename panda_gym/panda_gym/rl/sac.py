import gym
import sys
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

sys.path.append('/home/ros2/panda_grasping_ws/src/panda_grasping')
from panda_gym.panda_gym.envs.panda.panda_grasping_env import PandaGraspingEnv

# env = PandaGraspingEnv()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_mean = self.mean(x)
        action_std = torch.exp(self.log_std)
        return action_mean, action_std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.q_value(x)
        return q_value

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

if __name__ == "__main__":
    env = PandaGraspingEnv()
    state_dim = 14
    action_dim = 8
    hidden_size = 64
    lr = 0.001
    tau = 0.005
    gamma = 0.99
    done = False

    actor = Actor(state_dim, action_dim, hidden_size)
    critic1 = Critic(state_dim, action_dim, hidden_size)
    critic2 = Critic(state_dim, action_dim, hidden_size)

    actor_target = Actor(state_dim, action_dim, hidden_size)
    critic1_target = Critic(state_dim, action_dim, hidden_size)
    critic2_target = Critic(state_dim, action_dim, hidden_size)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=lr)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=lr)

    replay_buffer = deque(maxlen=100)

    num_episodes = 20
    max_steps_per_episodes = 5
    print("Actor", actor)
    print("Critic1", critic1)
    print("Critic2", critic2)
    
    low_action_space = torch.tensor(env.action_space.low, dtype=torch.float32)
    high_action_space = torch.tensor(env.action_space.high, dtype=torch.float32)

    for episode in range(num_episodes):
        print(f"=== Episode {episode} ====")
        state = env.reset()
        print("STATE", state)
        state = np.concatenate(state)
        print("State", state)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        for t in range(max_steps_per_episodes):
            action_mean, action_std = actor(state_tensor)
            print(f"==== Step {t} ======")
            print("Action_mean", action_mean)
            print("action_std", action_std)
            action = torch.normal(action_mean, action_std)
            print("Action", action)
            action = torch.clamp(action, low_action_space, high_action_space)
            print("clamped action", action)

            next_state, reward, done = env.step(action)
            next_state = np.concatenate(next_state)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            q1 = critic1(state_tensor, action)
            q2 = critic2(state_tensor, action)

            target_q = reward + gamma * torch.min(critic1_target(next_state_tensor, actor_target(next_state_tensor)),
                                                    critic2_target(next_state_tensor, actor_target(next_state_tensor)))
            
            actor_loss = -torch.mean(q1)

            critic1_loss = nn.MSELoss()(q1, target_q.detach())
            critic2_loss = nn.MSELoss()(q2, target_q.detach())

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic1_optimizer.zero_grad()
            critic1_loss.backward()
            critic1_optimizer.step()

            critic2_optimizer.zero_grad()
            critic2_loss.backward()
            critic2_optimizer.step()

            soft_update(actor_target, actor, tau)
            soft_update(critic1_target, critic1, tau)
            soft_update(critic2_target, critic2, tau)

            state_tensor = next_state_tensor

            if done:
                break

    torch.save(actor.state_dict(), 'sac_actor.pt')




