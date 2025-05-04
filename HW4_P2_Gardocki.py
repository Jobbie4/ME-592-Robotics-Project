# -*- coding: utf-8 -*-
"""
Created on Sat May  3 21:28:01 2025

@author: markg
"""

#ME5920 hw4
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from gym.wrappers import RecordVideo

#parameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
EPISODES = 500

#neural network
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

#replay
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

#training
def train(env):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(obs_size, n_actions)
    target_net = DQN(obs_size, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START
    all_rewards = []
    all_losses = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = []

        for t in range(env.spec.max_episode_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_v = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_v)
                    action = q_values.max(1)[1].item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            #training
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))

                states = torch.FloatTensor(batch[0])
                actions = torch.LongTensor(batch[1]).unsqueeze(1)
                rewards = torch.FloatTensor(batch[2])
                next_states = torch.FloatTensor(batch[3])
                dones = torch.BoolTensor(batch[4])

                q_values = policy_net(states).gather(1, actions).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                next_q_values[dones] = 0.0
                expected_q_values = rewards + GAMMA * next_q_values

                loss = nn.MSELoss()(q_values, expected_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss.append(loss.item())

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        all_rewards.append(total_reward)
        all_losses.append(avg_loss)
        

        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return policy_net, all_rewards, all_losses

def record_agent(model, video_path="cartpole_video"):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_path, name_prefix="dqn_cartpole")
    state, _ = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            state_v = torch.FloatTensor(state).unsqueeze(0)
            action = model(state_v).max(1)[1].item()

        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print(f"Video saved to: {video_path}/")

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    trained_model, rewards, losses = train(env)
    
    record_agent(trained_model)

    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Average Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()
