# -*- coding: utf-8 -*-
"""
Created on Sat May  3 23:43:27 2025

@author: markg
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy.signal

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size  # buffer has to have room
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        
    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        
    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full
        self.ptr, self.path_start_idx = 0, 0
        
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in data.items()}

class PPOAgent:
    def __init__(self, observation_space, action_space,
                 clip_ratio=0.2, policy_lr=3e-4, vf_lr=1e-3,
                 train_policy_iters=80, train_vf_iters=80,
                 target_kl=0.01):
        
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n
        
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_policy_iters = train_policy_iters
        self.train_vf_iters = train_vf_iters
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = Adam(learning_rate=policy_lr)
        self.critic_optimizer = Adam(learning_rate=vf_lr)
        
    def _build_actor(self):
        inputs = Input(shape=(self.obs_dim,))
        x = Dense(64, activation='tanh')(inputs)
        x = Dense(64, activation='tanh')(x)
        outputs = Dense(self.act_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_critic(self):
        inputs = Input(shape=(self.obs_dim,))
        x = Dense(64, activation='tanh')(inputs)
        x = Dense(64, activation='tanh')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def get_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[np.newaxis, :]
            
        action_probs = self.actor(obs).numpy()[0]
        
        if deterministic:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(self.act_dim, p=action_probs)
            
        logp = np.log(action_probs[action] + 1e-10)
        
        value = self.critic(obs).numpy()[0, 0]
        
        return action, value, logp
    
    def train_one_epoch(self, data):
        obs = data['obs']
        act = tf.cast(data['act'], tf.int32)
        adv = data['adv']
        ret = data['ret']
        old_logp = data['logp']
        
        kl_sum = 0
        for _ in range(self.train_policy_iters):
            with tf.GradientTape() as tape:
                action_probs = self.actor(obs)
                
                logp_all = tf.math.log(action_probs + 1e-10)
                
                indices = tf.stack([tf.range(act.shape[0], dtype=tf.int32), act], axis=1)
                logp = tf.gather_nd(logp_all, indices)
                
                ratio = tf.exp(logp - old_logp)
                
                clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                loss_pi = -tf.reduce_mean(tf.minimum(ratio * adv, clip_adv))
                
            grads = tape.gradient(loss_pi, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            
            action_probs_new = self.actor(obs)
            kl = tf.reduce_mean(
                tf.reduce_sum(
                    action_probs * tf.math.log(action_probs / (action_probs_new + 1e-10) + 1e-10),
                    axis=1
                )
            )
            kl_sum += kl.numpy()
            
            if kl.numpy() > 1.5 * self.target_kl:
                break
        
        for _ in range(self.train_vf_iters):
            with tf.GradientTape() as tape:
                v = self.critic(obs)
                loss_v = tf.reduce_mean((v - ret) ** 2)
                
            grads = tape.gradient(loss_v, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        
        return kl_sum / (self.train_policy_iters)
    
    def save_models(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
    
    def load_models(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

def train_ppo(env_name="LunarLander-v3", total_steps=1_000_000, 
              steps_per_epoch=4000, gamma=0.99, lam=0.95,
              clip_ratio=0.2, policy_lr=3e-4, vf_lr=1e-3, 
              train_policy_iters=80, train_vf_iters=80, target_kl=0.01):

    env = gym.make(env_name)
    agent = PPOAgent(
        env.observation_space, env.action_space,
        clip_ratio=clip_ratio, policy_lr=policy_lr, vf_lr=vf_lr,
        train_policy_iters=train_policy_iters, train_vf_iters=train_vf_iters,
        target_kl=target_kl
    )
    
    buffer = PPOBuffer(
        env.observation_space.shape[0], env.action_space.n,
        steps_per_epoch, gamma, lam
    )
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    epochs = total_steps // steps_per_epoch
    
    best_avg_reward = -np.inf
    
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            action, value, logp = agent.get_action(obs)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            buffer.store(obs, action, reward, value, logp)
            
            obs = next_obs
            
            done = terminated or truncated
            if done or (t == steps_per_epoch - 1):
                if done:
                    last_val = 0
                else:
                    _, last_val, _ = agent.get_action(obs)
                    
                buffer.finish_path(last_val)
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    window_size = min(100, len(episode_rewards))
                    avg_reward = np.mean(episode_rewards[-window_size:])
                    avg_rewards.append(avg_reward)
                    
                    print(f"Epoch: {epoch + 1}/{epochs}, Episode: {len(episode_rewards)}, "
                          f"Reward: {episode_reward:.2f}, Length: {episode_length}, "
                          f"Avg Reward (last 100): {avg_reward:.2f}")
                    
                    if avg_reward > best_avg_reward and len(episode_rewards) >= 10:
                        best_avg_reward = avg_reward
                        agent.save_models("ppo_lunar_actor_best.weights.h5", "ppo_lunar_critic_best.weights.h5")
                        print(f"New best model saved with avg reward: {best_avg_reward:.2f}")
                    
                    if avg_reward >= 200.0 and len(episode_rewards) >= 100:
                        print(f"Environment solved with average reward {avg_reward:.2f}")
                        agent.save_models("ppo_lunar_actor_solved.weights.h5", "ppo_lunar_critic_solved.weights.h5")
                        return episode_rewards, episode_lengths, avg_rewards, agent
                    
                    obs, _ = env.reset()
                    episode_reward = 0
                    episode_length = 0
        
        data = buffer.get()
        
        kl = agent.train_one_epoch(data)
        print(f"Epoch: {epoch+1}/{epochs}, KL Divergence: {kl:.4f}")
        
        if (epoch + 1) % 10 == 0:
            agent.save_models(f"ppo_lunar_actor_{epoch+1}.weights.h5", f"ppo_lunar_critic_{epoch+1}.weights.h5")
    
    agent.save_models("ppo_lunar_actor_final.weights.h5", "ppo_lunar_critic_final.weights.h5")
    return episode_rewards, episode_lengths, avg_rewards, agent

def evaluate_agent(agent, env_name="LunarLander-v3", episodes=10, render=False):
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    episode_rewards = []
    
    for i in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        episode_rewards.append(episode_reward)
        print(f"Evaluation episode {i+1}/{episodes}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average evaluation reward over {episodes} episodes: {avg_reward:.2f}")
    env.close()
    return episode_rewards

def plot_training_results(episode_rewards, avg_rewards):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(avg_rewards)
    plt.title('100-Episode Moving Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def load_and_evaluate(actor_path, critic_path, env_name="LunarLander-v3", episodes=5, render=True):
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    agent = PPOAgent(env.observation_space, env.action_space)
    agent.load_models(actor_path, critic_path)
    
    eval_rewards = evaluate_agent(agent, env_name, episodes, render)
    return eval_rewards

if __name__ == "__main__":
    episode_rewards, episode_lengths, avg_rewards, agent = train_ppo(
        env_name="LunarLander-v3",  # Updated to v3
        total_steps=500_000,
        steps_per_epoch=4000
    )
    
    plot_training_results(episode_rewards, avg_rewards)
    
    evaluate_agent(agent, episodes=10)
    
    load_and_evaluate("ppo_lunar_actor_best.weights.h5", "ppo_lunar_critic_best.weights.h5", episodes=5)
