import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import random
from src.agent import SACAgent

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

def train():
    # Environment setup
    env = gym.make('HalfCheetah-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Training parameters
    max_episodes = 1000
    max_steps = 1000
    batch_size = 256
    updates_per_step = 1
    random_steps = 10000
    
    # Initialize agent and replay buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(state_dim, action_dim, device)
    replay_buffer = ReplayBuffer()
    
    # Logging
    episode_rewards = []
    running_reward = 0
    best_reward = -np.inf
    
    # Training loop
    total_steps = 0
    for episode in range(max_episodes):
        state = env.reset()[0]
        episode_reward = 0
        
        for step in range(max_steps):
            if total_steps < random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            next_state, reward, done, truncated, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Train agent
            if len(replay_buffer) > batch_size and total_steps > random_steps:
                for _ in range(updates_per_step):
                    agent.train(replay_buffer, batch_size)
            
            if done or truncated:
                break
        
        # Logging
        episode_rewards.append(episode_reward)
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('cheetah_best_actor')
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Running Reward: {running_reward:.2f}")
            
            # Plot progress
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.title('Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig('cheetah_progress.png')
            plt.close()
    
    # Save final model
    agent.save('cheetah_final_actor')
    
    # Plot final results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Training Results')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('cheetah_rewards.png')
    plt.close()

if __name__ == "__main__":
    train() 