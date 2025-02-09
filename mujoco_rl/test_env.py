import gym
import torch
import numpy as np
from src.agent import SACAgent

def test_agent(model_path, episodes=5):
    # Environment setup
    env = gym.make('HalfCheetah-v4', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(state_dim, action_dim, device)
    agent.load(model_path)
    
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            done = done or truncated
            
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Test the best model
    print("Testing best model:")
    test_agent('cheetah_best_actor')
    
    # Test the final model
    print("\nTesting final model:")
    test_agent('cheetah_final_actor') 