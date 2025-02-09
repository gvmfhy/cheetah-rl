import torch
import torch.nn.functional as F
import numpy as np
from .models import Actor, Critic

class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_tune_alpha=True
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters to target networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        
        if auto_tune_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
    
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if evaluate:
                action = self.actor(state)
            else:
                action = self.actor(state)
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1, 1)
            return action.cpu().numpy()[0]
    
    def train(self, replay_buffer, batch_size=256):
        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            # Select next action according to policy
            next_action = self.actor(next_state)
            noise = torch.randn_like(next_action) * 0.1
            next_action = torch.clamp(next_action + noise, -1, 1)
            
            # Compute target Q values
            target_q1, target_q2 = self.critic_1_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Compute current Q values
        current_q1, _ = self.critic_1(state, action)
        current_q2, _ = self.critic_2(state, action)
        
        # Compute critic loss
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Update actor
        actions_pred = self.actor(state)
        q1, q2 = self.critic_1(state, actions_pred)
        q = torch.min(q1, q2)
        actor_loss = -q.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update temperature parameter alpha (if auto-tuning)
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (self.target_entropy + q.detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'critic_1_target_state_dict': self.critic_1_target.state_dict(),
            'critic_2_target_state_dict': self.critic_2_target.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target_state_dict']) 