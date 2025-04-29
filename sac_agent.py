import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device).float()
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device).float()
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device).float()
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device).float()
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device).float()
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
    def select_action(self, states, evaluate=False):
        states = states.float()
        if evaluate:
            with torch.no_grad():
                actions = self.actor(states)
        else:
            with torch.no_grad():
                actions = self.actor(states)
                noise = torch.randn_like(actions) * 0.3
                actions = torch.clamp(actions + noise, -1.0, 1.0)
        return actions

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        reward_batch = reward_batch.unsqueeze(1).float()
        done_batch = done_batch.unsqueeze(1).float()
        with torch.no_grad():
            next_actions = self.actor(next_state_batch.float())
            target_q1 = self.target_critic1(next_state_batch.float(), next_actions)
            target_q2 = self.target_critic2(next_state_batch.float(), next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
        
        current_q1 = self.critic1(state_batch.float(), action_batch.float())
        current_q2 = self.critic2(state_batch.float(), action_batch.float())
        
        critic1_loss = F.mse_loss(current_q1.squeeze(-1), target_q.squeeze(-1))
        critic2_loss = F.mse_loss(current_q2.squeeze(-1), target_q.squeeze(-1))
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        actions_pred = self.actor(state_batch.float())
        q1_pred = self.critic1(state_batch.float(), actions_pred)
        q2_pred = self.critic2(state_batch.float(), actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.alpha * torch.log(torch.clamp(1 - actions_pred.pow(2), min=1e-6)) - q_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 