import torch
from torch import optim
import numpy as np
import os
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise

# https://github.com/A-Raafat/DDPG-bipedal/blob/master/My_DDPG.ipynb
class DDPGAgent():
    rl_type = 'ddpg'
    def __init__(self, Actor, Critic, state_size=24, action_size=4, 
            lr=1e-3, gamma=0.99, tau=0.001, batch_size=128, buffer_size=int(5e5)):
        
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.train_actor = Actor().to(self.device)
        self.target_actor= Actor().to(self.device)
        self.actor_optim = optim.Adam(self.train_actor.parameters(), lr=0.25*lr)
        print(f'Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        self.train_critic = Critic().to(self.device)
        self.target_critic= Critic().to(self.device)
        self.critic_optim = optim.Adam(self.train_critic.parameters(), lr=lr)
        print(f'Number of paramters of Critic Net: {sum(p.numel() for p in self.train_critic.parameters())}')

        self.noise_generator = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
        
        self.memory= ReplayBuffer(action_size= action_size, buffer_size= buffer_size, \
            batch_size= self.batch_size, device=self.device)
        
    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        if(len(self.memory)>self.batch_size):
            exp=self.memory.sample()
            self.learn(exp)
            
    def learn(self, exp):
        states, actions, rewards, next_states, done= exp
        
        #update critic
        next_actions = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-done))
        Q_expected = self.train_critic(states, actions)
        
        critic_loss = torch.nn.MSELoss()(Q_expected, Q_targets)
        #critic_loss = torch.nn.SmoothL1Loss()(Q_expected, Q_targets)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        #update actor
        actions_pred = self.train_actor(states)
        actor_loss = -self.train_critic(states, actions_pred).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        #using soft upates
        self.soft_update(self.train_actor, self.target_actor)
        self.soft_update(self.train_critic, self.target_critic)
        
            
    def get_action(self, state, explore=False):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        self.train_actor.eval()
        with torch.no_grad():
            action= self.train_actor(state).cpu().data.numpy()[0]
        self.train_actor.train()

        if explore:
            noise = self.noise_generator()
            #print(noise)
            action += noise
        return action
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)