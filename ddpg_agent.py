import torch
from torch import optim
import numpy as np
import os
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise
from itertools import chain

# https://github.com/A-Raafat/DDPG-bipedal/blob/master/My_DDPG.ipynb
class DDPGAgent():
    rl_type = 'ddpg'
    def __init__(self, Actor, Critic, state_size=24, action_size=4, 
            lr=1e-3, weight_decay=1e-4, gamma=0.99, tau=0.001, batch_size=128, buffer_size=int(7e5)):
        
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.train_actor = Actor().to(self.device)
        self.target_actor= Actor().to(self.device).eval()
        self.hard_update(self.train_actor, self.target_actor)
        self.actor_optim = optim.Adam(self.train_actor.parameters(), lr=0.1*lr, weight_decay=weight_decay)
        print(f'Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        self.train_critic = Critic().to(self.device)
        self.target_critic= Critic().to(self.device).eval()
        self.hard_update(self.train_critic, self.target_critic)
        self.critic_optim = optim.Adam(self.train_critic.parameters(), lr=lr, weight_decay=weight_decay)
        print(f'Number of paramters of Critic Net: {sum(p.numel() for p in self.train_critic.parameters())}')

        self.noise_generator = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size), theta=1.2, sigma=0.55, dt=0.02) # theta=0.15, sigma=0.2
        
        self.memory= ReplayBuffer(action_size= action_size, buffer_size= buffer_size, \
            batch_size= self.batch_size, device=self.device)
        
    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        if(len(self.memory)>self.batch_size):
            exp=self.memory.sample()
            self.learn(exp)
            
    def learn(self, exp):
        states, actions, rewards, next_states, done = exp
        
        #update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            Q_targets_next = self.target_critic(next_states, next_actions).detach()
            Q_targets = rewards + (self.gamma * Q_targets_next * (1-done))
        
        Q_expected = self.train_critic(states, actions)
        
        critic_loss = torch.nn.MSELoss()(Q_expected, Q_targets)
        #print(f"{critic_loss} - {Q_expected[0][0]} - {Q_targets[0][0]}")
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
        
    @torch.no_grad()        
    def get_action(self, state, explore=False):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        action = self.train_actor(state).cpu().data.numpy()[0]

        if explore:
            noise = self.noise_generator()
            #print(noise)
            action += noise
        return action
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_ckpt(self, model_type, prefix='last'):
        actor_file = os.path.join("models", self.rl_type, "_".join([prefix, model_type, "actor.pth"]))
        critic_file = os.path.join("models", self.rl_type, "_".join([prefix, model_type, "critic.pth"]))
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic.state_dict(), critic_file)

    def train_mode(self):
        self.train_actor.train()
        self.train_critic.train()

    def eval_mode(self):
        self.train_actor.eval()
        self.train_critic.eval()

    def freeze_networks(self):
        for p in chain(self.train_actor.parameters(), self.train_critic.parameters()):
            p.requires_grad = False