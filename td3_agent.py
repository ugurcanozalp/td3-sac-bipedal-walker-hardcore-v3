import torch
from torch import optim
import numpy as np
import os
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise, GaussianNoise

# https://github.com/A-Raafat/DDPG-bipedal/blob/master/My_DDPG.ipynb
class TD3Agent():
    rl_type = 'td3'
    def __init__(self, Actor, Critic, clip_low, clip_high, state_size=24, action_size=4, update_freq=int(4),
            lr=1e-3, weight_decay=1e-4, gamma=0.99, tau=0.004, batch_size=128, buffer_size=int(5e5)):
        
        self.state_size = state_size
        self.action_size = action_size
        self.update_freq = update_freq

        self.learn_call = int(0)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.clip_low = torch.tensor(clip_low)
        self.clip_high = torch.tensor(clip_high)

        self.train_actor = Actor().to(self.device)
        self.target_actor= Actor().to(self.device).eval()
        self.hard_update(self.train_actor, self.target_actor) # hard update at the beginning
        self.actor_optim = optim.Adam(self.train_actor.parameters(), lr=lr, weight_decay=weight_decay) 
        print(f'Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        self.train_critic_1 = Critic().to(self.device)
        self.target_critic_1 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_1, self.target_critic_1) # hard update at the beginning
        self.critic_1_optim = optim.Adam(self.train_critic_1.parameters(), lr=lr, weight_decay=weight_decay)

        self.train_critic_2 = Critic().to(self.device)
        self.target_critic_2 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_2, self.target_critic_2) # hard update at the beginning
        self.critic_2_optim = optim.Adam(self.train_critic_2.parameters(), lr=lr, weight_decay=weight_decay)
        print(f'Number of paramters of Single Critic Net: {sum(p.numel() for p in self.train_critic_2.parameters())}')

        self.noise_generator = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size), theta=1.2, sigma=0.55, dt=0.02)
        self.target_noise = GaussianNoise(mu=np.zeros(action_size), sigma=0.15, clip=0.5)
        
        self.memory= ReplayBuffer(action_size= action_size, buffer_size= buffer_size, \
            batch_size= self.batch_size, device=self.device)
        
    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        if(len(self.memory)>self.batch_size):
            exp=self.memory.sample()
            self.learn(exp)
            
    def learn(self, exp):
        self.learn_call+=1
        states, actions, rewards, next_states, done = exp
        
        #update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = next_actions + torch.from_numpy(self.target_noise()).float().to(self.device)
            next_actions = torch.clamp(next_actions, self.clip_low, self.clip_high)
            Q_targets_next_1 = self.target_critic_1(next_states, next_actions)
            Q_targets_next_2 = self.target_critic_2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1-done))
        
        Q_expected_1 = self.train_critic_1(states, actions)
        critic_1_loss = torch.nn.MSELoss()(Q_expected_1, Q_targets)

        self.critic_1_optim.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optim.step()

        Q_expected_2 = self.train_critic_2(states, actions)   
        critic_2_loss = torch.nn.MSELoss()(Q_expected_2, Q_targets)
        #critic_loss = torch.nn.SmoothL1Loss()(Q_expected, Q_targets)

        self.critic_2_optim.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optim.step()
        
        if self.learn_call % self.update_freq == 0:
            #update actor
            actions_pred = self.train_actor(states)
            actor_loss = -self.train_critic_1(states, actions_pred).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
        
            #using soft upates
            self.soft_update(self.train_actor, self.target_actor)
            self.soft_update(self.train_critic_1, self.target_critic_1)
            self.soft_update(self.train_critic_2, self.target_critic_2)
        
    @torch.no_grad()        
    def get_action(self, state, explore=False):
        self.train_actor.eval()
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        #with torch.no_grad():
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

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_ckpt(self, model_type, prefix='last'):
        actor_file = os.path.join("models", self.rl_type, "_".join([prefix, model_type, "actor.pth"]))
        critic_1_file = os.path.join("models", self.rl_type, "_".join([prefix, model_type, "critic_1.pth"]))
        critic_2_file = os.path.join("models", self.rl_type, "_".join([prefix, model_type, "critic_2.pth"]))
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic_1.state_dict(), critic_1_file)
        torch.save(self.train_critic_2.state_dict(), critic_2_file)