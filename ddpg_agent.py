import torch
from torch import optim
import numpy as np
import os
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise
from itertools import chain

class DDPGAgent():
    rl_type = 'ddpg'
    def __init__(self, Actor, Critic, state_size=24, action_size=4, 
            lr=4e-4, weight_decay=0, gamma=0.98, tau=0.01, batch_size=64, buffer_size=int(500000), device=None):
        
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = torch.device(device)

        self.train_actor = Actor().to(self.device)
        self.target_actor= Actor().to(self.device).eval()
        self.hard_update(self.train_actor, self.target_actor)
        self.actor_optim = torch.optim.AdamW(self.train_actor.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        self.train_critic = Critic().to(self.device)
        self.target_critic= Critic().to(self.device).eval()
        self.hard_update(self.train_critic, self.target_critic)
        self.critic_optim = torch.optim.AdamW(self.train_critic.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Critic Net: {sum(p.numel() for p in self.train_critic.parameters())}')

        self.noise_generator = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size), theta=3.0, sigma=0.9, dt=0.04) # #theta=1.2, sigma=0.55
        
        self.memory= ReplayBuffer(action_size= action_size, buffer_size= buffer_size, \
            batch_size= self.batch_size, device=self.device)
        
        self.mse_loss = torch.nn.MSELoss()

    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn_one_step()

    def learn_one_step(self):
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
        
        critic_loss = self.mse_loss(Q_expected, Q_targets)
        #critic_loss = torch.nn.SmoothL1Loss()(Q_expected, Q_targets)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_critic.parameters(), 1)
        self.critic_optim.step()
        
        #update actor
        actions_pred = self.train_actor(states)
        actor_loss = -self.train_critic(states, actions_pred).mean()
        
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_actor.parameters(), 1)
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

    def save_ckpt(self, model_type, env_type, prefix='last'):
        actor_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "actor.pth"]))
        critic_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic.pth"]))
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic.state_dict(), critic_file)

    def load_ckpt(self, model_type, env_type, prefix='last'):
        actor_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "actor.pth"]))
        critic_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic.pth"]))
        try:
            self.train_actor.load_state_dict(torch.load(actor_file, map_location=self.device))
        except:
            print("Actor checkpoint cannot be loaded.")
        try:
            self.train_critic.load_state_dict(torch.load(critic_file, map_location=self.device))
        except:
            print("Critic checkpoint cannot be loaded.")

    def train_mode(self):
        self.train_actor.train()
        self.train_critic.train()

    def eval_mode(self):
        self.train_actor.eval()
        self.train_critic.eval()

    def freeze_networks(self):
        for p in chain(self.train_actor.parameters(), self.train_critic.parameters()):
            p.requires_grad = False

    def step_end(self):
        self.noise_generator.step_end()

    def episode_end(self):
        self.noise_generator.episode_end()   