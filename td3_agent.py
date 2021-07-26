import torch
from torch import optim
import numpy as np
import os
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise, DecayingOrnsteinUhlenbeckNoise, GaussianNoise, DecayingGaussianNoise, DecayingRandomNoise
from itertools import chain

class TD3Agent():
    rl_type = 'td3'
    def __init__(self, Actor, Critic, clip_low, clip_high, state_size=24, action_size=4, update_freq=int(2),
            lr=4e-4, weight_decay=0, gamma=0.98, tau=0.01, batch_size=64, buffer_size=int(500000), device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.update_freq = update_freq

        self.learn_call = int(0)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = torch.device(device)

        self.clip_low = torch.tensor(clip_low)
        self.clip_high = torch.tensor(clip_high)

        self.train_actor = Actor().to(self.device)
        self.target_actor= Actor().to(self.device).eval()
        self.hard_update(self.train_actor, self.target_actor) # hard update at the beginning
        self.actor_optim = torch.optim.AdamW(self.train_actor.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        self.train_critic_1 = Critic().to(self.device)
        self.target_critic_1 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_1, self.target_critic_1) # hard update at the beginning
        self.critic_1_optim = torch.optim.AdamW(self.train_critic_1.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

        self.train_critic_2 = Critic().to(self.device)
        self.target_critic_2 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_2, self.target_critic_2) # hard update at the beginning
        self.critic_2_optim = torch.optim.AdamW(self.train_critic_2.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Single Critic Net: {sum(p.numel() for p in self.train_critic_2.parameters())}')

        self.noise_generator = DecayingOrnsteinUhlenbeckNoise(mu=np.zeros(action_size), theta=4.0, sigma=1.2, dt=0.04, sigma_decay=0.9995)
        self.target_noise = GaussianNoise(mu=np.zeros(action_size), sigma=0.2, clip=0.4)
        
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
        self.learn_call+=1
        states, actions, rewards, next_states, done = exp
        
        #update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = next_actions + torch.from_numpy(self.target_noise()).float().to(self.device)
            next_actions = torch.clamp(next_actions, self.clip_low, self.clip_high)
            Q_targets_next_1 = self.target_critic_1(next_states, next_actions)
            Q_targets_next_2 = self.target_critic_2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2).detach()
            Q_targets = rewards + (self.gamma * Q_targets_next * (1-done))
            #Q_targets = rewards + (self.gamma * Q_targets_next)
        
        Q_expected_1 = self.train_critic_1(states, actions)
        critic_1_loss = self.mse_loss(Q_expected_1, Q_targets)
        #critic_1_loss = torch.nn.SmoothL1Loss()(Q_expected_1, Q_targets)

        self.critic_1_optim.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_critic_1.parameters(), 1)
        self.critic_1_optim.step()

        Q_expected_2 = self.train_critic_2(states, actions)   
        critic_2_loss = self.mse_loss(Q_expected_2, Q_targets)
        #critic_2_loss = torch.nn.SmoothL1Loss()(Q_expected_2, Q_targets)
        
        self.critic_2_optim.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_critic_2.parameters(), 1)
        self.critic_2_optim.step()
        
        if self.learn_call % self.update_freq == 0:
            self.learn_call = 0
            #update actor
            actions_pred = self.train_actor(states)
            actor_loss = -self.train_critic_1(states, actions_pred).mean()
            
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.train_actor.parameters(), 1)
            self.actor_optim.step()
        
            #using soft upates
            self.soft_update(self.train_actor, self.target_actor)
            self.soft_update(self.train_critic_1, self.target_critic_1)
            self.soft_update(self.train_critic_2, self.target_critic_2)
        
    @torch.no_grad()        
    def get_action(self, state, explore=False):
        #self.train_actor.eval()
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        #with torch.no_grad():
        action = self.train_actor(state).cpu().data.numpy()[0]
        #self.train_actor.train()

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
        critic_1_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic_1.pth"]))
        critic_2_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic_2.pth"]))
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic_1.state_dict(), critic_1_file)
        torch.save(self.train_critic_2.state_dict(), critic_2_file)

    def load_ckpt(self, model_type, env_type, prefix='last'):
        actor_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "actor.pth"]))
        critic_1_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic_1.pth"]))
        critic_2_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic_2.pth"]))
        try:
            self.train_actor.load_state_dict(torch.load(actor_file, map_location=self.device))
        except:
            print("Actor checkpoint cannot be loaded.")
        try:
            self.train_critic_1.load_state_dict(torch.load(critic_1_file, map_location=self.device))
            self.train_critic_2.load_state_dict(torch.load(critic_2_file, map_location=self.device))
        except:
            print("Critic checkpoints cannot be loaded.")              

    def train_mode(self):
        self.train_actor.train()
        self.train_critic_1.train()
        self.train_critic_2.train()

    def eval_mode(self):
        self.train_actor.eval()
        self.train_critic_1.eval()
        self.train_critic_2.eval()

    def freeze_networks(self):
        for p in chain(self.train_actor.parameters(), self.train_critic_1.parameters(), self.train_critic_2.parameters()):
            p.requires_grad = False

    def step_end(self):
        self.noise_generator.step_end()

    def episode_end(self):
        self.noise_generator.episode_end()    

"""
def eval_grad_norm(name, model):
    total_norm = 0
    for p in filter(lambda p: p.grad is not None, model.parameters()):
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"{name}:{total_norm}")
"""