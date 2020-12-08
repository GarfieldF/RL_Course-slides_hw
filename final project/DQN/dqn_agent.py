
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import torch
import numpy as np
import random
import torch.nn.functional as F
from collections import deque, namedtuple
import ipdb


class ObsPreproc:
    def __init__(self, device):
        self.device = device

    def __call__(self, x):
        # input: [B, W, H, C]
        # output: [B, C, H, W]
        x = np.array(x)
        x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = torch.transpose(x, 1, 3)
        x = x / 255.0
        return x


class ReplayBuffer:
    """Can be only used for multiprocess"""
    def __init__(self, buffer_size, batch_size, num_procs):
        self.buffer_size = buffer_size // num_procs
        self.batch_size = batch_size // num_procs
        self._storage = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "mask"])

    def add(self, state, action, reward, next_state, mask):
        """All elements are tensors, and Dim(0) is the `num_procs`
        """
        e = self.experience(state, action, reward, next_state, mask)
        self._storage.append(e)

    def sample(self):
        exps = random.sample(self._storage, k=self.batch_size)
        states = torch.cat([e.state for e in exps if e is not None], dim=0)
        actions = torch.cat([e.action for e in exps if e is not None], dim=0)
        rewards = torch.cat([e.reward for e in exps if e is not None], dim=0)
        next_states = torch.cat([e.next_state for e in exps if e is not None], dim=0)
        masks = torch.cat([e.mask for e in exps if e is not None], dim=0)
        batch = (states, actions, rewards, next_states, masks)
        return batch

    def __len__(self):
        return len(self._storage)


class LinearSchedule:
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def __call__(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class DQNAgent:
    def __init__(self, model, target_model, env, obs_preproc, device, lr=7e-4, gamma=0.99, num_procs=16, batch_size=100, buffer_size=100000, max_grad_norm=0.5):
        self.model = model
        self.target_model = target_model
        self.obs_preproc = obs_preproc
        self.env = env
        self.n_action = self.env.action_space.n
        self.lr = lr
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.num_procs = num_procs
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size=batch_size, num_procs=num_procs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.reset()
        if self.device == torch.device('cuda'):
            self.model.cuda()
            self.target_model.cuda()

    def reset(self):
        obs = self.env.reset()
        self.preprocessed_obs = self.obs_preproc(obs)

        self.episode = 0
        self.episode_return = deque([0] * 100, maxlen=100)   # record last 100 episode return
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device, dtype=torch.float)

    def act(self, obs, epsilon):
        """If epsilon >= 1, use random action"""
        if random.random() > epsilon:
            with torch.no_grad():
                Q_value = self.model(obs)
                action = Q_value.argmax(dim=1)
        else:
            action = torch.tensor(np.random.randint(0, self.n_action, size=obs.shape[0]), device=self.device, dtype=torch.int64)
        return action

    def collect_exp(self, num_frames, epsilon):
        """Collect num_frames exps
        """
        for i in range(num_frames):
            action = self.act(self.preprocessed_obs, epsilon)
            next_obs, reward, done, info = self.env.step(action.cpu().numpy())
            next_preprocessed_obs = self.obs_preproc(next_obs)
            reward = torch.tensor(reward, device=self.device)
            mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            self.replay_buffer.add(
                self.preprocessed_obs,
                action,
                reward,
                next_preprocessed_obs,
                mask
            )

            # For log
            self.log_episode_return += reward
            for i, done_ in enumerate(done):
                if done_:
                    self.episode += 1
                    self.episode_return.append(self.log_episode_return[i].item())
                    self.log_episode_return[i] = 0

            # For next iteration
            self.preprocessed_obs = next_preprocessed_obs
        #ipdb.set_trace()
        log = {
            'episode': self.episode,
            'average_return': np.mean(self.episode_return),
        }
        return log

    def update_parameters(self, update_times):
        for _ in range(update_times):
            states, actions, rewards, next_states, masks = self.replay_buffer.sample()

            output = self.model(states)
            #ipdb.set_trace()
            curr_Q=torch.gather(output,1, actions.unsqueeze(1)).squeeze(1)
            #ipdb.set_trace()
            with torch.no_grad():
                next_max_Q = self.target_model(next_states).max(dim=1)[0]
                target_Q = rewards + self.gamma * masks * next_max_Q
            loss = F.mse_loss(curr_Q, target_Q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        info = {
            'value': curr_Q.mean().item(),
            'loss': loss.item()
        }
        return info

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


class TestAgent:
    def __init__(self, model, env, obs_preproc, device, test_episode=100):
        self.model = model
        self.obs_preproc = obs_preproc
        self.device = device
        self.env = env  # Not ParallelEnv, Not ClipReward
        self.test_episode = test_episode

    def act(self, obs):
        preprocessed_obs = self.obs_preproc([obs])
        with torch.no_grad():
            Q_value = self.model(preprocessed_obs)
            #ipdb.set_trace()
            action = Q_value.argmax(dim=1)
        return action.cpu().numpy()

    def evaluate(self):
        self.model.eval()
        episode_return_list = []
        for i in range(self.test_episode):
            seed = np.random.randint(0, 0xFFFFFF)
            self.env.seed(seed)
            obs = self.env.reset()
            done = False
            episode_return = 0
            while not done:
                action = self.act(obs)
                obs, reward, done, _ = self.env.step(action)
                episode_return += reward
            print(episode_return)

            episode_return_list.append(episode_return)
        info = {'average_return': np.mean(episode_return)}
        self.model.train()
        return info

    def display(self):
        self.model.eval()
        seed = np.random.randint(0, 0xFFFFFF)
        self.env.seed(seed)
        obs = self.env.reset()
        need_key = True
        episode = 0
        episode_return = 0
        print('`Enter`: next step\n`E`: Run until end-of-episode\n`Q`: Quit')
        while True:
            if need_key:
                key = input('Press key:')
                if key == 'q':  # quit
                    break
                if key == 'e':  # Run until end-of-episode
                    need_key = False
            self.env.render()
            action = self.act(obs).squeeze(0)
            obs, reward, done, _ = self.env.step(action)
            print(reward)
            episode_return += reward
            if done:
                episode += 1
                obs = self.env.reset()
                import ipdb;ipdb.set_trace()
                print('episode: {}, episode_return: {}'.format(episode, episode_return))
                episode_return = 0
                need_key = True
        self.model.train()
