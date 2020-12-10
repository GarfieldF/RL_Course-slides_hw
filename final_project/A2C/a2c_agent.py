import torch
import numpy as np
import torch.nn.functional as F
from collections import deque


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


class A2CAgent:
    def __init__(self, model, env, obs_preproc, device, lr=7e-4, gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5):
        self.model = model
        self.obs_preproc = obs_preproc
        self.env = env
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.reset()
        if self.device == torch.device('cuda'):
            self.model.cuda()

    def reset(self):
        obs = self.env.reset()
        self.preprocessed_obs = self.obs_preproc(obs)

        self.episode = 0
        self.episode_return = deque([0] * 100, maxlen=100)   # record last 100 episode return
        self.log_episode_return = torch.zeros(len(self.preprocessed_obs), device=self.device, dtype=torch.float)

    def act(self, obs):
        with torch.no_grad():
            action_dist, value = self.model(obs)
            action = action_dist.sample()
        return action, action_dist, value

    def collect_batch(self, num_frames_per_proc):
        """batch_size = num_frames_per_proc * num_proc
        """
        batch_obs = [None] * num_frames_per_proc
        batch_reward = [None] * num_frames_per_proc
        batch_action = [None] * num_frames_per_proc
        batch_value = [None] * num_frames_per_proc
        batch_adv = [None] * num_frames_per_proc
        batch_mask = [None] * num_frames_per_proc
        for i in range(num_frames_per_proc):
            action, action_dist, value = self.act(self.preprocessed_obs)
            obs, reward, done, info = self.env.step(action.cpu().numpy())
            batch_obs[i] = self.preprocessed_obs
            batch_mask[i] = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            batch_reward[i] = torch.tensor(reward, device=self.device, dtype=torch.float)
            batch_action[i] = action
            batch_value[i] = value

            # For log
            self.log_episode_return += batch_reward[i]
            for i, done_ in enumerate(done):
                if done_:
                    self.episode += 1
                    self.episode_return.append(self.log_episode_return[i].item())
                    self.log_episode_return[i] = 0

            # For next iteration
            self.preprocessed_obs = self.obs_preproc(obs)

        # calculate advantage value
        _, _, next_value = self.act(self.preprocessed_obs)
        for i in reversed(range(num_frames_per_proc)):
            if i < num_frames_per_proc - 1:
                next_value = batch_value[i + 1]
            else:
                next_value = next_value
            batch_adv[i] = batch_reward[i] + self.gamma * next_value * batch_mask[i] - batch_value[i]

        # Below tensors: Dim(0) = num_frames_per_proc x num_procs
        batch_obs = torch.cat(batch_obs, dim=0)
        batch_reward = torch.cat(batch_reward, dim=0)
        batch_action = torch.cat(batch_action, dim=0)
        batch_value = torch.cat(batch_value, dim=0)
        batch_adv = torch.cat(batch_adv, dim=0)
        batch_mask = torch.cat(batch_mask, dim=0)
        batch_target_value = batch_value + batch_adv
        batch = (batch_obs, batch_action, batch_reward, batch_mask, batch_adv, batch_target_value)

        log = {'episode': self.episode, 'average_return': np.mean(self.episode_return)}
        return batch, log

    def update_parameters(self, batch):
        obs, action, reward, mask, advantage, target_value = batch
        action_dist, value = self.model(obs)
        entropy = action_dist.entropy().mean()
        policy_loss = -torch.mean(action_dist.log_prob(action) * advantage.detach())
        value_loss = F.mse_loss(value, target_value)
        #import ipdb; ipdb.set_trace()
        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
        info = {
            'entropy': entropy.item(),
            'value': value.mean().item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'loss': loss.item()
        }
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return info


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
            action_dist, value = self.model(preprocessed_obs)
            action = action_dist.sample()
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
            episode_return += reward
            if done:
                episode += 1
                obs = self.env.reset()
                print('episode: {}, episode_return: {}'.format(episode, episode_return))
                episode_return = 0
                need_key = True
        self.model.train()
