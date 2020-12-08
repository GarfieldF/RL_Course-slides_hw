#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Script
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import sys
from model import CNNModel
from a2c_agent import A2CAgent, ObsPreproc, TestAgent
sys.path.append('..')
from common import make_env, print_dict  # noqa

seed = 1000
num_procs = 16   # The number of processes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-4
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
num_frames_per_proc = 5  # num_frames_per_proc * num_procs = batch_size
train_epochs = 300000
test_episode = 10
log_interval = 100
test_interval = 1000
save_interval = 1000

env = make_env('BreakoutNoFrameskip-v4', seed, num_procs)
in_ch = env.observation_space.shape[-1]
n_action = env.action_space.n
import ipdb;ipdb.set_trace()
model = CNNModel(in_ch, n_action)
obs_preproc = ObsPreproc(device=device)
agent = A2CAgent(model, env, obs_preproc, device, lr, gamma, entropy_coef, value_loss_coef)

test_env = make_env('BreakoutNoFrameskip-v4', seed, 1, clip_reward=False)
test_agent = TestAgent(model, test_env, obs_preproc, device, test_episode)


for i in range(train_epochs):
    batch, log = agent.collect_batch(num_frames_per_proc)
    info = agent.update_parameters(batch)
    if i % log_interval == 0:
        print_dict({'step': i}, info, log)
    if i % test_interval == 0:
        print('=' * 20 + 'Test Agent' + '=' * 20)
        info = test_agent.evaluate()
        print_dict(info)
    if i % save_interval == 0:
        print('Save Model')
        torch.save(model.state_dict(), 'ckpt.pth')
