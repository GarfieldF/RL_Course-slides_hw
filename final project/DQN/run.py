#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import sys
from model import CNNModel
from dqn_agent import DQNAgent, ObsPreproc, LinearSchedule, TestAgent
sys.path.append('..')
from common import make_env, print_dict  # noqa
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import utils
seed = 1000
num_procs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-4
gamma = 0.99
train_epochs = 300000
batch_size = 40
buffer_size = 5000
learning_start = 10000   # Start learning after some steps
final_epsilon = 0.1   # Final epsilon
update_freq = 4   # update frequency
update_times = 4  # update times
test_episode = 10
log_interval = 100
test_interval = 1000
save_interval = 1000
update_target_interval = 100
timestr = datetime.now().strftime("%m-%d_%H:%M:%S")
sys.stdout = utils.LoggerOutput('./' + timestr  + '.txt')
writer=SummaryWriter("./log/"+ timestr+"/")
env = make_env('BreakoutNoFrameskip-v4', seed, num_procs)
in_ch = env.observation_space.shape[-1]
n_action = env.action_space.n

model = CNNModel(in_ch, n_action)
target_model = CNNModel(in_ch, n_action)
obs_preproc = ObsPreproc(device=device)
agent = DQNAgent(model, target_model, env, obs_preproc, device, lr, gamma, num_procs, batch_size, buffer_size)
exploration = LinearSchedule(400000, final_p=final_epsilon, initial_p=1.0)


test_env = make_env('BreakoutNoFrameskip-v4', seed, 1, clip_reward=False)
test_agent = TestAgent(model, test_env, obs_preproc, device, test_episode)


agent.collect_exp(learning_start, 1.0)    # collect some samples via random action
for i in range(train_epochs):
    curr_epsilon = exploration(i)
    log = agent.collect_exp(update_freq, curr_epsilon)
    info = agent.update_parameters(update_times)
    writer.add_scalar("reward",log['average_return'],i)
    writer.add_scalar("loss",info['loss'],i)
    writer.add_scalar("value",info['value'],i)
    if i % log_interval == 0:
        print_dict({'step': i, 'epsilon': curr_epsilon}, info, log)
    if i % test_interval == 0:
        print('=' * 20 + 'Test Agent' + '=' * 20)
        info = test_agent.evaluate()
        print_dict(info)
    if i % save_interval == 0:
        print('Save Model')
        torch.save(model.state_dict(), 'ckpt.pth')
    if i % update_target_interval == 0:
        agent.update_target_model()
