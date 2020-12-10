#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import torch
from model import CNNModel
from dqn_agent import ObsPreproc, TestAgent
import sys
sys.path.append('..')
from common import make_env  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./ckpt.pth', help='The model path')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = make_env('BreakoutNoFrameskip-v4', 0, 1, clip_reward=False)
in_ch = env.observation_space.shape[-1]
n_action = env.action_space.n

model = CNNModel(in_ch, n_action)
model.cuda()
model.load_state_dict(torch.load(opt.model_path, map_location=device))

obs_preproc = ObsPreproc(device=device)
test_agent = TestAgent(model, env, obs_preproc, device, 30)

info=test_agent.evaluate()
#print(info['average_return'])
#writer.add_scalar("reward",info['average_return'],i)
test_agent.display()
