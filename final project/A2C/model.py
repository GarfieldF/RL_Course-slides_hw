#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Class
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import torch.nn as nn
import torch.distributions as D
from torch.nn import init


def weight_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class CNNModel(nn.Module):
    """A simple model
    Can be replaced by any complicated model
    """
    def __init__(self, in_ch, n_action):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4),  # [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),  # [64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),  # [64, 7, 7]
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(512, n_action),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(512, 1)
        self.apply(weight_init_orthogonal)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        action_prob = self.actor(x)
        dist = D.Categorical(action_prob)
        value = self.critic(x)
        return dist, value.squeeze(1)
