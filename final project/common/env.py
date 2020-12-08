#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Dagui Chen
Email: goblin_chen@163.com

``````````````````````````````````````
Env Class

Some Wrapper class to make environment trainable

Modify from:
    https://github.com/openai/baselines
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.remove( "/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import gym
from gym import spaces
import numpy as np
from multiprocessing import Process, Pipe
from collections import deque


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking randonm number of no-ops of reset.
    No-op is assumed to be action 0.
    """
    def __init__(self, env, noop_max=30, noop_action=0):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = noop_action
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing
    """
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame
    """
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations.
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            elif i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign
        """
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over/
    Done by Deepmind for the DQN and co. since it helps value estimation
    """
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0  # for some game, this means number of lives
        self.was_real_done = True

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info


class WrapFrame(gym.ObservationWrapper):
    """Wrap frames to 84x84 as done in the Nature paper and later work
    """
    def __init__(self, env, width=84, height=84):
        super(WrapFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class StackFrame(gym.Wrapper):
    """Stack the frames as the state
    """
    def __init__(self, env, num_stack):
        super(StackFrame, self).__init__(env)
        self.num_stack = num_stack
        self.obs_queue = deque(maxlen=self.num_stack)
        raw_os = self.env.observation_space
        low = np.repeat(raw_os.low, num_stack, axis=-1)
        high = np.repeat(raw_os.high, num_stack, axis=-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=raw_os.dtype)

    def reset(self):
        obs = self.env.reset()
        for i in range(self.num_stack - 1):
            self.obs_queue.append(np.zeros_like(obs, dtype=np.uint8))
        self.obs_queue.append(obs)
        stack_obs = np.concatenate(self.obs_queue, axis=-1)  # [..., c * num_stack]
        return stack_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs_queue.append(obs)
        stack_obs = np.concatenate(self.obs_queue, axis=-1)
        return stack_obs, reward, done, info


##############################
#          Multi-processing
##############################
def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Wrapper):
    def __init__(self, envs):
        """envs is a list of env
        If sub_env is done, automatically reset
        """
        assert len(envs) >= 1, 'No environment is given'
        super(ParallelEnv, self).__init__(envs[0])  # make self.observation_space consistent with sigle process
        self._num_procs = len(envs)
        self.envs = envs
        self.closed = False

        self.locals = []
        for env in self.envs:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(('step', action))
        results = zip(*[local.recv() for local in self.locals])
        return results

    def render(self, mode, **kwargs):
        raise NotImplementedError

    def unwrapped(self):
        raise NotImplementedError


##############################
#          Util-function
##############################
def make_atari(env_id, seed, episodic_life=True, clip_reward=True, noop_max=30, skip=4):
    env = gym.make(env_id)
    if noop_max is not None:
        assert isinstance(noop_max, int)
        env = NoopResetEnv(env, noop_max=noop_max)
    if skip is not None:
        assert isinstance(skip, int)
        env = MaxAndSkipEnv(env, skip=skip)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WrapFrame(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env.seed(seed)
    return env


def make_env(env_id, seed, num_procs, num_stack=4, **kwargs):
    def _make_env(rank):
        env = make_atari(env_id, seed + rank, **kwargs)
        return env
    if num_procs > 1:
        envs = [_make_env(i) for i in range(num_procs)]
        env = ParallelEnv(envs)
    else:
        env = _make_env(0)
    if num_stack is not None:
        assert isinstance(num_stack, int)
        env = StackFrame(env, num_stack=num_stack)
    return env
