"""
[reference] https://github.com/ChangyWen/wolpertinger_ddpg
[reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py
"""

import os
import torch
import numpy as np
from torch.autograd import Variable
import logging
import math
from scipy.stats import entropy

def action2index(n_predicates, n_entities, rel: int, rhs: int):
    assert rel < n_predicates 
    assert rhs < n_entities
    return rel * n_entities + rhs

def index2action(n_predicates, n_entities, idx: int):
    assert idx < n_predicates * n_entities
    return idx // n_entities, idx % n_entities

# def reward_func(right=None, done=False, pos=0):
#     """
#     Reward function for the RL agent.
#     right: whether the final guess is right
#     done: whether it is the termination of IS module
#     """
#     if done==True:
#         if right==True:
#             r = 1.0
#         else:
#             r = -1.0
#     else:
#         r = 0.0
#     return r

def reward_func(right=None, done=False, pos=0):
    """
    Reward function for the RL agent.
    right: whether the final guess is right
    done: whether it is the termination of IS module
    """
    if done==True:
        if right==True:
            r = 1.0
        else:
            r = 0.0
    else:
        r = 0.0
    return r

def reward_func_intrinsic(posterior, posterior_la, mode='KL', r_upper=0.5, ratio=0.1):
    if mode == 'KL':
        r = np.abs(entropy(pk=posterior, qk=posterior_la, base=2)) * ratio
        # r = np.abs(entropy(pk=posterior_la, qk=posterior, base=2)) * ratio
    elif mode == 'MSE':
        distance = (posterior - posterior_la) ** 2
        r = np.mean(distance) * ratio
    else:
        raise RuntimeError
    return max(r, r_upper)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=0.01):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p
        self.step               = (final_p - initial_p)/schedule_timesteps

    def value(self, eps):
        """See Schedule.value"""
        new_eps = min(eps + self.step, self.final_p)
        assert new_eps >= self.initial_p 
        return new_eps


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_data(self):
        return self.data[:self.length]

    def append(self, v):
        # print(v, type(v))
        assert isinstance(v, np.ndarray) or isinstance(v, float) or isinstance(v, bool), "v_type:{}".format(type(v))
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v
    

def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)







