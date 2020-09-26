from __future__ import absolute_import
from collections import deque, namedtuple
import numpy as np
import random
import warnings

### for DQN and DRQN ###

class ReplayBuffer(object):
    """
    Replay buffer API. 
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store_transition(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


class RecurrentReplayBuffer(object):
    """
    Replay buffer with recurrent episodic experience API.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store_transition(self, observations, actions, rewards, next_observations, dones):
        """
        :param observations: np.array of [rel, rhs, response] of shape (seq_len, 3)
        :param actions:  np.array of flattened (rel, rhs) of shape (seq_len, 1)
        :param rewards:  np.array of [rewards] of shape (seq_len, 1)
        """
        observation      = np.expand_dims(observations, 0)
        next_observation = np.expand_dims(next_observations, 0)
            
        self.buffer.append((observation, actions, rewards, next_observation, dones))
    
    def sample(self, batch_size):
        observation, action, reward, next_observation, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(observation), action, reward, np.concatenate(next_observation), done
    
    def __len__(self):
        return len(self.buffer)