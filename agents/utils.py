from collections import deque 
import numpy as np
import random

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

class RecurrentReplayBuffer2(object):
    """
    Replay buffer with recurrent episodic experience API. 
    """
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length=sequence_length

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x-self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            # correct for sampling near beginning
            final = self.memory[max(start+1,0):end+1]
            
            # correct for sampling across episodes
            for i in range(len(final)-2, -1, -1):
                if final[i][3] is None:
                    final = final[i+1:]
                    break
                    
            # pad beginning to account for corrections
            while(len(final)<self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]))] + final
                            
            samp+=final

        # returns flattened version
        return samp, None, None

    def __len__(self):
        return len(self.memory)
