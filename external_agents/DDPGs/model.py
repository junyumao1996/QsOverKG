#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.drqn import Observation_Embedding

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=64, init_w=3e-3):
        super(Actor, self).__init__()   # hidden1=64, hidden2=64
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.tanh(out)
        out = self.softsign(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=128, init_w=3e-3):
        super(Critic, self).__init__()    # hidden1=64, hidden2=64
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # concatenate along columns
        c_in = torch.cat([out, a],len(a.shape)-1)
        out = self.fc2(c_in)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class Actor_Reurrent(nn.Module):
    def __init__(self, nb_states, nb_actions, n_predicates, n_entities, n_responses, hidden1=32, rnn_size=64, init_w=3e-3):
        super(Actor_Reurrent, self).__init__()
        self.fc1 = Observation_Embedding(n_predicates, n_entities, n_responses)  # output (..., 40)
        self.fc2 = nn.Linear(self.fc1.observation_size, hidden1)
        self.lstm = nn.LSTMCell(hidden1, rnn_size)
        self.fc3 = nn.Linear(rnn_size, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
        self.rnn_size = rnn_size

        self.cx = Variable(torch.zeros(1, rnn_size)).type(FLOAT)
        self.hx = Variable(torch.zeros(1, rnn_size)).type(FLOAT)
    
    def init_weights(self, init_w):
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def reset_lstm_hidden_state(self, done=True):
        if done == True:
            self.cx = Variable(torch.zeros(1, self.rnn_size)).type(FLOAT)
            self.hx = Variable(torch.zeros(1, self.rnn_size)).type(FLOAT)
        else:
            self.cx = Variable(self.cx.data).type(FLOAT)
            self.hx = Variable(self.hx.data).type(FLOAT)

    def forward(self, x, hidden_states=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if hidden_states == None:
            hx, cx = self.lstm(x, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx, cx = self.lstm(x, hidden_states)

        x = hx
        x = self.fc3(x)
        x = self.tanh(x)
        return x, (hx, cx)
