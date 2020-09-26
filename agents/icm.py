import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import random
import numpy as np
import math

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(IntrinsicCuriosityModule, self).__init__()
        self.feature_size = num_inputs
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_actions)
        )
        self.action_embed = nn.Embedding(num_actions, self.feature_size)
        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.feature_size),
            nn.Softmax(dim=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        self.action_embed.weight.data.uniform_(-0.1, 0.1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        state_ft = state
        next_state_ft = next_state
        state_ft = state_ft.view(-1, self.feature_size)
        next_state_ft = next_state_ft.view(-1, self.feature_size)
        action = self.action_embed(action).view(-1, self.feature_size)

        return None, self.forward_net(
            torch.cat((state_ft, action), 1)), next_state_ft


