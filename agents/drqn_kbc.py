import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from .toy_models import Agent
from .utils import ReplayBuffer, RecurrentReplayBuffer


# assign gpu if it available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
EPSILON = 0.9           # action selection 
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
TARGET_REPLACE_ITER = int(1e4) # how often to update the network

class QNetwork(nn.Module):
    """ Critic Q Network."""
    def __init__(self, state_size, action_size, fc1_unit=64,
                 fc2_unit=64):

        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.fc1= nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RerruentQNetwork(nn.Module):
    """ Recurrnet Q Network."""
    def __init__(self, state_size, action_size, lstm_n_layer=1, bidirection=False, state_hidden=64,  fc1_unit=64,
                 fc2_unit = 64):

        super(RerruentQNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.bi = bidirection
        self.state_size = state_size
        self.hidden_size = state_hidden
        self.lstm = nn.GRU(input_size=self.state_size, hidden_size=self.hidden_size, num_layers=self.lstm_n_layer, 
            batch_first=True, bidirectional=self.bi)
        self.Qnet = QNetwork(self.stata_hidden_size, action_size, fc1_unit=64,
            fc2_unit = 64)

    def forward(self, x):
        """
        x: input (batch size, sequence len, feature size)
        """
        batch_size = x.shape[0]
        sequence_len = x.shape[1]
        out, hidden = self.model(x)
        out = out.view(-1, self.state_size)
        values = self.Qnet(out)
        values = values.view(batch_size, sequence_len, -1)
        return values

class DQN(object):
    def __init__(self, state_size, action_size):
        # set random seed manually
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        self.eval_net, self.target_net = QNetwork(state_size, action_size), QNetwork(state_size, action_size)
        self.eval_net.to(device)
        self.target_net.to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory = ReplayBuffer(BUFFER_SIZE)                         # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action


    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions

        b_s, b_a, b_r, b_s_, _ = self.memory.sample(BATCH_SIZE)
        b_s = torch.FloatTensor(b_s).to(device)
        b_a = torch.LongTensor(b_a.astype(int)).to(device)
        b_r = torch.FloatTensor(b_r).to(device)
        b_s_ = torch.FloatTensor(b_s_).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DRQN(object):
    def __init__(self, state_size, action_size):
        # set random seed manually
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        self.eval_net, self.target_net = RecurrentQNetwork(state_size, action_size), RecurrentQNetwork(state_size, action_size)
        self.eval_net.to(device)
        self.target_net.to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory = RecurrentReplayBuffer(BUFFER_SIZE)                # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action


    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions

        b_s, b_a, b_r, b_s_, _ = self.memory.sample(BATCH_SIZE)
        b_s = torch.FloatTensor(b_s).to(device)
        b_a = torch.LongTensor(b_a.astype(int)).to(device)
        b_r = torch.FloatTensor(b_r).to(device)
        b_s_ = torch.FloatTensor(b_s_).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DRQN_KBC(Agent):
    def __init__(self, ):
        pass