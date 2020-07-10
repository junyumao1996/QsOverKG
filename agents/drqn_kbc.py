import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import copy
from .toy_models import Agent
from .utils import ReplayBuffer, RecurrentReplayBuffer
from datasets import InternalKB


# assign gpu if it available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
EPSILON = 0.9           # action selection
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
FREEZE_INTERVAL = int(1e4) # how often to update the network

class Observation_Embedding(nn.Module):
    """
    Embedding for constructing the observation.
    """
    def __init__(self, n_predicates, n_entities, n_responses, r_size=10):
        super(Observation_Embedding, self).__init__()
        
        self.q_embed = Question_Embedding(n_predicates, n_entities)      # question embedding
        self.r_embed = nn.Embedding(n_responses, r_size)                 # response embedding
        self.observation_size = self.q_embed.question_size + r_size

    def forward(self, x):
        """
        x: of size (...,  3:[rel, rhs, response]), idealy (batch size, sequence len, 3:[rel, rhs, response])
        """
        q = x[..., :-1]
        r = x[..., -1]
        x_q = self.q_embed(q)
        x_r = self.r_embed(r)
        return torch.cat([x_q, x_r], dim=-1)

class Question_Embedding(nn.Module):
    """
    Embedding representation of questions.
    """
    def __init__(self, n_predicates, n_entities, rel_size=128, rhs_size=128):
        super(Question_Embedding, self).__init__()
        self.rel_embed = nn.Embedding(n_predicates, rel_size)               # predicate embedding
        self.rhs_embed = nn.Embedding(n_entities, rhs_size)                 # entity embedding
        self.question_size = rel_size + rhs_size

    def forward(self, x):
        """
        x: of size (..., 2:[rel, rhs]), idealy (batch size, sequence len, 2:[rel, rhs])
        """
        rel = x[..., 0]
        rhs = x[..., 1]
        x_rel = self.rel_embed(rel)
        x_rhs = self.rhs_embed(rhs)
        return torch.cat([x_rel, x_rhs], dim=-1)

class QNetwork(nn.Module):
    """Critic Q Network."""
    def __init__(self, observation_size, action_size, fc1_unit=64,
                 fc2_unit=64):
        """
        :param observation_size: dimension of observation embedding
        :param action_size: number of actions
        """

        super(QNetwork, self).__init__() 
        self.fc1= nn.Linear(observation_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)
        
    def forward(self, x):
        """
        Build a network that maps state -> action values.
        x: input (..., observation_size)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RecurrentQNetwork(nn.Module):
    """Recurrnet Q Network."""
    def __init__(self, observation_size, action_size, state_hidden=256, lstm_n_layer=1, bidirection=False):
        """
        :param action_size: number of actions
        """

        super(RecurrentQNetwork, self).__init__()
        self.o_size = observation_size
        self.hidden_size = state_hidden
        self.bi = bidirection

        self.rnn = nn.GRU(input_size=self.o_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.lstm_n_layer, 
            batch_first=True, 
            bidirectional=self.bi)                                           # recurrent hidden representation
        self.Qnet = nn.Linear(self.hidden_size, action_size)                 # q value approximation

    def forward(self, x):
        """
        x: input (batch size, sequence len, observation_size)
        """
        out, hidden = self.rnn(x)   # out: (batch size, sequence len, self.hidden_size)
        q_values = self.Qnet(out[:, -1, :])
        return q_values

def action2index(n_predicates, n_entities, rel: int, rhs: int):
    assert rel < n_predicates 
    assert rhs < n_entities
    return rel * n_entities + rhs

def index2action(n_predicates, n_entities, idx: int):
    assert idx < n_predicates * n_entities
    return idx // n_entities, idx % n_entities

class DQN(object):
    def __init__(self, state_size, action_size, lr=5e-4):
        # set random seed manually
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        self.n_actions = action_size

        self.target_net = QNetwork(state_size, action_size).to(device)
        self.eval_net = QNetwork(state_size, action_size).to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory = ReplayBuffer(BUFFER_SIZE)                         # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        """
        :param: x: of shape (4*n_actions, )
        """
        x = torch.unsqueeze(torch.FloatTensor(x), 0)          # add batch size as 1
        # epsilon-greedy policy
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] 
        else:                               # random
            action = np.random.randint(0,  self.n_actions)
            action = action 
        return action

    def store_transition(self, s, a, r, s_, done):
        """
        Store the transition.
        :param s, s_: of shape (4*n_actions, )
        :param a: [idx]
        """
        self.memory.store_transition(s, a, r, s_, done)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % FREEZE_INTERVAL == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_, _ = self.memory.sample(BATCH_SIZE)
        b_s = torch.FloatTensor(b_s).to(device)
        b_a = torch.LongTensor(b_a).to(device)
        b_r = torch.FloatTensor(b_r).to(device)
        b_s_ = torch.FloatTensor(b_s_).to(device)


        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)                      # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()                         # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class AgentDQN(Agent):
    """
    DQN agent with designed state. 
    """
    def __init__(self, dataset: InternalKB, switch_thres, n_chances=20):
        super(AgentDQN, self).__init__(dataset, n_chances, switch_thres)
        self.n_actions = self.n_predicates * self.n_entities
        self.IS = DQN(np.prod(self.state_size), self.n_actions)
        self.episode_state = np.zeros((self.n_actions, 4))  # representation of current question&response history (i.e., state)

    def reset(self):
        """
        Reset agent state and episode (when new episode starts). 
        """
        self.t = 0
        self.module = "IS" 
        self.module_switch = False
        self.guess_right = False
        self.episode_log = {'question':list(), 'response':list()}
        self.state_size = [self.n_predicates * self.n_entities, 4]
        self.episode_state = np.zeros(self.state_size) 

    def update_response(self, response):
        """
        Append the lastest response.
        """
        self.episode_log['response'].append(response)
        self.t += 1
        assert len(self.episode_log['response']) == len(self.episode_log['question'])

    def get_feedback(self, feedback):
        """
        Record the entity log in this episode.
        :param feedback：True or False
        """
        self.guess_right = feedback # whether the current game is win
        if feedback:
            self.ent_log[self.guess] += 1

    def question(self):
        ##### construct sample for RL agent #####
        if self.t != 0:
            # generate the transition
            s, a, r, s_, done = self.env_step()
            # store the transition
            self.IS.memory.store_transition(s, a, r, s_, done)

        # fixed opportunities for modules
        if self.t < self.switch_thres:
            q = self.info_seeking()
            self.module = "IS" 
        else:
            q = self.know_acqusition()
            self.module = "KA"

        self.module_switch = True if self.t == self.switch_thres - 2 else False
        done = True if self.t == self.T - 2 else False

        self.episode_log['question'].append(q)
        # RL agent learning 
        #if len(self.IS.memory) == BUFFER_SIZE:
        if len(self.IS.memory) > 4* BATCH_SIZE:
            self.IS.learn()
        return q, self.module_switch, done
    
    def env_step(self):
        # record old state
        s = copy.deepcopy(self.episode_state).reshape(-1)
        # update the state
        ques = self.episode_log['question'][-1]
        res = self.episode_log['response'][-1] # main observation
        a = action2index(self.n_predicates, self.n_entities, ques[0], ques[1])
        self.episode_state[a, 0] = 1
        self.episode_state[a, res] = 1
        s_ = self.episode_state.reshape(-1)
        # get the reward
        r = reward_func(self.guess_right, self.module_switch)

        return s, [a], [r], s_, self.module_switch

    def info_seeking(self):
        action = self.IS.choose_action(self.episode_state.reshape(-1))
        rel, rhs = index2action(self.n_predicates, self.n_entities, action)
        return [rel, rhs]

def reward_func(right=None, done=False):
    """
    Reward function for the RL agent.
    right: whether the final guess is right
    done: whether it is the termination of IS module
    """
    if done==True & right==True:
        r = 1
    else:
        r = 0
    return r


class DRQN(object):
    def __init__(self, n_predicates, n_entities, n_responses, action_size, state_size=256, lr=5e-4):
        # set random seed manually
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        self.n_actions = action_size
        self.state_size = state_size

        self.o_embed = Observation_Embedding(n_predicates, n_entities, n_responses).to(device)

        self.target_net = RecurrentQNetwork(self.o_embed.observation_size, action_size, self.state_size).to(device)
        self.eval_net = RecurrentQNetwork(self.o_embed.observation_size, action_size, self.state_size).to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory = ReplayBuffer(BUFFER_SIZE)                         # initialize memory
        self.optimizer = torch.optim.Adam([self.o_embed.parameters(), self.eval_net.parameters()], lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        """
        :param: x: of shape ()
        """
        x = torch.unsqueeze(torch.FloatTensor(x), 0)          # add batch size as 1
        # epsilon-greedy policy
        if np.random.uniform() < EPSILON:   # greedy
            with torch.no_grad():
                actions_value = self.eval_net.forward(x)
                action = torch.max(actions_value, 1)[1].data.numpy()
                action = action[0] 
        else:                               # random
            action = np.random.randint(0,  self.n_actions)
            action = action 
        return action

    def store_transition(self, s, a, r, s_, done):
        """
        Store the transition.
        :param s, s_: of shape (4*n_actions, )
        :param a: [idx]
        """
        self.memory.store_transition(s, a, r, s_, done)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % FREEZE_INTERVAL == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_, _ = self.memory.sample(BATCH_SIZE)
        b_s = torch.FloatTensor(b_s).to(device)
        b_a = torch.LongTensor(b_a).to(device)
        b_r = torch.FloatTensor(b_r).to(device)
        b_s_ = torch.FloatTensor(b_s_).to(device)


        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)                      # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()                         # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class AgentDRQN(Agent):
    """
    Deep recurrent Q-learning agent.
    """
    def __init__(self, dataset: InternalKB, switch_thres, n_chances=20):
        super(AgentDQN, self).__init__(dataset, n_chances, switch_thres)
        self.n_responses = 4
        self.IS = DRQN(self.n_predicates, self.n_entities, self.n_responses, self.n_actions)

    def reset(self):
        """
        Reset agent state and episode (when new episode starts). 
        """
        self.t = 0
        self.module = "IS" 
        self.module_switch = False
        self.guess_right = False
        self.episode_log = {'question':list(), 'response':list()}

    def update_response(self, response):
        """
        Append the lastest response.
        """
        self.episode_log['response'].append(response)
        self.t += 1
        assert len(self.episode_log['response']) == len(self.episode_log['question'])

    def get_feedback(self, feedback):
        """
        Record the entity log in this episode.
        :param feedback：True or False
        """
        self.guess_right = feedback # whether the current game is win
        if feedback:
            self.ent_log[self.guess] += 1

    def question(self):
        ##### construct sample for RL agent #####
        if self.t != 0:
            # generate the transition
            s, a, r, s_, done = self.env_step()
            # store the transition
            self.IS.memory.store_transition(s, a, r, s_, done)

        # fixed opportunities for modules
        if self.t < self.switch_thres:
            q = self.info_seeking()
            self.module = "IS" 
        else:
            q = self.know_acqusition()
            self.module = "KA"

        self.module_switch = True if self.t == self.switch_thres - 2 else False
        done = True if self.t == self.T - 2 else False

        self.episode_log['question'].append(q)
        # RL agent learning 
        #if len(self.IS.memory) == BUFFER_SIZE:
        if len(self.IS.memory) > 4*BATCH_SIZE:
            self.IS.learn()
        return q, self.module_switch, done
    
    def env_step(self):
        # record old state
        s = copy.deepcopy(self.episode_state).reshape(-1)
        # update the state
        ques = self.episode_log['question'][-1]
        res = self.episode_log['response'][-1]     # main observation
        a = action2index(self.n_predicates, self.n_entities, ques[0], ques[1])
        self.episode_state[a, 0] = 1
        self.episode_state[a, res] = 1
        s_ = self.episode_state.reshape(-1)
        # get the reward
        r = reward_func(self.guess_right, self.module_switch)

        return s, [a], [r], s_, self.module_switch

    def info_seeking(self):
        action = self.IS.choose_action(self.episode_state.reshape(-1))
        rel, rhs = index2action(self.n_predicates, self.n_entities, action)
        return [rel, rhs]

