import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math
import copy
import os
from .toy_models import Agent
from external_agents.Bases.DQN import Model as DQN_Agent
from external_agents.Bases.utils.ReplayMemory import RecurrentExperienceReplayMemory
from datasets import InternalKB
from .utils import *


class Observation_Embedding(nn.Module):
    """
    Embedding for constructing the observation.
    """
    def __init__(self, n_predicates, n_entities, n_responses, r_size=8):
        super(Observation_Embedding, self).__init__()
        
        self.q_embed = Question_Embedding(n_predicates, n_entities)      # question embedding
        self.r_embed = nn.Embedding(n_responses, r_size)                 # response embedding
        self.observation_size = self.q_embed.question_size + r_size
        self.init_weights()

    def forward(self, x):
        """
        x: of size (...,  3:[rel, rhs, response]), idealy (batch size, sequence len, 3:[rel, rhs, response])
        """
        q = x[..., :-1]
        r = x[..., -1]
        x_q = self.q_embed(q)
        x_r = self.r_embed(r)
        return torch.cat([x_q, x_r], dim=-1)

    def init_weights(self):
        initrange = 0.1
        self.r_embed.weight.data.uniform_(-initrange, initrange)


class Question_Embedding(nn.Module):
    """
    Embedding representation of questions.
    """
    def __init__(self, n_predicates, n_entities, rel_size=16, rhs_size=8):
        super(Question_Embedding, self).__init__()
        self.rel_embed = nn.Embedding(n_predicates, rel_size)               # predicate embedding
        self.rhs_embed = nn.Embedding(n_entities, rhs_size)                 # entity embedding
        self.linear = nn.Linear(rel_size + rhs_size, 16)
        self.question_size = 16
        self.init_weights()

    def forward(self, x):
        """
        x: of size (..., 2:[rel, rhs]), idealy (batch size, sequence len, 2:[rel, rhs])
        """
        rel = x[..., 0]
        rhs = x[..., 1]
        x_rel = self.rel_embed(rel)
        x_rhs = self.rhs_embed(rhs)
        return self.linear(torch.cat([x_rel, x_rhs], dim=-1))

    def init_weights(self):
        initrange = 0.1
        self.rel_embed.weight.data.uniform_(-initrange, initrange)
        self.rhs_embed.weight.data.uniform_(-initrange, initrange)

class DRQN(nn.Module):
    def __init__(self, n_predicates, n_entities, n_responses, num_actions, gru_size=32, bidirectional=False):
        super(DRQN, self).__init__()

        self.input_shape = 3
        self.num_actions = num_actions
        self.gru_size = gru_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.o_embed = Observation_Embedding(n_predicates, n_entities, n_responses) # input size = (..., 3)
        self.gru = nn.GRU(self.o_embed.observation_size, self.gru_size, num_layers=1, batch_first=True,
                          bidirectional=bidirectional)
        self.fc1 = nn.Linear(self.gru_size, 64)
        self.fc2 = nn.Linear(64, self.num_actions)

    def forward(self, x, hx=None):
        x = self.o_embed(x)
        batch_size = x.size(0)
        sequence_length = x.size(1)

        x = x.view((-1, self.o_embed.observation_size))

        # format outp for batch first gru
        feats = x.view(batch_size, sequence_length, -1)

        if hx is None:
            out, hidden = self.gru(feats)
        else:
            out, hidden = self.gru(feats, hx)

        x = F.relu(self.fc1(out))
        x = self.fc2(x)
        # x = torch.tanh(x)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1 * self.num_directions, batch_size, self.gru_size, dtype=torch.float)

    def sample_noise(self):
        pass

class Model(DQN_Agent):
    """
    DRQN agent.
    """
    def __init__(self, static_policy=False, env=None, config=None, load_path=None):
        self.sequence_length = config.SEQUENCE_LENGTH
        self.n_predicates = env.n_predicates + 1   # (+ 1 extra null response at t=0)
        self.n_entities = env.n_entities + 1
        self.n_responses = env.n_responses + 1
        self.action_dim = env.action_dim
        super(Model, self).__init__(static_policy, env, config)

        if type(load_path) != type(None):
            self.load_model(load_path)
        # self.target_model.load_state_dict(self.model.state_dict())
        self.reset_hx()

    def declare_networks(self):
        self.model = DRQN(self.n_predicates, self.n_entities, self.n_responses, self.action_dim)
        self.target_model = DRQN(self.n_predicates, self.n_entities, self.n_responses, self.action_dim)

    def declare_memory(self):
        self.memory = RecurrentExperienceReplayMemory(self.experience_replay_size, self.sequence_length)

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (self.batch_size, self.sequence_length) + (self.num_feats,)

        batch_state = torch.LongTensor(batch_state).view(shape).to(self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).view(self.batch_size, self.sequence_length, -1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).view(self.batch_size, self.sequence_length).to(self.device)
        # get set of next states for end of each sequence
        batch_next_state = tuple(
            [batch_next_state[i] for i in range(len(batch_next_state)) if (i + 1) % (self.sequence_length) == 0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool).to(self.device)
        try:  # sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], 
                                                 dtype=torch.long).unsqueeze(dim=1).to(self.device)
            non_final_next_states = torch.cat([batch_state[non_final_mask, 1:, :], non_final_next_states], dim=1)
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights


    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        current_q_values, _ = self.model(batch_state)
        current_q_values = current_q_values.gather(2, batch_action).squeeze()

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=self.device,
                                            dtype=torch.float)
            if not empty_next_state_values:
                max_next, _ = self.target_model(non_final_next_states)
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = batch_reward + ((self.gamma ** self.nsteps) * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)

        # # mask first half of losses
        # split = self.sequence_length // 2
        # mask = torch.zeros(self.sequence_length, device=self.device, dtype=torch.float)
        # mask[split:] = 1.0
        # mask = mask.view(1, -1)
        # loss *= mask
        loss = loss.mean()

        return loss

    def get_action(self, s, eps=0.1, action_mask=None):
        """
        mask: np.array of shape (1, self.num_actions)
        """
        if type(action_mask) == type(None):
            mask = torch.ones((1, self.num_actions)).to(self.device)
        else:
            mask = torch.tensor(action_mask).view(1, -1).to(self.device)

        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(s)
            if np.random.random() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([self.seq], device=self.device, dtype=torch.long).to(self.device)
                self.model.sample_noise()
                a, _ = self.model(X)
                a = a[:, -1, :]  # select last element of seq
                a *= mask
                a = a.max(1)[1]
                return a.item()
            else:
                for i in range(10000):
                    a = np.random.randint(0, self.num_actions)
                    if mask[0, a] == 1.0:
                        break
                return a

    def reset_hx(self):
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'eval_net.pkl')))

    def save_model(self, exp_path):
        torch.save(self.model.state_dict(), os.path.join(exp_path, 'eval_net.pkl'))


class AgentDRQN(Agent):
    """
    Deep recurrent Q-learning agent.
    """
    def __init__(self, dataset: InternalKB, switch_thres, config, n_chances=20, lr=5e-4, load_path=None, mode='train'):
        self.n_responses = 3 
        super(AgentDRQN, self).__init__(dataset, n_chances, switch_thres)
        self.n_actions = self.n_predicates * self.n_entities
        self.config = config
        self.config_add(switch_thres)
        self.build_RL_env()     
        self.IS = Model(env = self.env, config=self.config)  
        self.mode = mode

        if load_path != None:
            # load pre-trained model
            self.IS.load_model(load_path)
            print("Load model sucessfully")

    def save_model(self, exp_path):
        self.IS.save_model(exp_path)

    def config_add(self, switch_thres):
        """
        Append env-specific info to config. 
        """
        self.config.SEQUENCE_LENGTH = switch_thres

    def build_RL_env(self):
        """
        Build a gym-like RL env.
        """
        self.env = env_wrapper(self.n_actions, self.n_predicates, self.n_entities, self.n_responses)

    def reset(self):
        """
        Reset agent state and episode (when new episode starts). 
        """
        self.t = 0
        self.module = "IS" 
        self.module_switch = False
        self.guess_right = False 
        self.episode_log = {'question':[[self.n_predicates, self.n_entities]], 'question_flat':list(), 'response':[self.n_responses]}  # intialize a first observation
        # assemble the first observation
        obs = np.zeros((3,), dtype=np.int32)
        obs[:-1] = np.array(self.episode_log['question'][-1])
        obs[-1] = self.episode_log['response'][-1]
        self.observation = copy.deepcopy(obs)
        self.action_mask = np.ones(self.n_predicates * self.n_entities)
        self.posterior = 0

    def update_response(self, response):
        """
        Append the lastest response.
        """
        self.episode_log['response'].append(response)
        self.t += 1
        assert len(self.episode_log['response']) == len(self.episode_log['question'])
        if self.module == "IS":
            self.t_IS += 1
        else:
            self.t_KA += 1

    def get_feedback(self, feedback, prob):
        """
        Record the entity log in this episode.
        :param feedback：True or False
        """
        self.guess_right = feedback # whether the current game is win
        if feedback:
            self.ent_log[self.guess] += 1
            # update the multi-nouli parameters
            # self.KB_update(self.episode_log['question'], self.episode_log['response'], True)
        self.posterior = prob

    def question(self):
        # record rewards
        if self.t != 0 and self.t <= self.switch_thres:
            # generate the transition
            o, a, r, o_, done = self.env_step()
            o_ = None if done else o_
            # print(o, a, r, o_, done)
            self.IS.update(o, a, r, o_, self.t_IS)

        # fixed opportunities for modules
        if self.t < self.switch_thres:
            epsilon = self.config.epsilon_by_frame(self.t_IS) if self.mode == 'train' else 0.
            q = self.info_seeking(epsilon)
            self.module = "IS" 
        else:
            q = self.know_acqusition()
            self.module = "KA"

        done = True if self.t == self.T else False   # indicator for game over
        self.module_switch = True if self.t == self.switch_thres - 1 else False

        if done and self.guess_right:
            # update the multi-nouli parameters
            self.KB_update(self.episode_log['question'], self.episode_log['response'], True)

        if done == False:
            self.episode_log['question'].append(q)
            q_flat = action2index(self.n_predicates, self.n_entities, q[0], q[1])
            self.episode_log['question_flat'].append([q_flat])
            self.action_mask[q_flat] = 0.0

        return q, self.module_switch, done

    def env_step(self):
        # last state
        o =  np.zeros((3,),dtype=np.int32)
        o[:-1] = np.array(self.episode_log['question'][-2])
        o[-1] = self.episode_log['response'][-2]
        # last action
        a = self.episode_log['question_flat'][-1]
        # new state
        o_ =  np.zeros((3,),dtype=np.int32)
        o_[:-1] = np.array(self.episode_log['question'][-1])
        o_[-1] = self.episode_log['response'][-1]
        self.observation = copy.deepcopy(o_)
        # get the reward
        r = reward_func(self.guess_right, self.module_switch)
        # r = reward_func_dense(self.guess_right, self.module_switch, self.posterior)

        return o, a[0], r, o_, self.module_switch

    
    def info_seeking(self, epsilon):
        action = self.IS.get_action(self.observation, epsilon, self.action_mask)
        rel, rhs = index2action(self.n_predicates, self.n_entities, action)
        return [rel, rhs]


class env_wrapper():
    def __init__(self, action_dim, n_predicates, n_entities, n_responses
    ):
        self.state_dim = 3
        self.action_dim = action_dim
        self.n_predicates = n_predicates
        self.n_entities = n_entities 
        self.n_responses = n_responses

    def reset(self):
        pass

    def step(self):
        pass