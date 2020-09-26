import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import math
import copy
import os
from .toy_models import Agent
from .icm import IntrinsicCuriosityModule
from .utils import *
from external_agents.Bases.DQN import Model as DQN_Agent
from external_agents.Bases.utils.ReplayMemory import RecurrentExperienceReplayMemory
from datasets import InternalKB

class DRQN(nn.Module):
    def __init__(self, n_predicates, n_entities, n_responses, num_actions, gru_size=32, bidirectional=False):
        super(DRQN, self).__init__()

        self.input_shape = 3
        self.num_actions = num_actions
        self.gru_size = gru_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        
        self.gru = nn.GRU(n_entities - 1, self.gru_size, num_layers=1, batch_first=True,
                          bidirectional=bidirectional)
        self.fc1 = nn.Linear(self.gru_size, 64)
        self.fc2 = nn.Linear(64, self.num_actions)

    def forward(self, x, hx=None):

        batch_size = x.size(0)
        sequence_length = x.size(1)

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
        self.state_dim = env.state_dim

        self.eta_icm = config.eta_icm
        self.beta_icm = config.beta_icm
        # self.icm_fwd_criterion = nn.KLDivLoss()
        self.icm_fwd_criterion = nn.MSELoss()
        super(Model, self).__init__(static_policy, env, config)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm_model.parameters()), lr=self.lr)
        self.icm_model.to(self.device)

        if type(load_path) != type(None):
            self.load_model(load_path)

        self.reset_hx()

    def declare_networks(self):
        self.model = DRQN(self.n_predicates, self.n_entities, self.n_responses, self.action_dim)
        self.target_model = DRQN(self.n_predicates, self.n_entities, self.n_responses, self.action_dim)
        self.icm_model = IntrinsicCuriosityModule(self.state_dim, self.action_dim)

    def declare_memory(self):
        self.memory = RecurrentExperienceReplayMemory(self.experience_replay_size, self.sequence_length)

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (self.batch_size, self.sequence_length) + (self.num_feats,)

        batch_state = torch.tensor(batch_state, dtype=torch.float).view(shape).to(self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).view(self.batch_size, self.sequence_length, -1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).view(self.batch_size, self.sequence_length).to(self.device)
        # get set of next states for end of each sequence
        batch_next_state = tuple(
            [batch_next_state[i] for i in range(len(batch_next_state)) if (i + 1) % (self.sequence_length) == 0])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool).to(self.device)
        try:  # sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], 
                                                 dtype=torch.float).unsqueeze(dim=1).to(self.device)
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
        loss = loss.mean()

        state_size = batch_state.shape[2]
        batch_state = batch_state[non_final_mask, :, :].view(-1, state_size)
        batch_action = batch_action[non_final_mask, :, :].view(-1, 1)
        non_final_next_states = non_final_next_states.view(-1, state_size)
        # print("shape", non_final_mask.shape, batch_state.shape, batch_action.shape, non_final_next_states.shape)
        pred_action, pred_next_state, next_state = self.icm_model(batch_state, non_final_next_states, batch_action)
        fwd_loss = self.icm_fwd_criterion(pred_next_state, next_state).mean()
        curiosity_loss = self.lambda_icm * fwd_loss
        loss += curiosity_loss
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
                X = torch.tensor([self.seq], device=self.device, dtype=torch.float).to(self.device)
                a, _ = self.model(X)
                a = a[:, -1, :]      # select last element of seq
                a *= mask
                a = a.max(1)[1]
                return a.item()
            else:
                for i in range(10000):
                    a = np.random.randint(0, self.num_actions)
                    if mask[0, a] == 1.0:
                        break
                return a

    def icm_reward(self, state, action, next_state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.device)
        pred_logits, pred_phi, phi = self.icm_model(state, next_state, action)
        fwd_loss = self.icm_fwd_criterion(pred_phi, phi) / 2
        intrinsic_reward = self.eta_icm * fwd_loss.detach()
        return intrinsic_reward

    def reset_hx(self):
        self.seq = [np.zeros(self.num_feats) for j in range(self.sequence_length)]

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'eval_net.pkl')))

    def save_model(self, exp_path):
        torch.save(self.model.state_dict(), os.path.join(exp_path, 'eval_net.pkl'))


class AgentDRQNICM(Agent):
    """
    Deep recurrent Q-learning agent.
    """
    def __init__(self, dataset: InternalKB, switch_thres, config, n_chances=20, lr=5e-4, load_path=None, mode='train'):
        self.n_responses = 3 
        super(AgentDRQNICM, self).__init__(dataset, n_chances, switch_thres)
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
        # obs = np.zeros((3 + self.n_entities,), dtype=np.float)
        # obs[:2] = np.array(self.episode_log['question'][-1])
        # obs[2] = self.episode_log['response'][-1]
        # obs[3:] = np.ones(self.n_entities, dtype=np.float) / self.n_entities
        # self.observation = copy.deepcopy(obs)
        self.action_mask = np.ones(self.n_predicates * self.n_entities)
        self.t_last = 0           # last time posterior get updated
        self.posterior = np.ones(self.n_entities) / self.n_entities
        self.posterior_last = np.ones(self.n_entities) / self.n_entities
        self.observation = copy.deepcopy(self.posterior)

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
        self.posterior_last = copy.deepcopy(self.posterior)
        _, _ = self.guess_generate_incremental(self, truncate=True)

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
        # self.posterior = prob

    def question(self):
        # record rewards
        if self.t != 0 and self.t <= self.switch_thres:
            # generate the transition
            o, a, r, o_, done = self.env_step()
            o_ = None if done else o_
            # print(o, a, r, o_, done, '\n')
            self.IS.update(o, a, r, o_, self.t_IS)

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
            # self.KB_update_periodic(self.guess, self.episode_log['question'], self.episode_log['response'], 50000)

        if done == False:
            self.episode_log['question'].append(q)
            q_flat = action2index(self.n_predicates, self.n_entities, q[0], q[1])
            self.episode_log['question_flat'].append([q_flat])
            self.action_mask[q_flat] = 0.0

        return q, self.module_switch, done

    def env_step(self):
        # last state
        # o = np.zeros((3 + self.n_entities,), dtype=np.float)
        # o[:2] = np.array(self.episode_log['question'][-2]).astype(np.float)
        # o[2] = float(self.episode_log['response'][-2])
        # o[3:] = self.posterior_last
        o = copy.deepcopy(self.posterior_last)
        # last action
        a = self.episode_log['question_flat'][-1]
        # new state
        # o_ =  np.zeros((3 + self.n_entities,), dtype=np.float)
        # o_[:2] = np.array(self.episode_log['question'][-1]).astype(np.float)
        # o_[2] = float(self.episode_log['response'][-1])
        # o_[3:] = self.posterior
        o_ = copy.deepcopy(self.posterior)
        self.observation = copy.deepcopy(o_)
        # get the reward
        r = reward_func(self.guess_right, self.module_switch)
        intr_reward = self.IS.icm_reward(o, a, o_)
        r += intr_reward.item()

        return o, a[0], r, o_, self.module_switch

    
    def info_seeking(self, epsilon):
        action = self.IS.get_action(self.observation, epsilon, self.action_mask)
        rel, rhs = index2action(self.n_predicates, self.n_entities, action)
        return [rel, rhs]

    def guess_generate(self, method='uniform', normalize=True):
        self.guess_generate_incremental(self)
        return self.guess, np.max(self.posterior)

class env_wrapper():
    def __init__(self, action_dim, n_predicates, n_entities, n_responses
    ):
        self.state_dim = n_entities  # 3 + n_entities 
        self.action_dim = action_dim
        self.n_predicates = n_predicates
        self.n_entities = n_entities 
        self.n_responses = n_responses

    def reset(self):
        pass

    def step(self):
        pass