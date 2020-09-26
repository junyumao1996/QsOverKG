import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import random
import numpy as np
import math
import copy
import os
from .toy_models import Agent
from .icm import IntrinsicCuriosityModule
from external_agents.Bases.BaseAgent import BaseAgent
from external_agents.Bases.utils.hyperparameters import Config
from external_agents.Bases.utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory
from datasets import InternalKB
from .utils import *


class DQN(nn.Module):
    def __init__(self, num_features, num_actions, config):
        super(DQN, self).__init__()

        self.num_features = num_features
        self.num_hidden1 = config.n_hidden1
        self.num_hidden2 = config.n_hidden2
        self.num_actions = num_actions

        self.hidden1 = nn.Linear(self.num_features, self.num_hidden1)
        self.hidden2 = nn.Linear(self.num_hidden1, self.num_hidden2)
        self.fc3 = nn.Linear(self.num_hidden2, self.num_actions)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.fc3(x)
        return x

    def init(self):
        self.hidden1.weight.data.normal_(0, 0.1)
        self.hidden2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, load_path=None):
        super(Model, self).__init__()
        self.config = config
        self.device = config.device

        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START

        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA

        self.static_policy = static_policy
        self.num_feats = env.state_dim
        self.num_actions = env.action_dim
        self.env = env

        self.declare_networks()

        if type(load_path) != type(None):
            self.load_model(load_path)

        self.target_model.load_state_dict(self.model.state_dict())

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        self.icm_model.to(self.device)

        self.eta_icm = config.eta_icm
        self.lambda_icm = config.lambda_icm
        # self.icm_fwd_criterion = nn.KLDivLoss()
        self.icm_fwd_criterion = nn.MSELoss()

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm_model.parameters()), lr=self.lr)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions, self.config)
        self.target_model = DQN(self.num_feats, self.num_actions, self.config)
        self.icm_model = IntrinsicCuriosityModule(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if(len(self.nstep_buffer) < self.nsteps):
            return
        
        R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)

        # random transition batch is taken from experience replay memory
        # transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + (self.num_feats,)

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape).to(self.device)
        #print(batch_state)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1).to(self.device)
        #print(batch_action)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool).to(self.device)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            # print('shape', max_next_q_values.shape, batch_reward.shape, non_final_next_states.shape,non_final_mask.shape)
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        # loss = self.huber(diff)
        # loss = diff**2
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.huber(diff).squeeze() * weights
        else:
            loss = self.huber(diff)
        loss = loss.mean()

        pred_action, pred_next_state, next_state = self.icm_model(batch_state[non_final_mask, :], non_final_next_states, batch_action[non_final_mask])
        fwd_loss = self.icm_fwd_criterion(pred_next_state, next_state).mean()
        curiosity_loss = self.beta_icm * fwd_loss
        loss += curiosity_loss

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()
        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()

    def get_action(self, s, eps=0.1, action_mask=None):
        """
        mask: np.array of shape (1, self.num_actions)
        """
        if type(action_mask) == type(None):
            mask = torch.ones((1, self.num_actions)).to(self.device)
        else:
            mask = torch.tensor(action_mask).view(1, -1).to(self.device)

        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                q_values = self.model(X) * mask
                a = q_values.max(1)[1].view(1, 1)
                return a.item()
            else:
                for i in range(200):
                    a = np.random.randint(0, self.num_actions)
                    if mask[0, a] == 1.0:
                        break
                return a

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def icm_reward(self, state, action, next_state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.device)
        pred_logits, pred_phi, phi = self.icm_model(state, next_state, action)
        fwd_loss = self.icm_fwd_criterion(pred_phi, phi) / 2
        intrinsic_reward = self.eta_icm * fwd_loss.detach()
        return intrinsic_reward

    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)
    
    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'eval_net.pkl')))

    def save_model(self, exp_path):
        torch.save(self.model.state_dict(), os.path.join(exp_path, 'eval_net.pkl'))


class AgentDQNICM(Agent):
    """
    DQN agent with designed state. 
    """
    def __init__(self, dataset: InternalKB, switch_thres, config, n_chances=20, lr=5e-4, load_path=None, mode='train'):
        super(AgentDQNICM, self).__init__(dataset, n_chances, switch_thres)
        self.n_actions = self.n_predicates * self.n_entities
        self.config = config
        self.config_add(switch_thres)
        self.build_RL_env() 
        self.IS = Model(env=self.env, config=self.config)
        self.state = np.zeros((self.n_actions, 4))  # representation of current question&response history (i.e., state)
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
        self.env = env_wrapper(self.state_size, self.n_actions)

    def reset(self):
        """
        Reset agent state and episode (when new episode starts). 
        """
        self.t = 0
        self.module = "IS" 
        self.module_switch = False
        self.guess_right = False
        self.episode_log = {'question':list(), 'response':list()}

        self.action_mask = np.ones(self.n_predicates * self.n_entities)
        self.t_last = 0                    # last time posterior get updated
        self.posterior = np.ones(self.n_entities) / self.n_entities
        self.posterior_last = np.ones(self.n_entities) / self.n_entities
        self.state_size = len(self.posterior)
        self.state = np.zeros(self.state_size )

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
        _, _ = self.guess_generate_incremental(self, truncate=False)

    def question(self):
        ##### construct sample for RL agent #####
        if self.t != 0 and self.t <= self.switch_thres:
            # generate the transition
            o, a, r, o_, done = self.env_step()
            o_ = None if done else o_
            # print(o, a, r, o_, done)
            self.IS.update(o, a, r, o_, self.t_IS)

        if self.t < self.switch_thres:
            epsilon = self.config.epsilon_by_frame(self.t_IS) if self.mode == 'train' else 0.
            q = self.info_seeking(epsilon)
            self.module = "IS" 
        else:
            q = self.know_acqusition()
            self.module = "KA"

        self.module_switch = True if self.t == self.switch_thres - 1 else False
        done = True if self.t == self.T else False

        if done and self.guess_right:
            # update the multi-nouli parameters
            # self.KB_update(self.episode_log['question'], self.episode_log['response'])
            self.KB_update_periodic(self.guess, self.episode_log['question'], self.episode_log['response'], 50000)

        if done == False:
            self.episode_log['question'].append(q)
            self.action_mask[action2index(self.n_predicates, self.n_entities, q[0], q[1])] = 0.0

        return q, self.module_switch, done
    
    def env_step(self):
        # record old state
        s = copy.deepcopy(self.posterior_last)
        # update the state
        ques = self.episode_log['question'][-1]
        res = self.episode_log['response'][-1]                # main observation
        a = action2index(self.n_predicates, self.n_entities, ques[0], ques[1])
        # self.state[a, 0] = 1
        # self.state[a, res + 1] = 1
        s_ = copy.deepcopy(self.posterior)
        # get the reward
        r = reward_func(self.guess_right, self.module_switch)
        intr_reward = self.IS.icm_reward(s, a, s_)
        r += intr_reward.item()

        return s, a, r, s_, self.module_switch

    def info_seeking(self, epsilon):
        action = self.IS.get_action(self.state.reshape(-1), epsilon, self.action_mask)
        rel, rhs = index2action(self.n_predicates, self.n_entities, action)
        return [rel, rhs]

class env_wrapper():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def reset(self):
        pass

    def step(self):
        pass