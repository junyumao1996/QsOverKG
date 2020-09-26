import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import copy
import os
from .toy_models import Agent
import time
from datasets import InternalKB
from torch.optim import Adam
from .wolpdpg import Model
from .embedder import KBC_Embed, Embed_Config
from external_agents.DDPGs.util import *
from .utils import *


def init_parser():
    parser = argparse.ArgumentParser(description='WOLP_DDPG')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--max-episode-length', type=int, default=20, metavar='M', help='maximum length of an episode (default: 1440)')   # change this one
    parser.add_argument('--load', default=False, metavar='L', help='load a trained model')
    parser.add_argument('--load-model-dir', default='', metavar='LMD', help='folder to load trained models from')
    parser.add_argument('--gpu-ids', type=int, default=[1], nargs='+', help='GPUs to use [-1 CPU only]')
    parser.add_argument('--gpu-nums', type=int, default=1, help='#GPUs to use (default: 1)')
    parser.add_argument('--id', default='0', type=str, help='experiment id')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--k_ratio', default=0.025, type=float, help='k ratio')

    parser.add_argument('--hidden1_a', default=1024, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2_a', default=128, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--hidden1_c', default=32, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2_c', default=16, type=int, help='hidden num of second fully connect layer')

    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--c_lr', default=0.001, type=float, help='critic net learning rate')    # original: 0.001
    parser.add_argument('--p_lr', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=30000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=int(3e5), type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau-update', default=0.001, type=float, help='moving average for target network')   # 0.001
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay for L2 Regularization loss')   # original: 1e-5

    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.5, type=float, help='noise sigma')   # 0.5
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--epsilon', default=int(5e5), type=int, help='Linear decay of exploration policy')   # int(5e5)

    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--normalize', default=False, type=bool, help='normalize action space')
    return parser


class AgentWolDDPGSA(Agent):
    """
    IS: Wolpertinger DDPG agent.
    """
    def __init__(self, dataset: InternalKB, switch_thres, config, n_chances=20, lr=5e-4, load_path=None, train_mode=True, is_update=False, exp_path=None):
        self.n_responses = 3 
        super(AgentWolDDPGSA, self).__init__(dataset, n_chances, switch_thres)
        self.config = config
        self.config_add()

        self.Embedder = KBC_Embed(self.kb.get_indicator_set(), self.kb.to_skip, config.nb_action)
        self.is_update = is_update
        self.exp_dir = exp_path
        self.action_embed_update(n_epoch=300, first_update=True)
 
        self.IS = Model(self.state_size, config.train_args, self.action_embedding)
        self.n_actions = self.n_predicates * self.n_entities
        self.train_mode = train_mode

        if load_path != None:
            # load pre-trained model
            self.IS.load_weights(load_path)
            self.action_embedding = np.load(os.path.join(load_path, 'action.npy'))
            self.IS.update_action_space(self.action_embedding)
            self.train_mode = False
            self.is_update = False
            print("Load model sucessfully")

    def save_model(self, exp_path):
        self.IS.save_model(exp_path)

    def config_add(self):
        """
        Append env-specific info to config. 
        """
        pass

    def action_embed_update(self, frequency=20000, n_epoch=20, first_update=False):
        t_total = self.t_IS + self.t_KA
        if (self.is_update and t_total % frequency < self.T) or first_update:
            self.Embedder.update_examples(self.kb.get_indicator_set())
            _ = self.Embedder.train(n_epoch)
            self.action_embedding = self.Embedder.action_embedding_save(self.exp_dir)
            try:
                self.IS.update_action_space(self.action_embedding)
            except:
                pass
            print("Action Space updated")

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
        self.t_last = 0                          # last time posterior get updated
        self.posterior = np.ones(self.n_entities) / self.n_entities
        self.posterior_last = np.ones(self.n_entities) / self.n_entities
        self.state_size = len(self.posterior)
        self.state = np.zeros(self.state_size)

        try:
            self.IS.reset_when_start(self.state.reshape(-1))
        except:
            pass

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
        if self.t != 0 and self.t <= self.switch_thres:
            # construct sample for RL agent 
            o, a, r, o_, done = self.env_step()
            self.IS.update(o, a, r, o_, done, self.t_IS, self.train_mode)

        if self.module == "IS":
            q = self.info_seeking()
            self.module = self.Balancer.give_module_name(self.t, self.posterior)
            self.module_switch = True if self.module == 'KA' else False
        else:
            q = self.know_acqusition()
            self.module_switch = False

        self.module_switch = True if self.t == self.switch_thres - 1 else False
        done = True if self.t == self.T else False

        if done and self.guess_right:
            # update the multi-nouli parameters
            self.KB_update_periodic(self.guess, self.episode_log['question'], self.episode_log['response'], 1)   # 10000
            self.action_embed_update(self.T * 2000)
            self.balancer_update_check(self.T * 2000)       

        if done == False:
            self.episode_log['question'].append(q)
            self.action_mask[action2index(self.n_predicates, self.n_entities, q[0], q[1])] = 0.0

        return q, self.module_switch, done
    
    def env_step(self):
        # record old state
        s = copy.deepcopy(self.posterior_last)

        # get action index
        ques = self.episode_log['question'][-1]
        res = self.episode_log['response'][-1]            
        a = action2index(self.n_predicates, self.n_entities, ques[0], ques[1])

        # record current state
        s_ = copy.deepcopy(self.posterior)
        
        # get the reward
        r = reward_func(self.guess_right, self.module_switch)
        r_intri = reward_func_intrinsic(self.posterior, self.posterior_last, ratio=10) if self.config.intr_reward == True else 0.0
        r += r_intri

        return s, a, r, s_, self.module_switch

    def info_seeking(self):
        self.action, self.action_idx = self.IS.get_action(self.state.reshape(-1), self.t_IS, self.action_mask)
        rel, rhs = index2action(self.n_predicates, self.n_entities, self.action_idx)
        return [rel, rhs]