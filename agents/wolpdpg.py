"""
Partial implementation of wolpertinger refers to https://github.com/ChangyWen/wolpertinger_ddpg and 
"""

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
from external_agents.DDPGs.ddpg import DDPG
from external_agents.DDPGs import action_space
from external_agents.DDPGs.util import *
from .utils import *

import argparse

criterion = nn.MSELoss()

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
    parser.add_argument('--tau-update', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay for L2 Regularization loss')   # original: 1e-5

    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.5, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--epsilon', default=int(5e5), type=int, help='Linear decay of exploration policy')  

    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--normalize', default=False, type=bool, help='normalize action space')

    parser.add_argument('--max-actions', default=200000, type=int, help='# max actions')
    parser.add_argument('--max-episode', type=int, default=200000, help='maximum #episode.')
    return parser

class WolpertingerAgent(DDPG):
    def __init__(self, nb_states, args, action_embeddings, k_ratio=0.1):
        nb_actions = action_embeddings.shape[1]
        self.experiment = args.id
        self.normalize = args.normalize
        # according to the papers, it can be scaled to hundreds of millions
        self.action_space = action_space.Custom_Space(action_embeddings, args.normalize)   # knn mapper
        self.k_nearest_neighbors = max(1, int(self.action_space.get_number_of_actions() * k_ratio))  # k
        super().__init__(args, nb_states, nb_actions, self.action_space.return_space_boundary())
        
    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def update_action_space(self, action_embeddings):
         self.action_space = action_space.Custom_Space(action_embeddings, self.normalize)

    def get_action_space(self):
        return self.action_space

    def wolp_action(self, s_t, proto_action, action_mask=None):
        # get the proto_action's k nearest neighbors
        raw_actions, actions, idx = self.action_space.search_point(proto_action, self.k_nearest_neighbors)

        if not isinstance(s_t, np.ndarray):
           s_t = to_numpy(s_t, gpu_used=self.gpu_used)
        # make all the state, action pairs for the critic
        s_t = np.tile(s_t, [raw_actions.shape[1], 1])

        s_t = s_t.reshape(len(raw_actions), raw_actions.shape[1], s_t.shape[1]) if self.k_nearest_neighbors > 1 \
            else s_t.reshape(raw_actions.shape[0], s_t.shape[1])
        raw_actions = to_tensor(raw_actions, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])
        s_t = to_tensor(s_t, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])

        # evaluate each pair through the critic
        actions_evaluation = self.critic([s_t, raw_actions])

        # assemble action mask
        if action_mask is None:
            mask = np.ones(actions_evaluation.shape)
        else:
            assert actions_evaluation.shape[0] == 1
            mask = np.array(action_mask[idx[0]]).reshape((1, -1, 1))   
        mask = to_tensor(mask, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])
        actions_evaluation -= torch.min(actions_evaluation)

        # find the index of the pair with the maximum value
        max_index = np.argmax(to_numpy(actions_evaluation * mask, gpu_used=self.gpu_used), axis=1)
        max_index = max_index.reshape(len(max_index),)

        raw_actions = to_numpy(raw_actions, gpu_used=self.gpu_used)
        # return the best action, i.e., wolpertinger action from the full wolpertinger policy
        if self.k_nearest_neighbors > 1:
            return raw_actions[[i for i in range(len(raw_actions))], max_index, :].reshape(len(raw_actions), -1), \
                   actions[[i for i in range(len(actions))], max_index, :].reshape(len(actions), -1), \
                    idx[[i for i in range(len(idx))], max_index]
        else:
            return raw_actions[max_index], actions[max_index], idx[max_index]

    def select_action(self, s_t, decay_epsilon=True, action_mask=None):
        # taking a continuous action from the actor
        proto_action = super().select_action(s_t, decay_epsilon)

        raw_wolp_action, wolp_action, idx = self.wolp_action(s_t, proto_action, action_mask)
        assert isinstance(raw_wolp_action, np.ndarray)
        self.a_t = raw_wolp_action[0]
        # return the best neighbor of the proto action, this is an action for env step
        return wolp_action[0], idx[0] 

    def random_action(self):
        proto_action = super().random_action()
        raw_action, action, idx = self.action_space.search_point(proto_action, 1)
        raw_action = raw_action[0]
        action = action[0]
        assert isinstance(raw_action, np.ndarray)
        self.a_t = raw_action
        return action[0], idx[0]

    def random_action_discrete(self, action_mask):
        for i in range(100):
            raw_action, action, idx = self.action_space.random_point()
            if action_mask[idx] == 1.0: 
                break
        self.a_t = raw_action
        return action, idx

    def select_target_action(self, s_t):
        proto_action = self.actor_target(s_t)   
        proto_action = to_numpy(torch.clamp(proto_action, -1.0, 1.0), gpu_used=self.gpu_used)
        raw_wolp_action, wolp_action, _ = self.wolp_action(s_t, proto_action)
        return raw_wolp_action

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        # the operation below of critic_target does not require backward_P
        next_state_batch = to_tensor(next_state_batch, volatile=True, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])
        next_wolp_action_batch = self.select_target_action(next_state_batch)

        next_q_values = self.critic_target([
            next_state_batch,
            to_tensor(next_wolp_action_batch, volatile=True, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]),
        ])


        # but it requires bp in computing gradient of critic loss
        next_q_values.volatile = False

        # next_q_values = 0 if is terminal states
        target_q_batch = to_tensor(reward_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]) + \
                         self.gamma * \
                         to_tensor(terminal_batch.astype(np.float64), gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0]) * \
                         next_q_values

        # Critic update
        self.critic.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.
        # self.critic_optim.zero_grad()

        state_batch = to_tensor(state_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])
        action_batch = to_tensor(action_batch, gpu_used=self.gpu_used, gpu_0=self.gpu_ids[0])
        q_batch = self.critic([state_batch, action_batch])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()  # computes gradients
        self.critic_optim.step()  # updates the parameters

        # Actor update
        self.actor.zero_grad()
        # self.actor_optim.zero_grad()

        # self.actor(to_tensor(state_batch)): proto_action_batch
        policy_loss = - self.critic([state_batch, self.actor(state_batch)])
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()


        # Target update
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)


class Model(WolpertingerAgent):
    """
    Wolpertinger DDPG agent, a further wrap of agent for consistent API with other agents.
    """
    def __init__(self, nb_states, args, action_embeddings, load_path=None, mode='train'):
        super(Model, self).__init__(nb_states, args, action_embeddings, args.k_ratio)
        self.k_ratio = args.k_ratio
        self.warmup = args.warmup
        self.is_training = True if mode =='train' else False

    def get_action(self, s_t, step, action_mask):
        # cost time less than 0.01
        if step <= self.warmup:
            action, action_idx = self.random_action_discrete(action_mask)
        else:
            action, action_idx = self.select_action(s_t, True, action_mask)
        return action, action_idx

    def reset_when_start(self, s):
        self.reset(s)

    def update(self, s, a, r, s_, done, step, train_mode=True):
        r_t = r
        s_t1 = s_
        # agent observe and update policy
        self.observe(r_t, s_t1, done)
        if step > self.warmup and train_mode:
            self.update_policy()
        if done:
            self.memory.append(
                    s_t1,
                    self.select_action(s_t1)[0],
                    0., True
                )

    def load_weights(self, load_path):
        if dir is None: return

        if self.gpu_used:
            # load all tensors to GPU (gpu_id)
            ml = lambda storage, loc: storage.cuda(self.gpu_ids)
        else:
            # load all tensors to CPU
            ml = lambda storage, loc: storage

        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(load_path), map_location=ml))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(load_path), map_location=ml))

    def save_model(self, exp_path):
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] > 0:
            with torch.cuda.device(self.gpu_ids[0]):
                torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(exp_path))
                torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(exp_path))
        elif len(self.gpu_ids) > 1:
            torch.save(self.actor.module.state_dict(), '{}/actor.pkl'.format(exp_path))
            torch.save(self.actor.module.state_dict(), '{}/critic.pkl'.format(exp_path))
        else:
            torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(exp_path))
            torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(exp_path))
        

class AgentWolDDPG(Agent):
    """
    IS: Wolpertinger DDPG agent.
    """
    def __init__(self, dataset: InternalKB, switch_thres, config, n_chances=20, lr=5e-4, load_path=None, train_mode=True):
        self.n_responses = 3 
        super(AgentWolDDPG, self).__init__(dataset, n_chances, switch_thres)
        self.config = config
        self.config_add()
        self.IS = Model(np.prod(self.state_size), config.train_args, config.action_embedding)
        self.IS.reset_when_start(self.state.reshape(-1))
        self.n_actions = self.n_predicates * self.n_entities
        self.train_mode = train_mode

        if load_path is not None:
            # load pre-trained model
            self.IS.load_weights(load_path)
            self.train_mode = False
            print("Load model sucessfully")

    def save_model(self, exp_path):
        self.IS.save_model(exp_path)

    def config_add(self):
        """
        Append env-specific info to config. 
        """
        pass

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
        self.state = np.zeros(tuple(self.state_size))
        self.action_mask = np.ones(self.n_predicates * self.n_entities)
        self.posterior = 0
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

    def get_feedback(self, feedback, prob):
        """
        Record the entity log in this episode.
        :param feedback：True or False
        """
        self.guess_right = feedback # whether the current game is win
        if feedback:
            self.ent_log[self.guess] += 1

        self.posterior = prob

    def question(self):
        ##### construct sample for RL agent #####
        if self.t != 0 and self.t <= self.switch_thres:
            # generate the transition
            o, a, r, o_, done = self.env_step()
            self.IS.update(o, a, r, o_, done, self.t_IS)

        # fixed opportunities for modules
        if self.t < self.switch_thres:
            q = self.info_seeking()
            self.module = "IS" 
        else:
            q = self.know_acqusition()
            self.module = "KA"

        self.module_switch = True if self.t == self.switch_thres - 1 else False
        done = True if self.t == self.T else False

        if done and self.guess_right:
            # update the multi-nouli parameters
            self.KB_update(self.episode_log['question'], self.episode_log['response'])

        if done == False:
            self.episode_log['question'].append(q)
            self.action_mask[action2index(self.n_predicates, self.n_entities, q[0], q[1])] = 0.0

        return q, self.module_switch, done
    
    def env_step(self):
        # record old state
        s = copy.deepcopy(self.state).reshape(-1)
        # update the state
        ques = self.episode_log['question'][-1]
        res = self.episode_log['response'][-1]                # main observation
        a = self.action
        a_idx = self.action_idx
        self.state[a_idx, 0] = 1
        self.state[a_idx, res + 1] = 1
        s_ = self.state.reshape(-1)
        # get the reward
        r = reward_func(self.guess_right, self.module_switch)

        return s, a, r, s_, self.module_switch

    def info_seeking(self):
        self.action, self.action_idx = self.IS.get_action(self.state.reshape(-1), self.t_IS, self.action_mask)
        rel, rhs = index2action(self.n_predicates, self.n_entities, self.action_idx)
        return [rel, rhs]
