# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import os

import torch
from torch import nn
from torch import optim
import numpy as np

from external_agents.kbc.datasets import Dataset_simple
from external_agents.kbc.models import CP, ComplEx
from external_agents.kbc.regularizers import F2, N3
from external_agents.kbc.optimizers import KBCOptimizer
from agents.utils import action2index


class Embed_Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = 'ComplEx'     #'ComplEx', 'CP' is default
        self.regularizer = 'N3'
        self.optimizer = 'Adagrad'
        self.max_epochs = 1        # 100
        self.valid = 5
        self.rank = 8       
        self.batch_size = 32       # 100
        self.reg = 1e-2
        self.init = 1e-3
        self.learning_rate = 1e-1
        self.decay1 = 0.9
        self.decay2 = 0.999

class KBC_Embed():
    def __init__(self, data, to_skip, nb_action=8):
        args = Embed_Config()
        args.rank = nb_action
        self.to_skip = to_skip
        self.dataset = Dataset_simple(data, to_skip)
        self.examples = torch.from_numpy(self.dataset.get_train().astype('int64'))
        self.embed_size = args.rank
        self.model_name = args.model
        self.n_predicates, self.n_entities = int(self.dataset.get_shape()[1]/2), self.dataset.get_shape()[0]
        self.model = {
            'CP': lambda: CP(self.dataset.get_shape(), args.rank, args.init),
            'ComplEx': lambda: ComplEx(self.dataset.get_shape(), int(args.rank), args.init),
            }[args.model]()
        self.model.to(args.device)
        self.regularizer = {
            'F2': F2(args.reg),
            'N3': N3(args.reg),
            }[args.regularizer]
        self.optim_method = {
            'Adagrad': lambda: optim.Adagrad(self.model.parameters(), lr=args.learning_rate),
            'Adam': lambda: optim.Adam(self.model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
            'SGD': lambda: optim.SGD(self.model.parameters(), lr=args.learning_rate)
            }[args.optimizer]()
        self.optimizer = KBCOptimizer(self.model, self.regularizer, self.optim_method, args.batch_size, False)
        self.max_epochs = args.max_epochs
    
    def update_examples(self, data):
        self.dataset = Dataset_simple(data, self.to_skip)
        self.examples = torch.from_numpy(self.dataset.get_train().astype('int64'))

    def train(self, epochs=None):
        curve = {'MRR':[] , 'hits':[]}
        n_epoch = self.max_epochs if epochs is None else epochs
        for e in range(n_epoch):
            cur_loss = self.optimizer.epoch(self.examples)
            if (e + 1) % 3 == 0:
                train = avg_both(*self.dataset.eval(self.model, 50000))
                curve['MRR'].append(train['MRR'])
                curve['hits'].append(train['hits@[1,3,10]'].numpy())
        print("Action Embedding Completed")
        return curve

    def embedding_output(self):
        if self.model_name == "ComplEx":
            rel = self.model.embeddings[1]
            ent = self.model.embeddings[0]
            return rel.weight.data[:self.n_predicates, :self.embed_size], ent.weight.data[:, :self.embed_size],
        else:
            rel = self.model.rel
            ent = self.model.rhs
            return rel.weight.data[:self.n_predicates, :], ent.weight.data

    def embedding_save(self, save_path):
        rel, ent = self.embedding_output()
        np.save(os.path.join(save_path, 'rel.npy'), rel.cpu().numpy())
        np.save(os.path.join(save_path, 'ent.npy'), ent.cpu().numpy())

    def action_embedding_save(self, save_path):
        rel, ent = self.embedding_output()
        rel, ent = rel.cpu().numpy(), ent.cpu().numpy()
        action_embed = np.zeros((self.n_predicates * self.n_entities, self.embed_size*2))
        for p in range(self.n_predicates):
            for o in range(self.n_entities):
                action_idx = action2index(self.n_predicates, self.n_entities, p, o)
                action_embed[action_idx, :] = np.concatenate([rel[p, :], ent[o, :]])
        np.save(os.path.join(save_path, 'action.npy'), action_embed)
        return action_embed

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}