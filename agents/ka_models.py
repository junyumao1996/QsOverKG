from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
from torch import optim
import numpy as np
import wandb

from external_agents.kbc.datasets import Dataset_simple
from external_agents.kbc.models import CP, ComplEx
from external_agents.kbc.regularizers import F2, N3
from external_agents.kbc.optimizers import KBCOptimizer


class KA_Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = 'ComplEx'   # 'ComplEx' 'CP' 
        self.regularizer = 'N3'
        self.optimizer = 'Adagrad'
        self.max_epochs = 30    # 30
        self.inc_epochs = 4 
        self.valid = 5
        self.rank = 16          # Complex will double this rank
        self.batch_size = 128   # 100
        self.reg = 1e-2
        self.init = 1e-3
        self.learning_rate = 1e-1
        self.decay1 = 0.9
        self.decay2 = 0.999

class KA_Agent():
    def __init__(self, data, to_skip):
        args = KA_Config()
        self.args = args
        self.dataset = Dataset_simple(data, to_skip)
        self.examples = torch.from_numpy(self.dataset.get_train().astype('int64'))
        self.model = {
            'CP': lambda: CP(self.dataset.get_shape(), args.rank, args.init),
            'ComplEx': lambda: ComplEx(self.dataset.get_shape(), args.rank, args.init),
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
        self.n_update = 0

    def dataset_reload_train(self, dataset, to_skip):
        self.dataset = Dataset_simple(dataset, to_skip)
        self.examples = torch.from_numpy(self.dataset.get_train().astype('int64'))
        curve = self.train()
        return curve

    def train(self):
        if self.n_update == 0:
            n_epoch = self.args.max_epochs
        else: 
            n_epoch = self.args.inc_epochs

        curve = {'MRR':[] , 'hits':[]}
        for e in range(n_epoch):
            cur_loss = self.optimizer.epoch(self.examples)
            self.n_update += 1
            # if e == n_epoch - 1:
            #     train = avg_both(*self.dataset.eval(self.model, 50000))
            #     curve['MRR'].append(train['MRR'])
            #     curve['hits'].append(train['hits@[1,3,10]'].numpy())
        return curve

    def _score(self, queries):
        """
        queries: np.array of shape (n_query, 3)
        """
        x = torch.from_numpy(queries.astype('int64')).to(self.args.device)
        scores = self.model.score(x)
        return scores

    def _rank(self, queries, k=1):
        """
        k: top k entries with highest score
        """
        scores = self._score(queries)
        if k == 1:
            top_idx = scores.max(0)[1]
        else:
            top_idx = scores.sort(0, True)[1][:k, :]          # of shape (k, 1)
        return top_idx.view(-1).cpu().numpy()                 # of shape (k)

    def one_circle(self, candidate_set, action_mask=None):
        """
        unsure_set: np.array of candidate entries
        """
        idx = self._rank(candidate_set)
        candidate = candidate_set[idx].reshape(-1)
        rel, rhs = candidate[1], candidate[2]
        return [rel, rhs]

        
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
