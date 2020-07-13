import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
import time
from typing import Dict, Tuple, List, Union
from abc import ABC, abstractmethod
from kbc.models import CP, ComplEx
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer
from datasets import InternalKB
#from .utils import EntropyIS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(ABC):
    def __init__(self, dataset: InternalKB, n_chances=20, switch_thres: int=None):
        self.kb = dataset
        self.n_entities = self.kb.n_entities
        self.n_predicates = self.kb.n_predicates
        self.T = n_chances
        self.ent_log = np.zeros(self.n_entities)
        self.switch_thres = switch_thres

        self.KA = CP((self.n_entities, self.n_predicates,  self.n_entities), rank=1000)
        
        self.reset()

    @abstractmethod
    def question(self):
        """
        Compose/generate the question. 
        (to be overwritten by the child)
        """
        pass

    @abstractmethod
    def info_seeking(self):
        """
        Information Seeking (IS) module. 
        :return : a list of [relation, object]
        (to be overwritten by the child)
        """
        pass

    def save_model(self, exp_path):
        pass

    def reset(self):
        """
        Reset agent state and episode (when new episode starts). 
        """
        self.t = 0
        self.module = "IS" 
        self.module_switch = False
        self.episode_log = {'question':list(), 'response':list()}

    def know_acqusition(self):
        """
        Knowledge Acquision (KA) module.
        :return : a list of [relation, object]
        """
        pass

    def KB_update(self, q_seq: List[Tuple[int, int, int]], r_seq: List[int]):
        """
        Update the internal KB from questions and responses log.
        :param q_seq: question sequence as [(lhs, rhs, rhs)]
        :param r_seq: response sequence as [response]
        """
        assert len(q_seq) == len(r_seq)
        for q, r in zip(q_seq, r_seq):
            if q in self.kb.data.keys():
                self.kb.data[q][r] += 1
            else:
                self.kb.data[q] = self.kb.entry_init
                self.kb.data[q][r] += 1

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
        if feedback:
            self.ent_log[self.guess] += 1

    def guess_generate(self, method='log', normalize=False):
        """
        Generate the final guess given the reponses.
        """
        # compute independent likelihood
        q_log = np.array(self.episode_log['question'])
        a_log = np.array(self.episode_log['response'])
        prob = np.ones((self.n_entities, self.t))

        # t0 = time.time()
        for t in range(self.t): 
            prob[:, t] = self.kb.slice_prob((range(self.n_entities), [q_log[t, 0]], [q_log[t, 1]], [a_log[t]])).ravel()
            
        # t1 = time.time()
        # print("t1-t0", t1-t0)

        prob_log = np.log(prob)          # size (n_ent, t)
        lik_log = np.sum(prob_log, axis=1)

        # compute prior
        if method == 'log':
            prior = self.ent_log + 1    # laplace smoothing
            prior = prior / np.sum(prior)
        elif method == 'uniform':
            prior = np.ones(self.n_entities) / self.n_entities
        else:
            raise Exception
        prior_log  = np.log(prior)

        # bayes' rule
        posterior = prior_log + lik_log
        posterior += np.min(posterior)
        max_post = np.max(posterior)
        self.guess = np.random.choice(np.where(posterior == max_post)[0])
        # print(posterior)

        # t2 = time.time()
        # print("t2-t1", t2-t1)

        if normalize:
            prob = np.exp(max_post)/np.sum(np.exp(posterior))
        else:
            prob = None
        return self.guess, prob

    def guess_generate_torch(self, method='log'):
        """
        Generate the final guess given the reponses (torch tensor based).
        """
        # compute independent likelihood
        q_log = np.array(self.episode_log['question'])
        a_log = np.array(self.episode_log['response'])
        prob = torch.ones((self.n_entities, self.t), requires_grad=False).to(device)
        for t in range(self.t):
            prob[:, t] = torch.tensor(self.kb.slice((range(self.n_entities), [q_log[t, 0]], [q_log[t, 1]], [a_log[t]])).ravel(), requires_grad=False).to(device)
        prob_log = torch.log(prob)             # size (n_ent, t)
        lik_log = torch.sum(prob_log, dim=1)

        # compute prior
        if method == 'log':
            prior = torch.tensor(self.ent_log, requires_grad=False).to(device) + 1    # laplace smoothing
            prior = prior / torch.sum(prior)
        elif method == 'uniform':
            prior = torch.ones(self.n_entities, requires_grad=False).to(device) / self.n_entities
        else:
            raise Exception
        prior_log  = torch.log(prior)

        # bayes' rule
        posterior = prior_log + lik_log
        self.guess = torch.argmax(posterior)

        return self.guess


class AgentRandom(Agent):
    """
    Random question picking (fixed opportunities)
    """
    def __init__(self, dataset: InternalKB, switch_thres, n_chances=20, load_path=None):
        super(AgentRandom, self).__init__(dataset, n_chances, switch_thres)

        if load_path != None:
            pass

    def question(self):
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
        return q, self.module_switch, done

    def info_seeking(self):
        rel = np.random.randint(self.n_predicates)
        rhs = np.random.randint(self.n_entities)
        return [rel, rhs]



class AgentEntropy(Agent):
    """
    Question picking based on entropy ranking (fixed opportunities)
    """
    def __init__(self, dataset: InternalKB, switch_thres, n_chances=20):
        super(AgentRandom, self).__init__(dataset, n_chances, switch_thres)

    def question(self):
        # fixed opportunities for modules
        if self.t < self.switch_thres:
            q = self.info_seeking()
            self.module = "IS" 
        else:
            q = self.know_acqusition()
            self.module = "KA"

        self.module_switch = True if self.t == self.switch_thres-1 else False
        done = True if self.t == self.T - 2 else False

        self.episode_log['question'].append(q)
        return q, self.module_switch, done

    def info_seeking(self):
        pass
        #return EntropyIS(self.kb_multinoulli)


