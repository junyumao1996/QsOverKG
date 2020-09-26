import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# from numba import jit
import time
from typing import Dict, Tuple, List, Union
from abc import ABC, abstractmethod

from datasets import InternalKB
from .ka_models import KA_Agent
from .balancer import Balancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(ABC):
    def __init__(self, dataset: InternalKB, n_chances=20, switch_thres=None):
        self.kb = dataset
        self.n_entities = self.kb.n_entities
        self.n_predicates = self.kb.n_predicates
        self.T = n_chances
        self.ent_log = np.zeros(self.n_entities)
        self.k_buffer = []            # knowledge buffer to contain responses temporally
        if type(switch_thres) == int:
            self.t_low = switch_thres - 1
            self.t_high = switch_thres - 1
            self.switch_thres = switch_thres
        else:
            self.t_low = switch_thres[0] - 1
            self.t_high = switch_thres[1] - 1  
            self.switch_thres = switch_thres[1]

        self.IS = None
        self.KA = KA_Agent(self.kb.get_indicator_set(), self.kb.to_skip)
        self.Balancer = Balancer(10000, self.t_low, self.t_high)

        self.t_IS = 0    # total game steps of IS
        self.t_KA = 0    # total game steps of KA
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
        self.guess_right = False
        self.episode_log = {'question':list(), 'response':list()}
        self.t_last = 0           # last time posterior get updated
        self.posterior = None

    def know_acqusition(self, k=100):
        """
        Knowledge Acquision (KA) module.
        :return : a list of [relation, object]
        """
        t0 = time.time()
        unsure_set = self.kb.output_unsure_set(self.guess, k)
        q = self.KA.one_circle(unsure_set)
        return q

    def ka_dataset_update_check(self, frequency=20000):
        t_total = self.t_IS + self.t_KA
        if t_total % frequency < self.T:
            curve = self.KA.dataset_reload_train(self.kb.get_indicator_set(), self.kb.to_skip)
            print("KA module Updated")

    def balancer_update_check(self, frequency=20000):
        t_total = self.t_IS + self.t_KA
        if (t_total + 1) % frequency == 0:
            acc_test = self.Balancer.train()
            print("Balancer Updated - Acc: {}".format(acc_test))

    def KB_update(self, q_seq, r_seq: List[int], truncate=False):
        """
        Update the internal KB from questions and responses log.
        :param q_seq: question sequence as [(lhs, rhs, rhs)]
        :param r_seq: response sequence as [response]
        """
        assert len(q_seq) == len(r_seq)
        if truncate:
            q_seq = q_seq[1:]
            r_seq = r_seq[1:]
        for q, r in zip(q_seq, r_seq):
            q = copy.deepcopy(q)
            q.insert(0, self.guess)
            q = tuple(q)
            self.kb.append(q, r)

    def KB_update_periodic(self, target_entity, q_seq, r_seq, update_freq=50000, truncate=False):
        """
        Update the internal KB from questions and responses log.
        :param q_seq: question sequence as [(lhs, rhs, rhs)]
        :param r_seq: response sequence as [response]
        """
        assert len(q_seq) == len(r_seq)
        if truncate:
            q_seq = q_seq[1:]
            r_seq = r_seq[1:]
        for q, r in zip(q_seq, r_seq):
            q = copy.deepcopy(q)
            q.insert(0, target_entity)
            q = tuple(q)
            self.k_buffer.append([q, r])
        if len(self.k_buffer) > update_freq:
            for data in self.k_buffer:
                self.kb.append(data[0], data[1])
            self.k_buffer = []

    def update_response(self, response):
        """
        Append the lastest response.
        """
        self.episode_log['response'].append(response)
        assert len(self.episode_log['response']) == len(self.episode_log['question'])
        self.t += 1
        if self.module == "IS" or self.module_switch:
            self.t_IS += 1
        else:
            self.t_KA += 1

    def get_feedback(self, feedback, prob):
        """
        Record the entity log in this episode.
        :param feedback：True or False
        """
        self.guess_right = feedback 
        # add episodic (posterior, rusult) datapoint to balancer
        data_point = np.array(np.hstack([self.posterior.reshape(-1), np.array([float(feedback)])]))
        self.Balancer.add_datapoint(data_point)

    def guess_generate(self, method='uniform', normalize=True):
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

        prob_log = np.log(prob)             # size (n_ent, t)
        lik_log = np.sum(prob_log, axis=1)

        # compute prior
        if method == 'log':
            prior = self.ent_log + 1        # laplace smoothing
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

        if normalize:
            posterior = np.exp(posterior)
            self.posterior = posterior/np.sum(posterior)
            prob = np.max(self.posterior)
        else:
            prob = None
        return self.guess, prob

    def guess_generate_incremental(self, normalize=True, truncate=False):
        """
        Generate the final guess given the reponses (incremental).
        """
        # compute independent likelihood
        if truncate == False:
            q_log = np.array(self.episode_log['question'][self.t_last:])
            a_log = np.array(self.episode_log['response'][self.t_last:])
        else:
            q_log = np.array(self.episode_log['question'][self.t_last + 1:])
            a_log = np.array(self.episode_log['response'][self.t_last + 1:])

        t_during = q_log.shape[0]
        prob = np.ones((self.n_entities, t_during))

        for t in range(t_during):
            prob[:, t] = self.kb.slice_prob((range(self.n_entities), [q_log[t, 0]], [q_log[t, 1]], [a_log[t]])).ravel()
            
        prob_log = np.log(prob)              # size (n_ent, t)
        lik_log = np.sum(prob_log, axis=1)

        # incremential Bayes prior
        if self.posterior is None:
            prior = np.ones(self.n_entities) / self.n_entities
        else:
            prior = self.posterior
        prior_log  = np.log(prior)

        # bayes' rule
        posterior = prior_log + lik_log
        posterior += np.min(posterior)
        max_post = np.max(posterior)
        self.guess = np.random.choice(np.where(posterior == max_post)[0])

        if normalize:
            posterior = np.exp(posterior)
            self.posterior = posterior/np.sum(posterior)
            prob = np.max(self.posterior)
        else:
            prob = None
        self.t_last = self.t
        return self.guess, prob

class AgentRandom(Agent):
    """
    Random question picking (fixed opportunities)
    """
    def __init__(self, dataset: InternalKB, switch_thres, n_chances=20, load_path=None):
        super(AgentRandom, self).__init__(dataset, n_chances, switch_thres)
        if load_path != None:
            pass

    def question(self):
        if self.module == "IS":
            q = self.info_seeking()
            self.module = self.Balancer.give_module_name(self.t, self.posterior)
            self.module_switch = True if self.module == 'KA' else False
            if self.t >= self.t_low - 1:
                self.guess_generate_incremental(self)
        else:
            q = self.know_acqusition()
            self.module_switch = False

        done = True if self.t == self.T else False

        if done and self.guess_right:
            # update the multi-nouli parameters
            self.KB_update(self.episode_log['question'], self.episode_log['response'])

        if done == False:
            self.episode_log['question'].append(q)

        self.ka_dataset_update_check(self.T * 2000)      # update KA per 2000 games
        self.balancer_update_check(self.T * 2000)        # train Balancer per 1000 games
        return q, self.module_switch, done

    def info_seeking(self):
        rel = np.random.randint(self.n_predicates)
        rhs = np.random.randint(self.n_entities)
        return [rel, rhs]

    def guess_generate(self, method='uniform', normalize=True):
        self.guess_generate_incremental(self)
        return self.guess, np.max(self.posterior)
    

class AgentRandom2(Agent):
    """
    Random question picking (fixed opportunities)
    """
    def __init__(self, dataset: InternalKB, switch_thres, n_chances=20, load_path=None):
        super(AgentRandom2, self).__init__(dataset, n_chances, switch_thres)
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

        self.module_switch = True if self.t == self.switch_thres - 1 else False
        done = True if self.t == self.T else False

        if done and self.guess_right:
            # update the multi-nouli parameters
            self.KB_update(self.episode_log['question'], self.episode_log['response'])

        if done == False:
            self.episode_log['question'].append(q)

        self.ka_dataset_update_check(self.T * 2000)  # update KA per 2000 games
        # self.balancer_update_check(self.T * 1000)    # train Balancer per 1000 games
        return q, self.module_switch, done

    def info_seeking(self):
        rel = np.random.randint(self.n_predicates)
        rhs = np.random.randint(self.n_entities)
        return [rel, rhs]
    
