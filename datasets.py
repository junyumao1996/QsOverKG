import numpy as np
# from numba import jit
import os 
from pathlib import Path
import json
import pickle
from typing import Dict, Tuple, List, Union
import copy
import math
# set random seed of numpy
np.random.seed(23)

ENTRY_INIT_YES = [20, 1, 2]
ENTRY_INIT_NO = [1, 1, 1]

# def dataset_to_tensor(data_path, file_names: List):
#     """
#     Load data files to numpy array.
#     """
#     dataset = {}
#     entities_to_id = json.load(open(os.path.join(data_path, "ent2id.json")))
#     relations_to_id = json.load(open(os.path.join(data_path, "rel2id.json")))
#     n_ent = len(entities_to_id)
#     n_rel = len(relations_to_id)
#     del entities_to_id, relations_to_id
#     dataset['info'] = {}
#     dataset['info']['n_ent'] = n_ent
#     dataset['info']['n_rel'] = n_rel

#     probas = pickle.load(open(os.path.join(data_path, "probas.pickle"), 'rb'))
#     dataset['info']['probas'] = probas

#     dataset_tensor = np.zeros((n_ent, n_rel, n_ent), dtype=np.int8)
#     for file_name in file_names:
#         examples = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
#         for lhs, rel, rhs in examples:
#             dataset_tensor[lhs, rel, rhs] = 1
#     dataset['tensor'] = dataset_tensor
#     return dataset

def dataset_to_dict(data_path, file_names: List):
    """
    Load data files to dict (sparse representation of tensor to save memory).
    This is for external knowledage base (in simulator). 
    """
    dataset = {}
    entities_to_id = json.load(open(os.path.join(data_path, "ent2id.json")))
    relations_to_id = json.load(open(os.path.join(data_path, "rel2id.json")))
    n_ent = len(entities_to_id)
    n_rel = len(relations_to_id)
    del entities_to_id, relations_to_id

    dataset['info'] = {}
    dataset['info']['n_ent'] = n_ent
    dataset['info']['n_rel'] = n_rel

    probas = pickle.load(open(os.path.join(data_path, "probas.pickle"), 'rb'))
    dataset['info']['probas'] = probas

    # load triples 
    dataset_dict = {}
    for file_name in file_names:
        examples = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
        # print(len(examples), file_name)
        for lhs, rel, rhs in examples:
            dataset_dict[(lhs, rel, rhs)] = 1      # binary (1: yes, 0: no) fact for external knowledage base (in simulator)
    dataset['tensor'] = dataset_dict               # wrap in a competible name
    # exit()

    # load to-skips
    inp_f = open(os.path.join(data_path, 'to_skip.pickle'), 'rb')
    to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
    inp_f.close()
    dataset['info']['to_skip'] = to_skip

    return dataset

def dataset_split(dataset, split_ratio=0.8):
    """
    Extract partial dataset based on split_ratio.
    This is for internal knowledage base (in agent).
    """
    dataset_s = copy.copy(dataset)
    tensor_s = {}
    dataset_example = []
    for key in dataset['tensor'].keys():
        if np.random.uniform(0, 1) < split_ratio:
            tensor_s[key] = copy.deepcopy(ENTRY_INIT_YES)    # (#yes, #no, #unknown) initialization of the known fact
            dataset_example.append(list(key))
    dataset_s['tensor'] = tensor_s
    return dataset_s, np.array(dataset_example)


class InternalKB(object):
    """
    KB of mutlinoulli distribution for agent.
    """
    def __init__(self, dataset):
        self.data = dataset['tensor']
        self.data_info = dataset['info']
        self.n_entities = dataset['info']['n_ent']
        self.n_predicates = dataset['info']['n_rel']
        self.to_skip = dataset['info']['to_skip']
        self.entry_init_yes = ENTRY_INIT_YES
        self.entry_init_no = ENTRY_INIT_NO
        self.sure_condition = lambda key : sum(self.data[key]) > 10 
        # self.indicator_condition = lambda key : self.data[key][2]/np.sum(self.data[key]) < 0.3
        self.indicator_condition = lambda key : self.data[key][0]/sum(self.data[key]) > 0.5
        self.initialize_sets()

    def initialize_sets(self):
        """
        Initialize the set of entries which agent is pretty sure about by the given limited dataset. 
        """
        self.sure_set = {}                            # collection of keys consider as sure entry (have a certain level of responses)
        self.indicator_set = set()                    # collection of keys consider as yes/no facts
        for lhs in range(self.n_entities):
            self.sure_set[str(lhs)] = set()
        for key in self.data.keys():
            lhs, rel, rhs = key
            self.sure_set[str(lhs)].add((rel, rhs))       # add into sure set
            self.indicator_set.add(key)                   # add into indicator set

    def slice(self, shape: Tuple[List[int], List[int], List[int], List[int]]):
        """
        Slice the orginal kb (in response count). 
        """
        m, n, l, h = shape
        tensor = np.ones((len(m), len(n), len(l), len(h)), dtype=np.int32)
        for i, lhs in enumerate(m):
            for j, rel in enumerate(n):
                for k, rhs in enumerate(l):
                    if (lhs, rel, rhs) in self.data.keys():
                        for g, count in enumerate(h):
                            tensor[i, j, k, g] = np.array(self.data[(lhs, rel, rhs)][count])
        return tensor

    def slice_prob(self, shape: Tuple[List[int], List[int], List[int], List[int]]):
        """
        Slice the kb and get the multinoulli params.
        """
        m, n, l, h = shape
        tensor = np.ones((len(m), len(n), len(l), len(h))) / 3
        for i, lhs in enumerate(m):
            for j, rel in enumerate(n):
                for k, rhs in enumerate(l):
                    if (lhs, rel, rhs) in self.data.keys():
                        for g, count in enumerate(h):
                            prob = self.data[(lhs, rel, rhs)][count] / np.sum(self.data[(lhs, rel, rhs)])
                            tensor[i, j, k, g] = prob
        return tensor

    def append(self, key: Tuple, response):
        """
        One response append. 
        """
        if key in self.data.keys():
            self.data[key][response] += 1
        else:
            self.data[key] = copy.deepcopy(self.entry_init_no)
            self.data[key][response] += 1
        # update into sure set
        self._update_sure_set(key)
        # edit indicator entries (for KA)
        self._update_indicator_set(key)

    def _update_sure_set(self, key):
        """
        Update the sure set for uncertainty-based choosing.
        """
        lhs, rel, rhs = key
        if self.sure_condition(key):
            self.sure_set[str(lhs)].add((rel, rhs))

    def _update_indicator_set(self, key):
        """
        Update the indicator set for value-based choosing.
        """
        # print(key, self.data[key])
        if self.indicator_condition(key):
            try:
                self.indicator_set.add(key)
            except:
                pass
        else:
            try:
                self.indicator_set.remove(key)
            except:
                pass

    def get_indicator_set(self):
        """
        Output indicator set as np.array of examples. (for KA training)
        """
        return np.array([list(key) for key in self.indicator_set])

    def get_fact_set(self):
        """
        Same as indicator set but in different form. (for KBR computation)
        """
        return set([key for key in self.indicator_set])

    def output_unsure_set(self, entity, k, max_itr=300):
        """
        Obtain the candidate entity set based on uncertainty. 
        k: size of the generated set
        """
        sure_set_entity = self.sure_set[str(entity)]
        unsure_set_entity = []
        sure_set_candidate = []
        for i in range(max_itr):
            rel = np.random.randint(self.n_predicates)
            rhs = np.random.randint(self.n_entities)
            if (rel, rhs) not in sure_set_entity:
                unsure_set_entity.append([entity, rel, rhs])
                if len(unsure_set_entity) >= k:
                    break
            else: 
                sure_set_candidate.append([entity, rel, rhs])
        
        if len(unsure_set_entity) < k:
            n_requires = k - len(unsure_set_entity) 
            unsure_set_entity.extend(sure_set_candidate[:n_requires])

        return np.array(unsure_set_entity, dtype=np.int)


def uncertainty(number):
    return 1./math.sqrt(number)


class ExternalKB(object):
    """
    KB of binary facts for simulator.
    """
    def __init__(self, dataset):
        self.data = dataset['tensor']
        self.data_info = dataset['info']
        self.n_entities = dataset['info']['n_ent']
        self.n_predicates = dataset['info']['n_rel']
        self.n_examples = len(dataset['tensor'].keys())

    def slice(self, shape: Tuple[List[int], List[int], List[int]]):
        """
        Slice the orginal kb. 
        """
        m, n, l = shape
        tensor = np.zeros((len(m), len(n), len(l)), dtype=np.int32)
        for i, lhs in enumerate(m):
            for j, rel in enumerate(n):
                for k, rhs in enumerate(l):
                    if (lhs, rel, rhs) in self.data.keys():
                            tensor[i, j, k] = self.data[(lhs, rel, rhs)]
        return tensor

    def lookup(self, shape: Tuple[int, int, int]):
        """
        One-element lookup of external fact base.
        """
        if shape in self.data.keys():
            return 0  # yes
        else:
            return 1  # no
