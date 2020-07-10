import numpy as np
from numba import jit
import os 
from pathlib import Path
import json
import pickle
from typing import Dict, Tuple, List, Union
import copy

def dataset_to_tensor(data_path, file_name):
    """
    Load data files to numpy array.
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

    dataset_tensor = np.zeros((n_ent, n_rel, n_ent), dtype=np.int8)
    examples = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
    for lhs, rel, rhs in examples:
        dataset_tensor[lhs, rel, rhs] = 1
    dataset['tensor'] = dataset_tensor

    return dataset

def dataset_to_dict(data_path, file_name):
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

    dataset_dict = {}
    examples = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
    for lhs, rel, rhs in examples:
        dataset_dict[(lhs, rel, rhs)] = 1 # binary (1: yes, 0: no) fact for external knowledage base (in simulator)
    dataset['tensor'] = dataset_dict

    return dataset

def dataset_split(dataset, split_ratio=0.8):
    """
    Extract partial dataset based on split_ratio.
    This is for internal knowledage base (in agent).
    """
    dataset_s = copy.copy(dataset)
    tensor_s = {}
    for key in dataset['tensor'].keys():
        if np.random.uniform(0, 1) < split_ratio:
            tensor_s[key] = [10, 1, 1]    # (#yes, #no, #unknown) initialization of the known fact
    dataset_s['tensor'] = tensor_s
    return dataset_s


class InternalKB(object):
    """
    KB of mutlinoulli distribution for agent.
    """
    def __init__(self, dataset):
        self.data = dataset['tensor']
        self.data_info = dataset['info']
        self.n_entities = dataset['info']['n_ent']
        self.n_predicates = dataset['info']['n_rel']
        self.entry_init = [10, 1, 1]

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


class ExternalKB(object):
    """
    KB of binary facts for simulator.
    """
    def __init__(self, dataset):
        self.data = dataset['tensor']
        self.data_info = dataset['info']
        self.n_entities = dataset['info']['n_ent']
        self.n_predicates = dataset['info']['n_rel']

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