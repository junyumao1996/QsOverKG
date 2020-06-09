import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path
import json
import pickle

def dataset_to_tensor(data_path, file_name):
    dataset = {}

    entities_to_id = json.load(open("ent2id.json"))
    relations_to_id = json.load(open("rel2id.json"))
    n_ent = len(entities_to_id)
    n_rel = len(relations_to_id)
    del entities_to_id, relations_to_id
    dataset['info'] = {}
    dataset['info']['n_ent'] = n_ent
    dataset['info']['n_rel'] = n_rel

    probas = pickle.load(open(os.path.join(data_path, "probas.pickle"), 'rb'))
    dataset['info']['probas'] = probas

    dataset_tensor = np.zeros(shape=(n_ent, n_rel, n_ent), dtype=int)
    examples = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
    for lhs, rel, rhs in examples:
        dataset_tensor[lhs, rel, rhs] = 1
    dataset['tensor'] = dataset_tensor

    return dataset