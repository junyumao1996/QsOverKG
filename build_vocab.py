import json
import os
import argparse
import numpy as np


file_path = '../data/FB15k-237/entity2wikidata.json'
dataset = json.load(open(file_path, 'r')) 
print("Number of entities:", len(dataset.keys()))
entity = np.random.choice(list(dataset.keys()))
print(entity)
print(dataset[entity])
