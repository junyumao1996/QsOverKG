#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple

def read_triples(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples

train_triples = read_triples('train.tsv')
valid_triples = read_triples('dev.tsv')
test_triples = read_triples('test.tsv')

all_triples = train_triples + valid_triples + test_triples

entity_set = {s for s, _, _ in all_triples} | {o for _, _, o in all_triples}
predicate_set = {p for _, p, _ in all_triples}

entity_lst = sorted(entity_set)
predicate_lst = sorted(predicate_set)

entity_to_idx = {entity: i for i, entity in enumerate(entity_lst)}
predicate_to_idx = {predicate: i for i, predicate in enumerate(predicate_lst)}

with open("train", "w") as f:
    f.writelines([f"{entity_to_idx[s]}\t{predicate_to_idx[p]}\t{entity_to_idx[o]}\n" for s, p, o in train_triples])

with open("valid", "w") as f:
    f.writelines([f"{entity_to_idx[s]}\t{predicate_to_idx[p]}\t{entity_to_idx[o]}\n" for s, p, o in valid_triples])

with open("test", "w") as f:
    f.writelines([f"{entity_to_idx[s]}\t{predicate_to_idx[p]}\t{entity_to_idx[o]}\n" for s, p, o in test_triples])
