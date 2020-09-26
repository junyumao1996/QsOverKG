import numpy as np
from sknetwork.ranking import PageRank, BiPageRank
from datasets import ExternalKB


class Simulator():
    def __init__(self, dataset: ExternalKB):
        self.kb = dataset                        # dict with keys of [lhs, pred, rhs]
        self.data_info = dataset.data_info
        self.n_entities = dataset.n_entities
        self.n_predicates = dataset.n_predicates

    def calc_propularity(self, method='number'):
        if method == 'number':
            popularity = self.data_info['probas']['both']
        elif method == 'uniform':
            popularity = np.ones(self.n_entities)
            popularity = popularity / np.sum(popularity)
        elif method == 'pagerank':
            adjacency = np.sum(self.kb, axis=[0,1])
            adjacency = adjacency[adjacency > 0].astype(np.int32)
            popularity = PageRank.fit_transform(adjacency)
        else:
            raise Exception

        return popularity

    def entity_select(self):
        """
        Select the target entity.
        """
        pop_distri = self.calc_propularity(method='number')
        target = np.random.choice(range(self.n_entities), size=1, p=pop_distri)
        self.target_entity = target[0]

    def get_fact_set(self):
        return set([tuple(key) for key in self.kb.data.keys()])

    def response(self, question, method='fixed', unknown_prob=0.1, false_prob=0.05):
        """
        :param question: 3-element list (subj, rel, obj) with a Nan in guessed position
        :return response: 0: yes, 1: no, 2: unknown
        """
        lhs = self.target_entity
        rel = question[0]
        rhs = question[1]
        assert unknown_prob + false_prob < 1
        if method == 'fixed':
            rand = np.random.rand(1)[0]
            if rand < unknown_prob:
                response = 2
            elif rand > 1-false_prob:
                response = 1 - self.kb.lookup((lhs, rel, rhs))
            else:
                response = self.kb.lookup((lhs, rel, rhs))
        elif method == 'variable':
            pass
        else:
            raise Exception
        return int(response)

    def guess_check(self, guess, verbose=False):
        right = guess == self.target_entity
        if verbose:
            comment = "Congratulations! You throw the right guess." if right else "Sorry! Maybe next time..."
            print(comment)
        return right, self.target_entity


        


