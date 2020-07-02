import numpy as np
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.ranking import PageRank, BiPageRank
from datasets import ExternalKB


class Simulator():
    def __init__(self, dataset: ExternalKB):
        self.kb = dataset                        # dict with keys of [lhs, pred, rhs]
        self.data_info = dataset.data_info
        self.n_entities = dataset.n_entities
        self.n_predicates = dataset.n_predicates

    def calc_propularity(self, method='uniform'):
        if method == 'number':
            popularity = self.data_info['probas']['both']
        elif method == 'uniform':
            popularity = np.ones(self.n_entities)
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
        pop = self.calc_propularity(method='uniform') 
        pop_distri = pop / np.sum(pop)
        target = np.random.choice(range(self.n_entities), size=1, p=pop_distri)
        self.target_entity = target[0]

    def response(self, question, method='fixed', unknown_prob=0.1):
        """
        :param question: 3-element list (subj, rel, obj) with a Nan in guessed position
        :return response: 0: yes, 1: no, 2: unknown
        """
        lhs = self.target_entity 
        rel = question[0]
        rhs = question[1]
        if method == 'fixed':
            know_status = np.random.choice([True, False], p=[1-unknown_prob, unknown_prob])
            if know_status:
                response = self.kb.lookup((lhs, rel, rhs))
            else:
                response = 2
        elif method == 'variable':
            pass
        else:
            raise Exception

        return response


    def guess_check(self, guess, verbose=False):
        right = guess == self.target_entity 
        if verbose:
            comment = "Congratulations! You throw the right guess." if right else "Sorry! Maybe next time..."
            print(comment)
        return right, self.target_entity


        


