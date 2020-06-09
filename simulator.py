import numpy as np
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.ranking import PageRank, BiPageRank


class Simulator():
    def __init__(self, dataset):
        self.kb = dataset['tensor']                       # 3-way tensor [lhs, pred, rhs]
        self.data_info = dataset['info']
        self.n_entities = self.kb.shape[0]

    def calc_propularity(self, method='uniform'):
        if method == 'number':
            popularity = self.data_info['probas']['both']
        elif method == 'uniform':
            popularity = np.ones(self.n_entities)
        elif method == 'pagerank':
            adjacency = np.sum(self.kb, axis=[0,1])
            adjacency = adjacency[adjacency > 0].astype(np.int32)
            popularity = pagerank.fit_transform(adjacency)
        else:
            raise Exception

        return popularity

    def entity_select(self):
        """
        Select the target entity.
        """
        pop = self.calc_propularity_uniform(method='unform') 
        pop_distri = popularity / np.sum(popularity)
        target = np.random.choice(range(self.n_entities), size=1, p=pop_distri)
        self.target_entity = target
        return target

    def reponse(self, question, tag, method='fixed', unknown_prob=0.1):
        """
        Args:
            question - 3-element list (subj, rel, obj) wit a Nan in guessed position
            tag - index of guessed position
        Returns:
            response - 0: no, 1: yes, 2: unknown
        """
        pos = question
        question[tag] = self.target_entity 
        subj, rel, obj = question
        if method == 'fixed':
            know_status = np.random.choice([True, False], p=[1-unknown_prob, unknown_prob])
            if know_status:
                response = self.kb[subj, rel, obj]
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


        


