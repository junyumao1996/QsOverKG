import numpy as np
import torch
from .utils import EntropyIS

class Agent(object):
    def __init__(self, n_entities, n_predicates, switch_thres=None):
        self.kb_multinoulli = np.ones((n_entities, n_predicates, n_entities, 3)) / 3   # 3 is # multinoulli parameters (no, yes, unknown)
        self.n_chances = 20
        self.ent_log = np.zeros(n_entities)
        self.t = 0
        self.episode_log = {'question':[], 'response':[]}
        self.switch_thres = switch_thres

    def reset():
        """
        Reset agent state and episode (when new episode starts). 
        """
        self.t = 0
        self.episode_log = {'question':[], 'response':[]}

    def question(self):
        """
        Compose/generate the question.
        """
        pass

    def update_response(self, response):
        """
        Append the lastest response.
        """
        self.episode_log['response'].append(response)
        assert len(self.episode_log['response']) == len(self.episode_log['question'])

    def info_seeking(self):
        """
        Information Seeking (IS) module.
        """
        pass

    def know_acqusition(self):
        """
        Knowledge Acquision (KA) module.
        """
        pass

    def get_feedback(self, feedback):
        """
        Record the entity log in this episode.
        """
        self.ent_log[feedback] += 1

    def guess_generate(self, method='number'):
        """
        Generate the final guess given the reponses.
        """
        # compute independent likelihood
        q_log = np.array(self.episode_log['question'])
        a_log = np.array(self.episode_log['response'])
        prob = self.kb_multinoulli[:, q_log[:, 0], q_log[:, 1], a_log]
        prob_log = np.log(prob)         # size (n_ent, 20)
        lik_log = np.sum(prob_log, axis=0)

        # compute prior
        if method == 'number':
            prior = self.ent_log[feedback] + 1   # laplace smoothing
            prior = prior / np.sum(prior)
        elif method == 'uniform':
            prior = np.ones(self.n_entities) / self.n_entities
        else:
            raise Exception
        prior_log  = np.log(prior)

        # bayes' rule
        posterior = prior + lik_log

        return np.argmax(posterior)

        

class AgentRandom(Agent):
    """
    Random question picking (fixed opportunities)
    """
    def __init__(self, n_entities, n_predicates, switch_thres):
        super(AgentRandom, self).__init__(n_entities, n_predicates, switch_thres)

    def question(self):
        # fixed opportunities for modules
        if self.t < self.switch_thres:
            question = self.info_seeking()
        else:
            question = self.know_acqusition()

        self.t += 1
        self.episode_log['question'].append(question)

        return question

    def info_seeking(self):
        rel = np.random.randint(n_predicates)
        obj = np.random.randint(n_entities)
        return rel, obj



class AgentEntropy(Agent):
    """
    Question picking based on entropy ranking (fixed opportunities)
    """
    def __init__(self, n_entities, n_predicates, switch_thres):
        super(AgentRandom, self).__init__(n_entities, n_predicates, switch_thres)

    def question(self):
        self.episode_log['question'].append(last_question)
        self.episode_log['response'].append(last_response)

        # fixed opportunities for modules
        if self.t < self.switch_thres:
            question = self.info_seeking()
        else:
            question = self.know_acqusition()
        return question

        self.t += 1

    def info_seeking(self):
        return EntropyIS(self.kb_multinoulli)


