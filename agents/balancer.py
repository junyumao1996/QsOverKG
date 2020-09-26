from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt

from .utils import RingBuffer


class Balancer():
    def __init__(self, m_size=10000, t_low=10, t_high=10):
        self.memory = RingBuffer(m_size)
        self.max_iter = 1000
        self.t_low = t_low        # lower bound for switch
        self.t_high = t_high      # upper bound for not switch ( =t_low when want fixed threshold)
        self.need_train = False if t_high == t_low else True   # fixed threshold need no train
        self.is_train = False

        self.data_processor = StandardScaler()

        # clr = SVC(gamma='auto', max_iter=self.max_iter)
        self.classifier = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=self.max_iter, batch_size=256, early_stopping=True, validation_fraction=0.1)
        # self.classifier = make_pipeline(StandardScaler(), clr)

        self.visualizor = PCA(n_components=2)
        # self.visualizor = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)

    def add_datapoint(self, x):
        self.memory.append(x)

    def visualize_data(self):
        data = self.memory.get_data()
        x = np.array(data)[:, :-1]
        y = np.array(data)[:, -1]
        pos = np.nonzero(y==1.0)[0]
        neg = np.nonzero(y!=1.0)[0]
        # x = self.visualizor.fit_transform(x)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].scatter(x[pos, -1], x[pos, -2], c="r", label="Positive class", alpha=0.5)
        ax[0, 1].scatter(x[neg, -1], x[neg, -2], c="b", label="Negative class", alpha=0.5)
        # fig, ax = plt.subplots()
        ax[1, 0].scatter(x[pos, -1], x[pos, -2], c="r", label="Positive class", alpha=0.5)
        ax[1, 0].scatter(x[neg, -1], x[neg, -2], c="b", label="Negative class", alpha=0.5)
        ax[1, 0].legend(loc=1)
        plt.show()

    def train(self):
        if self.need_train:
            return self.train_epoch()
        else:
            return None

    def train_epoch(self):
        data = np.array(self.memory.get_data())
        np.random.shuffle(data)
        X = data[:, :-1]
        X = self.data_processor.fit_transform(X)
        y = data[:, -1]
        pos = np.nonzero(y==1.0)[0]
        neg = np.nonzero(y!=1.0)[0]
        n = min(len(pos), len(neg))
        # sample for label balance
        idx = np.vstack([pos[:n], neg[:n]]).reshape(-1)
        X = X[idx, :]
        y = y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        score = []

        self.classifier.fit(X_train, y_train)
        self.is_train = True
        score = self.classifier.score(X_test, y_test)
        return score

    def predict(self, x, threshold=0.8):
        x = x.reshape(1, -1)
        x = self.data_processor.transform(x)
        y_predict, y_prob = self.classifier.predict(x), self.classifier.predict_proba(x)
 
        if np.max(y_prob) > threshold:
            result = True
        else:
            result = False
        return result

    def give_switch_order(self, t, x):
        if t >= self.t_high:
            return True       # switch
        else:
            if t < self.t_low:
                return False  # retain
            else:
                if self.is_train:
                    return self.predict(x)
                else:
                    return False

    def give_module_name(self, t, x):
        order = self.give_switch_order(t, x)
        module = 'KA' if order else 'IS'
        return module

