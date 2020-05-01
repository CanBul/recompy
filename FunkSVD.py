from Main import Recommender
from Training import train
from Test import test


class FunkSVD(Recommender):
    def __init__(self, unary=False):
        pass

    def train(self, train_data, learning_rate, regularization, test_data):
        self.train_data = train_data
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self):
        self.result = train(self, self.train_data,
                            self.learning_rate, self.regularization)
