from abc import ABC, abstractmethod


class Recommender(ABC):

    @abstractmethod
    def train(self, train_data, learning_rate, regularization, test_data):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_recommendation_for_existing_user(self, user_id):
        pass

    @abstractmethod
    def get_recommendation_for_new_user(self, user_ratings):
        pass

    @abstractmethod
    def get_similar_products(self, product_id):
        pass
