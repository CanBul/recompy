from abc import ABC, abstractmethod


class Recommender(ABC):

    @abstractmethod
    def train(self, train_data, test_data):
        pass

    @abstractmethod
    def set_hyperparameters(self, all_hyperparameters):
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


class Initialization(ABC):
    # returns n_latent_features with given parameters
    @abstractmethod
    def initialize_latent_vector(self, user_ids, item_ids, method, n_latent, mean, std):
        pass
