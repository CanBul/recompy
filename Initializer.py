import numpy as np


class initializer():
    def __init__(self, user_ids, item_ids, method, n_latent, mean, std):
        self.__mean = mean
        self.__std = std
        self.__n_latent = n_latent
        self.__user_ids = user_ids
        self.__item_ids = item_ids
        self.initalization_method = method

    @staticmethod
    def random_initializer(n_latent):
        return np.random.random_sample((n_latent,))

    @staticmethod
    def normal_initializer(mean, std, n_latent):
        return np.random.normal(mean, std, (n_latent,))

    @staticmethod
    def he_initializer(n_latent):
        return np.random.randn(n_latent) * np.sqrt(2/n_latent)

    def initialize_latent_vectors(self, initalization_method='he'):
        # Initialize item and user features as disctionary
        try:
                # if not valid method, return exception
            ['random', 'normal', 'he'].index(self.initalization_method)

        except ValueError:
            print(
                "Invalid initializer method. Choose among ['random','normal','he']")

        else:
            self.item_features = {}
            self.user_features = {}

            for user in self.__user_ids:
                # Generate random number for each user
                self.user_features.setdefault(
                    user, self.generate_random_feature())

            for item in self.__item_ids:
                # Generate random number for each item
                self.item_features.setdefault(
                    item, self.generate_random_feature())

            return self.user_features, self.item_features

    def generate_random_feature(self):

        # Generate features depending on given method
        if self.initalization_method == 'random':
            return self.random_initializer(self.__n_latent)

        elif self.initalization_method == 'normal':
            return self.normal_initializer(self.__mean, self.__std, self.__n_latent)

        elif self.initalization_method == 'he':
            return self.he_initializer(self.__n_latent)
