# import library files
from Main import Recommender
from Test import Test
from Initializer import initializer
from Similarities import Similarities
from Train_test_split import train_test_split

# import other libraries
import numpy as np
from numpy.linalg import norm
import copy
import bisect


class FunkSVD(Recommender):
    def __init__(self):
        # Initialize all hyperparameters
        self.set_hyperparameters()

    def set_hyperparameters(self, initialization_method='random', max_epoch=5, n_latent=10, learning_rate=0.01, regularization=0.1, early_stopping=False, init_mean=0, init_std=1):
        self.initialization_method = initialization_method
        self.max_epoch = max_epoch
        self.n_latent = n_latent
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.early_stopping = early_stopping
        self.init_mean = init_mean
        self.init_std = init_std

        self.min_train_error = np.inf
        self.min_test_error = np.inf

    def __set_data(self, data, test_portion):

        # get distinct users, items and user_existing_ratings, item_existing_users
        self.user_rated_items = {}
        self.items_rated_by_users = {}
        self.user_ids = []
        self.item_ids = []

        np.random.shuffle(data)

        # variables for train and test split
        user_dictionary = {}
        item_dictionary = {}
        self.train_data = []
        self.test_data = []

        for user, item, score in data:
            # Unique users and items

            if type(user) == np.float64:
                user = int(user)
            if type(item) == np.float64:
                item = int(item)
            user = str(user)
            item = str(item)
            score = float(score)

            if user not in self.user_rated_items:
                self.user_ids.append(user)
            if item not in self.items_rated_by_users:
                self.item_ids.append(item)

            self.items_rated_by_users.setdefault(item, []).append(user)
            self.user_rated_items.setdefault(user, []).append(item)

            if self.test_split:
                # train and test set
                user_dictionary.setdefault(user, 0)
                item_dictionary.setdefault(item, 0)

                if user_dictionary[user] * test_portion >= 1 and item_dictionary[item] * test_portion >= 1:

                    self.test_data.append([user, item, score])

                    user_dictionary[user] -= 1
                    item_dictionary[item] -= 1

                else:
                    self.train_data.append([user, item, score])

                    user_dictionary[user] += 1
                    item_dictionary[item] += 1
            else:
                self.train_data.append([user, item, score])

        print('Your data has {} distinct users and {} distinct items.'.format(
            len(self.user_ids), len(self.item_ids)))

        if self.test_split:

            print('Your data has been split into train and test set.')
        else:

            print('Your data has no test set.')

    def fit(self, data, test_split=True, test_portion=0.1, search_parameter_space=False):

        # Set train_data, test_data, user_ids etc. if search parameter is False
        # This lets us search parameter space with train-test split
        if not search_parameter_space:
            self.test_split = test_split
            self.__set_data(data, test_portion)

        # Initialization
        print('Initializing features for Users and Items...')
        initial = initializer(self.user_ids, self.item_ids, self.initialization_method,
                              self.n_latent, self.init_mean, self.init_std)

        self.user_features, self.item_features = initial.initialize_latent_vectors()

        # Training
        print('Starting training...')
        error_counter = 0
        for epoch in range(self.max_epoch):

            # updating user and item features
            for user, item, rating in self.train_data:

                error = rating - \
                    np.dot(self.user_features[user], self.item_features[item])
                # Use temp to update each item and user feature in sync.
                temp = self.user_features[user]

                # Update user and item feature for each user, item and rating pair
                self.user_features[user] += self.learning_rate * \
                    (error * self.item_features[item] -
                     self.regularization * self.user_features[user])
                self.item_features[item] += self.learning_rate * \
                    (error * temp - self.regularization *
                     self.item_features[item])

            # Calculate errors
            error_counter += 1
            train_error = Test.rmse_error(
                self.train_data, self.user_features, self.item_features)

            # Show error to Client
            if self.test_split:
                test_error = Test.rmse_error(
                    self.test_data, self.user_features, self.item_features)
                print('Epoch Number: {}/{} Training RMSE: {:.2f} Test RMSE: {}'.format(epoch+1, self.max_epoch,
                                                                                       train_error, test_error))

            else:
                print('Epoch Number: {}/{} Training RMSE: {:.2f}'.format(epoch+1, self.max_epoch,
                                                                         train_error))

            # Save best features depending on test_error
            if self.test_split and test_error < self.min_test_error:
                self.min_test_error = test_error
                best_user_features = copy.deepcopy(self.user_features)
                best_item_features = copy.deepcopy(self.item_features)

                error_counter = 0
            # Save best features if test data is False
            elif not self.test_split and train_error < self.min_train_error:
                self.min_train_error = train_error
                best_user_features = copy.deepcopy(self.user_features)
                best_item_features = copy.deepcopy(self.item_features)

            # Break if test_error didn't improve for the last n rounds and early stopping is true
            if self.early_stopping and error_counter >= self.early_stopping:

                print("Test error didn't get lower for the last {} epochs. Training is stopped.".format(
                    error_counter))
                print('Best test error is: {:.2f}. Best features are saved.'.format(
                    self.min_test_error))
                break

        print('Training has ended...')
        self.user_features = copy.deepcopy(best_user_features)
        self.item_features = copy.deepcopy(best_item_features)

    def get_recommendation_for_existing_user(self, user_id, howMany=10):
        result_list = []
        # this might be more effective using matrix multiplication
        for item in self.item_ids:
            # if user did not already rate the item
            if item not in self.user_rated_items[user_id]:
                prediction = np.dot(
                    self.user_features[user_id], self.item_features[item])
                bisect.insort(result_list, [prediction, item])

        return [x[1] for x in result_list[::-1][0:howMany]]

    def get_recommendation_for_new_user(self, user_ratings,
                                        similarity_measure='cosine_similarity', howManyUsers=3, howManyItems=5):

        # Get user predictions on same movies
        user_predictions = self.__user_prediction_for_same_movies(user_ratings)
        # Find most most similar user_ids
        user_ids = Similarities.get_most_similar_users(
            user_ratings, user_predictions, similarity_measure, howManyUsers)

        result_list = []
        # get user features for users who are most similar to given new user
        for user in user_ids:
            for item, item_feature in self.item_features.items():
                # predict ratings for most similar users
                prediction = np.dot(
                    self.user_features[user], item_feature)
                bisect.insort(result_list, [prediction, item])

        return [x[1] for x in result_list[::-1][:howManyItems]]

    def get_similar_products(self, product_id, howMany=10):

        result_list = []
        product_features = self.item_features[product_id]

        for item in self.item_ids:

            if item == product_id:
                continue
            # add cosine sim function from similarites
            cos_sim = Similarities.cosine_similarity(
                self.item_features[item], product_features)

            bisect.insort(result_list, [cos_sim, item])

        return [x[1] for x in result_list[::-1][0:howMany]]

    def __user_prediction_for_same_movies(self, user_ratings):
        result = {}
        for key in user_ratings:
            if key not in self.item_features:
                continue

            for user in self.user_features:
                result.setdefault(user, []).append(
                    np.dot(self.user_features[user], self.item_features[key]))

        return result


# data = np.genfromtxt('./data/movielens100k.csv', delimiter=',')
# myFunk = FunkSVD()
# myFunk.set_hyperparameters(early_stopping=5)
# myFunk.fit(data, test_split=True, test_portion=0.1)
# new_user = {'242': 4,
#             '302': 5,
#             '377': 4}
# #user_similarities = myFunk.get_most_similar_users(new_user, 'cosine_similarity', 3)
# #user_ids = user_similarities[:,0]
# # print(user_ids)
# myFunk.get_recommendation_for_new_user(new_user, 'cosine_similarity', 3)
