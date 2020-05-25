# import library files
from Test import Test
from Initializer import initializer
from Similarities import Similarities

# import other libraries
import numpy as np
import copy
import bisect

class NMF():
    def __init__(self):
        self.set_hyperparameters()

    def set_hyperparameters(self, initialization_method='random',n_latent=15, n_epochs=30, biased=False, reg_pu=.06,
                 reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,random_state=None, verbose=False):

        self.n_latent = n_latent
        self.initialization_method = initialization_method
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.random_state = random_state
        self.verbose = verbose

    def __set_data(self, data, test_portion):

        # get distinct users, items and user_existing_ratings, items_existing_users
        self.user_existing_ratings = {}
        self.items_rated_by_users = {}
        self.user_ids = []
        self.item_ids = []
        self.all_ratings_in_train = 0

        np.random.shuffle(data)

        # variables for train and test split
        user_dictionary = {}
        item_dictionary = {}

        self.user_n_ratings = {}
        self.item_n_ratings = {}

        self.train_data = []
        self.test_data = []

        self.train_data_user_ids = []
        self.train_data_item_ids = []
        self.test_data_user_ids = []
        self.test_data_item_ids = []

        for user, item, score in data:
            # Unique users and items

            try:
                user = int(user)
            except:
                pass
            try:
                item = int(item)
            except:
                pass

            user = str(user)
            item = str(item)
            score = float(score)

            if user not in self.user_existing_ratings:
                self.user_ids.append(user)
            if item not in self.items_rated_by_users:
                self.item_ids.append(item)

            self.items_rated_by_users.setdefault(item, []).append(user)
            self.user_existing_ratings.setdefault(user, []).append(item)

            ratio = len(self.test_data) / (len(self.train_data)+0.001)

            if self.test_split:
                # train and test set
                user_dictionary.setdefault(user, 0)
                item_dictionary.setdefault(item, 0)

                if user_dictionary[user] * test_portion >= 1 and item_dictionary[item] * test_portion >= 1 and ratio <= test_portion+0.02:

                    self.test_data.append([user, item, score])
                    if user not in self.test_data_user_ids: self.test_data_user_ids.append(user)
                    if item not in self.train_data_item_ids: self.test_data_item_ids.append(item)

                    try: self.user_n_ratings[user] += 1
                    except KeyError: self.user_n_ratings.setdefault(user, 1)
                    try: self.item_n_ratings[item] += 1
                    except KeyError: self.item_n_ratings.setdefault(item, 1)

                    user_dictionary[user] -= 1
                    item_dictionary[item] -= 1

                else:
                    self.train_data.append([user, item, score])
                    self.all_ratings_in_train += score
                    if user not in self.train_data_user_ids: self.train_data_user_ids.append(user)
                    if item not in self.train_data_item_ids: self.train_data_item_ids.append(item)

                    try: self.user_n_ratings[user] += 1
                    except KeyError: self.user_n_ratings.setdefault(user, 1)
                    try: self.item_n_ratings[item] += 1
                    except KeyError: self.item_n_ratings.setdefault(item, 1)

                    user_dictionary[user] += 1
                    item_dictionary[item] += 1
            else:
                self.train_data.append([user, item, score])
                self.all_ratings_in_train += score
                if user not in self.train_data_user_ids: self.train_data_user_ids.append(user)
                if item not in self.train_data_item_ids: self.train_data_item_ids.append(item)

                try: self.user_n_ratings[user] += 1
                except KeyError: self.user_n_ratings.setdefault(user, 1)
                try: self.item_n_ratings[item] += 1
                except KeyError: self.item_n_ratings.setdefault(item, 1)

        print('Your data has {} distinct users and {} distinct items.'.format(
            len(self.user_ids), len(self.item_ids)))

        if len(self.test_data) < 1 and self.test_split:
            self.test_split = False
            self.early_stopping = False
            print("Training set doesn't have enough data for given test portion.")

        if self.test_split:

            print('Your data has been split into train and test set.')
            print('Length of training set is {}. Length of Test set is {}'.format(
                len(self.train_data), len(self.test_data)))
        else:

            print('Your data has no test set.')
            print('Length of training set is {}'.format(len(self.train_data)))

    def fit(self, data,test_split=True, test_portion=0.1, search_parameter_space=False):

        if not search_parameter_space:

            self.test_split = test_split
            self.__set_data(data, test_portion)

        print('Initializing features for Users and Items...')
        initial = initializer(self.user_ids, self.item_ids, self.initialization_method,
                              self.n_latent, 0,0)

        pu, qi = initial.initialize_latent_vectors()
        bu = dict([(key, 0) for key in self.train_data_user_ids])
        bi = dict([(key, 0) for key in self.train_data_item_ids])

        if not self.biased:
           global_mean = 0
        else:
            global_mean = self.all_ratings_in_train / len(self.train_data)

        for current_epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            # (re)initialize nums and denoms to zero
            self.user_num = dict([(key, np.zeros(self.n_latent)) for key in self.train_data_user_ids])
            self.user_denom = dict([(key, np.zeros(self.n_latent)) for key in self.train_data_user_ids])
            self.item_num = dict([(key, np.zeros(self.n_latent)) for key in self.train_data_item_ids])
            self.item_denom  = dict([(key, np.zeros(self.n_latent)) for key in self.train_data_item_ids])

            for u, i, r in self.train_data:

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_latent):
                    dot += pu[u][f] * qi[i][f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                # Update biases
                if self.biased:
                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

                # Compute numerators and denominators
                for f in range(self.n_latent):
                    self.user_num[u][f] += qi[i][f] * r
                    self.user_denom[u][f] += qi[i][f] * est
                    self.item_num[i][f] += pu[u][f] * r
                    self.item_denom[i][f] += pu[u][f] * est

            # Update user factors
            for u in self.train_data_user_ids:
                n_ratings = self.user_n_ratings[u]
                for f in range(self.n_latent):
                    self.user_denom[u][f] += n_ratings * self.reg_pu * pu[u][f]
                    pu[u][f] *= self.user_num[u][f] / self.user_denom[u][f]

            # Update item factors
            for i in self.train_data_item_ids:
                n_ratings = self.item_n_ratings[i]
                for f in range(self.n_latent):
                    self.item_denom[i][f] += n_ratings * self.reg_qi * qi[i][f]
                    qi[i][f] *= self.item_num[i][f] / self.item_denom[i][f]

            # Calculate errors
            #error_counter += 1
            train_error = Test.rmse_error(
                self.train_data, pu, qi)

            # Show error to Client
            if self.test_split:
                test_error = Test.rmse_error(
                    self.test_data, pu, qi)
                print('Epoch Number: {}/{} Training RMSE: {:.2f} Test RMSE: {}'.format(current_epoch+1, self.n_epochs,
                                                                                        train_error, test_error))

            else:
                print('Epoch Number: {}/{} Training RMSE: {:.2f}'.format(current_epoch+1, self.n_epochs,
                                                                            train_error))

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi


    def get_recommendation_for_existing_user(self, user_id, howMany=10):
        result_list = []
        # this might be more effective using matrix multiplication
        for item in self.item_ids:
            # if user did not already rate the item
            if item not in self.user_existing_ratings[user_id]:
                prediction = np.dot(
                    self.pu[user_id], self.qi[item])
                bisect.insort(result_list, [prediction, item])

        return [x[1] for x in result_list[::-1][0:howMany]]


class FunkSVD():

    def __init__(self):
        # Initialize default hyperparameters
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

        # get distinct users, items and user_existing_ratings, items_existing_users
        self.user_existing_ratings = {}
        self.items_rated_by_users = {}
        self.user_ids = []
        self.item_ids = []

        np.random.shuffle(data)

        # variables for train and test split
        user_dictionary = {}
        item_dictionary = {}
        self.train_data = []
        self.test_data = []

        self.train_data_user_ids = []
        self.train_data_item_ids = []
        self.test_data_user_ids = []
        self.test_data_item_ids = []

        for user, item, score in data:
            # Unique users and items

            try:
                user = int(user)
            except:
                pass
            try:
                item = int(item)
            except:
                pass

            user = str(user)
            item = str(item)
            score = float(score)

            if user not in self.user_existing_ratings:
                self.user_ids.append(user)
            if item not in self.items_rated_by_users:
                self.item_ids.append(item)

            self.items_rated_by_users.setdefault(item, []).append(user)
            self.user_existing_ratings.setdefault(user, []).append(item)

            ratio = len(self.test_data) / (len(self.train_data)+0.001)

            if self.test_split:
                # train and test set
                user_dictionary.setdefault(user, 0)
                item_dictionary.setdefault(item, 0)

                if user_dictionary[user] * test_portion >= 1 and item_dictionary[item] * test_portion >= 1 and ratio <= test_portion+0.02:

                    self.test_data.append([user, item, score])
                    if user not in self.test_data_user_ids: self.test_data_user_ids.append(user)
                    if item not in self.train_data_item_ids: self.test_data_item_ids.append(item)

                    user_dictionary[user] -= 1
                    item_dictionary[item] -= 1

                else:
                    self.train_data.append([user, item, score])
                    if user not in self.train_data_user_ids: self.train_data_user_ids.append(user)
                    if item not in self.train_data_item_ids: self.train_data_item_ids.append(item)

                    user_dictionary[user] += 1
                    item_dictionary[item] += 1
            else:
                self.train_data.append([user, item, score])
                if user not in self.train_data_user_ids: self.train_data_user_ids.append(user)
                if item not in self.train_data_item_ids: self.train_data_item_ids.append(item)

        print('Your data has {} distinct users and {} distinct items.'.format(
            len(self.user_ids), len(self.item_ids)))

        if len(self.test_data) < 1 and self.test_split:
            self.test_split = False
            self.early_stopping = False
            print("Training set doesn't have enough data for given test portion.")

        if self.test_split:

            print('Your data has been split into train and test set.')
            print('Length of training set is {}. Length of Test set is {}'.format(
                len(self.train_data), len(self.test_data)))
        else:

            print('Your data has no test set.')
            print('Length of training set is {}'.format(len(self.train_data)))

    def fit(self, data, test_split=True, test_portion=0.1, search_parameter_space=False):

        # Set train_data, test_data, user_ids etc. if search parameter is False
        # If True, this lets us search parameter space with the same train-test split
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
                self.best_user_features = copy.deepcopy(self.user_features)
                self.best_item_features = copy.deepcopy(self.item_features)

                error_counter = 0
            # Save best features if test data is False
            elif not self.test_split and train_error < self.min_train_error:
                self.min_train_error = train_error
                self.best_user_features = copy.deepcopy(self.user_features)
                self.best_item_features = copy.deepcopy(self.item_features)

            # Break if test_error didn't improve for the last n rounds and early stopping is true
            if self.early_stopping and error_counter >= self.early_stopping:

                print("Test error didn't get lower for the last {} epochs. Training is stopped.".format(
                    error_counter))
                print('Best test error is: {:.2f}. Best features are saved.'.format(
                    self.min_test_error))
                break

        print('Training has ended...')
        self.user_features = copy.deepcopy(self.best_user_features)
        self.item_features = copy.deepcopy(self.best_item_features)

    def get_recommendation_for_existing_user(self, user_id, howMany=10):
        result_list = []
        # this might be more effective using matrix multiplication
        for item in self.item_ids:
            # if user did not already rate the item
            if item not in self.user_existing_ratings[user_id]:
                prediction = np.dot(
                    self.user_features[user_id], self.item_features[item])
                bisect.insort(result_list, [prediction, item])

        return [x[1] for x in result_list[::-1][0:howMany]]

    def get_recommendation_for_new_user(self, user_ratings,
                                        similarity_measure='mean_squared_difference', howManyUsers=3, howManyItems=5):

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

        # remove duplicates
        return_list = []
        for pair in result_list:
            if len(return_list) >= howManyItems:
                break
            if pair[1] in return_list:
                continue

            return_list.append(pair[1])

        return return_list

    def get_similar_products(self, item_id, howMany=10):

        result_list = []
        product_features = self.item_features[item_id]

        for item in self.item_ids:

            if item == item_id:
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