from Main import Recommender
from Test import test
import numpy as np
from Initializer import initializer
import copy
import bisect


class FunkSVD(Recommender):
    def __init__(self, unary=False):
        # Initialize all hyperparameters
        self.unary = unary
        self.set_hyperparameters()

    def set_hyperparameters(self, initialization_method='random', max_epoch=30, n_latent=10, learning_rate=0.01, regularization=0.1, early_stopping=False, init_mean=0, init_std=1):
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

    def train(self, train_data, test_split=False, test_portion=0.1):
        self.user_rated_items = {}

        for i, each in enumerate(train_data):

            # item ve user id stringe çevirilecek.

            # keep which user already rated which item
            self.user_rated_items.setdefault(each[0], []).append(each[1])

        # train test split
        self.test_split = test_split
        # test split method will be added
        if test_split:
            train_size = int(test_portion * len(train_data))
            np.random.shuffle(train_data)

            self.train_data = train_data[:train_size]
            self.test_data = train_data[train_size:]
        else:
            self.train_data = train_data

        # get distinct users
        self.user_ids = np.unique(train_data[:, 0])
        # get distinct items
        self.item_ids = np.unique(train_data[:, 1])

        print('Your data has {} distinct users and {} distinct items.'.format(
            len(self.user_ids), len(self.item_ids)))
        if test_split:
            print('Your data has been split into train and test set.')

    def fit(self):
        print('Initializing features for Users and Items...')
        initial = initializer(self.user_ids, self.item_ids, self.initialization_method,
                              self.n_latent, self.init_mean, self.init_std)

        self.user_features, self.item_features = initial.initialize_latent_vectors()

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
                    (error * self.item_features[item] - self.regularization * self.user_features[user])
                self.item_features[item] += self.learning_rate * \
                    (error * temp - self.regularization * self.item_features[item])

            # Get all of these below to their own method
            # Calculate errors
            error_counter += 1
            train_error = self.get_error(self.train_data)

            if self.test_split:
                test_error = self.get_error(self.test_data)
                print('Epoch Number: {}/{} Training RMSE: {:.2f} Test RMSE: {:.2f}'.format(epoch+1, self.max_epoch,
                                                                                           train_error, test_error))

            else:
                print('Epoch Number: {}/{} Training RMSE: {:.2f}'.format(epoch+1, self.max_epoch,
                                                                         train_error))

            # Save best features depending on test_error and reset counter
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

            # Break if test_error didn't improve for the last n rounds
            if self.early_stopping and error_counter >= self.early_stopping:

                self.user_features = copy.deepcopy(best_user_features)
                self.item_features = copy.deepcopy(best_item_features)

                print("Test error didn't get lower for the last {} epochs. Training is stopped.".format(
                    error_counter))
                print('Best test error is: {:.2f}. Related features are saved.'.format(
                    self.min_test_error))
                break

        print('Training has ended...')

    def get_recommendation_for_existing_user(self, user_id, howMany=10):
        result_list = []
        # this might be more effective using matrix multiplication
        for item in self.item_ids:
            # if user did not already rate the item
            if item not in self.user_rated_items[user_id]:
                prediction = np.dot(
                    self.best_user_features[user_id], self.best_item_features[item])
                bisect.insort(result_list, [prediction, item])

        return [x[1] for x in result_list[::-1][0:howMany]]

    def get_recommendation_for_new_user(self, user_ratings):
        pass

    def get_similar_products(self, product_id, howMany=10):

        result_list = []
        product_features = self.best_item_features[product_id]

        for item in self.item_ids:

            if item == product_id:
                continue

            cos_sim = np.dot(self.best_item_features[item], product_features) / (
                np.linalg.norm(self.best_item_features[item] * np.linalg.norm(product_features)))

            bisect.insort(result_list, [cos_sim, item])

        return [x[1] for x in result_list[::-1][0:howMany]]

    def get_error(self, data):
        # Get RMSE for the given data (train or test)

        total_error = 0
        counter = 0

        for user, item, rating in data:
            if user not in self.user_features or item not in self.item_features:
                continue

            total_error += (rating -
                            np.dot(self.user_features[user], self.item_features[item]))**2
            counter += 1

        return np.sqrt(total_error/counter)

    def novelty(self,recommendation_list):
        user_n = len(self.user_ids)
        novelty = 0

        for movie in recommendation_list:
            # Calculate novelty for each item in the recommendation list
            pop_item = self.train_data[self.train_data[:,1]==movie].shape[0]
            novelty += 1 - (pop_item/user_n)

        novelty = novelty/len(recommendation_list)

        return novelty

    def precision_recall_at_k(self,threshold,k):
        user_true_pred = dict()
        precision_k = dict()
        recall_k = dict()

        for row in self.test_data:

            user_id = row[0]
            item_id = row[1]
            true_rating = row[2]

            # Check the user in the test set also in the training set
            if user_id not in self.user_features or item_id not in self.item_features:
                continue


            estimated_rating = np.dot(self.user_features[user_id],self.item_features[item_id])
            try:
                user_true_pred[user_id].append((true_rating, estimated_rating))

            except KeyError:
              # Create a dictionary with list as values
                user_true_pred[user_id] = []
                user_true_pred[user_id].append((true_rating, estimated_rating))


        for user_id, rating in user_true_pred.items():

            rating.sort(key=lambda x: x[1], reverse=True)

            # Number of recommended items at k
            recommended_c = sum((estimated >= threshold) for (_,estimated) in rating[:k])

            # Number of relevant items 
            relevant_c = sum((true >= threshold) for (true,_) in rating)

            # Number of relevant and recommended items in top k
            recommended_in_relevant = sum(((estimated >= threshold) and (true >= threshold))
                                          for (true, estimated) in rating[:k])

            # Precision at K
            precision_k[user_id] = recommended_in_relevant / recommended_c if recommended_c != 0 else 0

            # Recall at K
            recall_k[user_id] = recommended_in_relevant / relevant_c if relevant_c != 0 else 0

            # Precision and recall can then be averaged over all users
            self.precision = sum(prec for prec in precision_k.values()) / len(precision_k)
            self.recall = sum(rec for rec in recall_k.values()) / len(recall_k)

        return print('Predicision@K: {}\nRecall@K: {}'.format(self.precision,self.recall))
