import numpy as np
from Similarities import Similarities

class ALS():

    def __init__(self, n_iterations, n_factors, regularization):
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.n_factors = n_factors


    def _set_data(self):
        n_users = len(np.unique(self.data[:,0]))
        n_items = len(np.unique(self.data[:,1]))
        new_item_ids = np.arange(0, n_items)
        new_user_ids = np.arange(0, n_users)
        self.item_ids_old_new = np.column_stack([np.unique(self.data[:,1]), new_item_ids])
        self.user_ids_old_new = np.column_stack([np.unique(self.data[:,0]), new_user_ids])
        ratings = np.zeros((n_users, n_items))
        for i in range(0, self.data.shape[0]):
            row = arr[i,:]
            item_column_index = self.item_ids_old_new[self.item_ids_old_new[:,0] == row[1]][:,1]
            user_row_index = self.user_ids_old_new[self.user_ids_old_new[:,0] == row[0]][:,1]
            ratings[int(user_row_index), int(item_column_index)] = row[2]

        self.ratings = ratings


    def train_test_split(self, test_portion = 0.1):
        test = np.zeros(self.ratings.shape)
        train = self.ratings.copy()
        test_set_size = test_portion * self.rating_length
        print(test_set_size)
        test_set_size_counter = 0

        # randomize users
        for user in range(self.ratings.shape[0]):
            test_index = np.random.choice(
                np.flatnonzero(self.ratings[user]), size = 3, replace = False)
            train[user, test_index] = 0.0
            test[user, test_index] = self.ratings[user, test_index]
            test_set_size_counter += len(test_index)
            if test_set_size_counter > test_set_size:
                break

        assert np.all(train * test == 0)
        return train, test


    def fit(self, data, test_portion = 0.1):
        self.data = data
        self.rating_length = data.shape[0]
        self._set_data()
        self.train, self.test = self.train_test_split(test_portion)

        self.n_user, self.n_item = self.train.shape
        self.user_factors = np.random.random((self.n_user, self.n_factors))
        self.item_factors = np.random.random((self.n_item, self.n_factors))
        self.test_mse_record = []
        self.train_mse_record = []
        print("Training has started.")
        for n in range(self.n_iterations):
            self.user_factors = self._als_step(self.train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(self.train.T, self.item_factors, self.user_factors)
            predictions = self.predict()
            predictions[predictions <= 0] = 0.5
            predictions[predictions > 5] = 5
            test_mse = self.compute_mse(self.test, predictions)
            train_mse = self.compute_mse(self.train, predictions)
            if(n % 10 == 0):
                print("Iteration number ", n)
                print("Train error is: ", train_mse)
                print("Test error is: ", test_mse)
            self.test_mse_record.append(test_mse)
            self.train_mse_record.append(train_mse)
        return self


    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.regularization
        b = ratings.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        return solve_vecs


    def predict(self):
        pred = self.user_factors.dot(self.item_factors.T)
        return pred


    def mean_squared_difference(a, b):
        summation = 0
        n = len(a)
        for i in range(0, n):
            difference = a[i] - b[i]
            squared_difference = difference**2
            summation = summation + squared_difference
        MSE = summation/n
        return np.sqrt(MSE)

    def _calculate_similarity(self, new_user, similarity_measure):
        #todo add similarty
        unique_user_ids = np.unique(self.data[:,0])
        similarities = []
        new_user_items = list(new_user.keys())
        new_user_ratings = list(new_user.values())

        intersected_item_index = self.item_ids_old_new[np.isin(self.item_ids_old_new[:,0], new_user_items)][:,1]
        intersected_item_index = list(intersected_item_index)
        intersected_item_index = [ int(x) for x in intersected_item_index ]
        user_ratings = self.ratings[:,list(intersected_item_index)]
        self.similarities = []
        for uid in unique_user_ids:
            user_information_index = int(self.user_ids_old_new[self.user_ids_old_new[:,0] == uid][:,1])
            unique_user_rating = list(user_ratings[user_information_index])
            unique_user_rating = [ int(x) for x in unique_user_rating]
            mse = ALS.mean_squared_difference(list(unique_user_rating), new_user_ratings)
            sim = [uid, user_information_index, mse]
            self.similarities.append(sim)


    def get_recommendation_for_new_user(self, new_user, similarity_measure='mean_squared_difference', howManyUsers, howManyItems):
        self._calculate_similarity(new_user,similarity_measure)
        self.similarities = np.asarray(self.similarities)
        self.similarities = self.similarities[self.similarities[:,2].argsort()]
        users_to_be_used = self.similarities[:howManyUsers]

        user_indexes = (list(users_to_be_used[:,1]))
        user_indexes = [int(x) for x in user_indexes]

        user_rating_matrix = self.ratings[user_indexes,]
        recommended_items_with_new_id = np.where(user_rating_matrix > 3.5)[1]
        indices = np.random.choice(len(recommended_items_with_new_id), howManyItems, replace=False)
        recommended_items_with_new_id = recommended_items_with_new_id[indices]
        recommended_items_with_old_id = self.item_ids_old_new[np.isin(self.item_ids_old_new[:,1], recommended_items_with_new_id)][:,0]
        return(recommended_items_with_old_id)


    def compute_mse(self, y_true, y_pred):
        mask = np.nonzero(y_true)
        mse = ALS.mean_squared_difference(y_true[mask], y_pred[mask])
        return mse
