import numpy as np


class Test():

    @staticmethod
    def rmse_error(data, user_features, item_features):
        # Get RMSE for the given data (train or test)

        total_error = 0
        counter = 0

        for user, item, rating in data:
            if user not in user_features or item not in item_features:
                continue

            total_error += (rating -
                            np.dot(user_features[user], item_features[item]))**2
            counter += 1

        return np.sqrt(total_error/counter)

    @staticmethod
    def novelty(train_data, recommendation_list, user_n):

        novelty = 0

        for movie in recommendation_list:
            # Calculate novelty for each item in the recommendation list
            pop_item = train_data[train_data[:, 1] == movie].shape[0]
            novelty += 1 - (pop_item/user_n)

        novelty = novelty/len(recommendation_list)

        return novelty

    @staticmethod
    def precision_recall_at_k(test_data, user_features, item_features, threshold, k):
        user_true_pred = dict()
        precision_k = dict()
        recall_k = dict()

        for row in test_data:

            user_id = row[0]
            item_id = row[1]
            true_rating = row[2]

            # Check the user in the test set also in the training set
            if user_id not in user_features or item_id not in item_features:
                continue
            estimated_rating = np.dot(
                user_features[user_id], item_features[item_id])
            try:
                user_true_pred[user_id].append((true_rating, estimated_rating))

            except KeyError:
              # Create a dictionary with list as values
                user_true_pred[user_id] = []
                user_true_pred[user_id].append((true_rating, estimated_rating))

        for user_id, rating in user_true_pred.items():

            rating.sort(key=lambda x: x[1], reverse=True)

            # Number of recommended items at k
            recommended_c = sum((estimated >= threshold)
                                for (_, estimated) in rating[:k])

            # Number of relevant items
            relevant_c = sum((true >= threshold) for (true, _) in rating)

            # Number of relevant and recommended items in top k
            recommended_in_relevant = sum(((estimated >= threshold) and (true >= threshold))
                                          for (true, estimated) in rating[:k])

            # Precision at K
            precision_k[user_id] = recommended_in_relevant / \
                recommended_c if recommended_c != 0 else 0

            # Recall at K
            recall_k[user_id] = recommended_in_relevant / \
                relevant_c if relevant_c != 0 else 0

            # Precision and recall can then be averaged over all users
            precision = sum(
                prec for prec in precision_k.values()) / len(precision_k)
            recall = sum(rec for rec in recall_k.values()) / len(recall_k)

        print('Predicision@K: {}\nRecall@K: {}'.format(precision, recall))

        return [precision, recall]
