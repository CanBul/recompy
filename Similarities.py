import numpy as np
from numpy.linalg import norm


class Similarities():

    @staticmethod
    def get_most_similar_users(user_ratings, user_predictions, similarity_measure, howMany):
        similarities = []

        for user, ratings in user_predictions.items():
            if similarity_measure == 'cosine_similarity':
                similarity = Similarities.cosine_similarity(
                    list(user_ratings.values()), ratings)
            elif similarity_measure == 'pearson_correlation':
                similarity = Similarities.pearson_correlation(
                    list(user_ratings.values()), ratings)
            elif similarity_measure == 'adjusted_cosine_similarity':
                similarity = Similarities.adjusted_cosine_similarity(
                    list(user_ratings.values()), ratings)
            elif similarity_measure == 'weighted_cosine_similarity':
                similarity = Similarities.weighted_cosine_similarity(
                    list(user_ratings.values()), ratings)
            elif similarity_measure == 'constrained_pearson_correlation':
                similarity = Similarities.weighted_cosine_similarity(
                    list(user_ratings.values()), ratings)
            elif similarity_measure == 'mean_squared_difference':
                similarity = Similarities.mean_squared_difference(
                    list(user_ratings.values()), ratings)
            elif similarity_measure == 'constrained_pearson_correlation':
                similarity = Similarities.constrained_pearson_correlation(
                    list(user_ratings.values()), ratings)

            similarities.append([user, similarity])

        similarities.sort(reverse=True, key=lambda x: x[1])

        return [each[0] for each in similarities[:howMany]]

    @staticmethod
    def cosine_similarity(a, b):

        return np.dot(a, b)/(norm(a)*norm(b))

    @staticmethod
    def pearson_correlation(a, b):
        corr = np.corrcoef(a, b)[0, 1]

        return corr

    @staticmethod
    def weighted_cosine_similarity(a, b):
        shared_item_count = len(a)
        cosine_similarity = np.dot(a, b)/(norm(a)*norm(b))
        weighted_cosine_similarity = cosine_similarity * \
            (1 / (1+np.exp(-1*shared_item_count)))

        return weighted_cosine_similarity

    @staticmethod
    def adjusted_cosine_similarity(a, b):
        mean_response = sum(sum(a, b)) / (2*len(a))
        a = a - mean_response
        b = b - mean_response

        return np.dot(a, b)/(norm(a)*norm(b))

    @staticmethod
    def mean_squared_difference(a, b):
        summation = 0
        n = len(a)
        for i in range(0, n):
            difference = a[i] - b[i]
            squared_difference = difference**2
            summation = summation + squared_difference
        MSE = summation/n

        return 1/MSE

    @staticmethod
    def constrained_pearson_correlation(a, b):
        median_a = np.median(a)
        median_b = np.median(b)
        nominator = np.dot((a - median_a), (b - median_b))
        denominator1 = np.sqrt(np.dot((a - median_a), (a - median_a)))
        denominator2 = np.sqrt(np.dot((b - median_b), (b - median_b)))
        denominator = denominator1 * denominator2
        cpc = nominator / denominator

        return cpc
