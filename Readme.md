# Recompy

Recompy is a library for recommender systems. It provides an easy framework to train models, calculate similarities, showing recommendations.

Recompy shows the train end test errors in each epoch. After a new user is created by defining item id and rating, recommendation can simply obtained. Recompy uses FunkSVD algorithm to train recommender system model. Multiple similarity metrics can be used to calculate user similarity for any given new user.

## Functions

_set_hyperparameters(initialization_method, max_epoch, n_latent,_
_learning_rate, regularization, early_stopping, init_mean_ init*sd):*
A function to set hyperparameters. Available initialization techniques are: Random initializer, Normal initializer and He initializer. init_mean and init_sd parameters are used in Normal Initializer as mean and standard deviation.

_train_test_split(rated_count, movie_ratio_to_be_splitted, test_split):_
A function to perform train test split.

_fit():_
Trains FunkSVD model.

_get_recommendation_for_existing_user(user_id, howMany):_
Gets howMany recommendations for given user_id.

_get_recommendation_for_new_user(user_ratings, similarity_measure,_
_howManyUsers, howManyItems):_ Gets recommendations for new user by a given similarity measure. Similarity measures can be Cosine Similarity, Pearson Correlation, Adjusted Cosine Similarity, Weighted Cosine Similarity, Constrained Pearson Correlation, Mean Squared Difference.

_get_similar_products(item_id, howMany):_
Gets howMany similar items to a given item.

_novelty(recommendation_list):_
Returns novelty of a given recommendation list.

_precision_recall_at_k(threshold, k):_
Returns precision and recall values of recommended items at k.
