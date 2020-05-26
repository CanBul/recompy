# Recompy

Recompy is a library for recommender systems. It provides an easy framework to train different models, calculate similarities and recommend items for both existing and new users.

Recompy comes bundled with MovieLens data which consists of 100.000 user, rating pair.

Recompy is a fairly optimized and lightweight library. Its only dependency is numpy library which is downloaded automatically when you pip install recompy. This feature is useful if you decide to run recompy on server side.

Recompy lets you search parameter space for finding the best model for your data. It keeps best features in memory depending on the test set error. Early stopping is also available. When it is set to an integer, training will be stopped if test set error doesn't improve for the last given epochs.  

Current version supports algorithms below:
* FunkSVD
* KNN
* NMF
* SVD++
* RUS
* ALS


# Installation
```shell
pip install recompy
```
# Usage

```python
from recompy import load_movie_data, FunkSVD

# get MovieLens data
data = load_movie_data()
# initialization of FunkSVD model
myFunk = FunkSVD()
# training of the model
myFunk.fit(data)

# Create new user. Key:Item ID, Value:Rating
new_user = {'1':5,
            '2':4,
            '4':3}
            
# To find the most similar user resulting from cosine similarity. Recommend 5 items using the most similar user 
myFunk.get_recommendation_for_new_user(new_user, similarity_measure = 'cosine_similarity', 
                                       howManyUsers = 1, howManyItems = 5)
```

## Available Methods

### set_hyperparameters():
A method to set hyperparameters for searching parameter space. 
Arguments:
1. initialization_method
            1. random ( default )
            2. he
            3. random
2. max_epoch
Available initialization techniques are: Random initializer, Normal initializer and He initializer. init_mean and init_sd parameters are used in Normal Initializer as mean and standard deviation.

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

### Note
This library is created as a part of AI Projects program @ inzva. You can see more at inzva.com
