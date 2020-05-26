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

## Available Methods for FunkSVD

### set_hyperparameters():

A method to set hyperparameters for searching parameter space. Arguments:

__initialization_method :__  How to initialize user and item embeddings
* random ( default )
* he
* normal

__max_epoch :__ Epoch count to train model. Default is 5.\
__n_latent :__ Length of user and item embeddings. Default is 10\
__learning_rate :__ Learning rate of the model. Default is 0.01\
__regularization :__ Regularization rate of the model. DEfault is 0.1\
__early_stopping :__ Number of epochs to stop if test error doesn't improve. Default is False.\
__init_mean :__ Initialization mean if initialization method is normal.\
__init_std :__ Initialization standard deviation if initialization is normal\


### fit():

Trains FunkSVD model.Arguments:

__data__ : Training data as numpy array.\
__test_split__ : Split data into train and test set. Default is True.
__test_portion__ : Portion of test set. Default is 0.10.
__search_parameter_space__ : If true, data will not split into train and test sets again.  

### get_recommendation_for_existing_user():

Gets recommendations for existing user that are not rated by user. Arguments:
__user_id :__ Existing user id
__howMany :__ Count of recommended items to be returned. Default is 10.

### get_recommendation_for_new_user(): 

Gets recommendations for new user depending on given similarity measure. Arguments:

__user_ratings :__ A python dictionary of items and corresponding scores.

__similarity_measure :__ Similarity measures can be:
* Cosine Similarity
* Pearson Correlation
* Adjusted Cosine Similarity
* Weighted Cosine Similarity
* Constrained Pearson Correlation
* Mean Squared Difference.

__howManyUsers :__ Count of most similar users to be used for recommendation. Default is 3

__howManyItems :__ Count of recommended items to be returned. Default is 5.

### get_similar_products():
Gets most similar items. Arguments:
__item_id :__ Id of the item.

__howMany :__ Count of similar items to be returned.


### Note
This library is created as a part of AI Projects program @ inzva. You can see more at inzva.com
