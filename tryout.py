from FunkSVD import FunkSVD
import numpy as np


data = np.genfromtxt('./data/movielens100k.csv', delimiter=',', skip_header=1)


myFunk = FunkSVD()
myFunk.set_hyperparameters(max_epoch=10, early_stopping=5, n_latent=45)
myFunk.fit(data)

print(myFunk.get_recommendation_for_existing_user('1'))

print(myFunk.get_similar_products('31'))

print(myFunk.get_recommendation_for_new_user({'301': 1}))
