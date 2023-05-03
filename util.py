import numpy as np
from sklearn.decomposition import NMF
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from random import randint

#define dataframes and matrices
##NMF
df_NMF = pd.read_csv('./data/user_movie.csv', index_col=0)
USERS = list(df_NMF.index)
MOVIES = list(df_NMF.columns)

#define contsants
num_similar_users = 10
num_recommendable_movies = 10

#define variables
movie_mean = df_NMF.mean()

#NMF model
R = df_NMF
NMF_model = NMF(n_components=40, max_iter=100,  init='random', random_state=10)
NMF_model.fit(R)
# movie-features matrix
Q = NMF_model.components_  
# user-feature matrix
P = NMF_model.transform(R)

## The reconstructed matrix!
R_hat = np.dot(P, Q)

#dump pickle
with open('./data/NMF_model.pkl', 'wb') as file_out:
    pickle.dump(NMF_model, file_out)

#NMF fitted model
with open('./data/NMF_model.pkl', 'rb') as file_in:
    fitted_model = pickle.load(file_in)

