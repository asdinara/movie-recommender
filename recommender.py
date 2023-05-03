'''Implements functins for making movie recommendations'''
import pandas as pd
import numpy as np
import pickle
import random
from util import df_NMF, fitted_model, num_recommendable_movies, USERS, MOVIES, movie_mean, Q

def random_recommender():
    random.shuffle(MOVIES)
    top_two = MOVIES[0:2]
    return top_two

def NMF(query, model=fitted_model, num_recommendable_movies=num_recommendable_movies):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    # 1. candidate generation
    user_input = pd.DataFrame(query, index=['new_users'], columns=MOVIES)
    user_input_imputed = user_input.fillna(value=movie_mean)
    
    # 2. construct new_user-item dataframe given the query
    P_user = model.transform(user_input_imputed)
    P_user = pd.DataFrame(P_user, index=['new_user'])
    
    # 3. scoring
    R_user_hat = np.dot(P_user, Q)
    R_user_hat = pd.DataFrame(R_user_hat, columns=MOVIES, index=['new_user'])
    # 4. ranking
    R_user_hat_transposed = R_user_hat.T.sort_values(by='new_user', ascending=False)
    # filter out movies already seen by the user
    user_initial_ratings_list = list(query.keys())
    # return the top-k highest rated movie ids or titles
    recommendables = list(R_user_hat_transposed.index)
    recommendations = [movie for movie in recommendables if movie not in user_initial_ratings_list][:num_recommendable_movies]
    return recommendations