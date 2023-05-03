from flask import Flask, render_template, request
from recommender import random_recommender, MOVIES, NMF

#__name__ reserved word
app= Flask(__name__)

#decorator that takes
@app.route('/')
def homepage():
    return render_template('home.html')

#decorator to take functions output and do smth with it in a browser

@app.route('/results')
def NMF_recommender():
    user_query = request.args.to_dict()
    user_movie_ratings = {}

    # iterate through the user_query dictionary
    for i in range(1, 4):
        movie_name_key = f"movie{i}_name"
        movie_rating_key = f"movie{i}_rating"

        # check if both movie_name_key and movie_rating_key are in the user_query
        if movie_name_key in user_query and movie_rating_key in user_query:
            movie_name = user_query[movie_name_key]
            movie_rating = int(user_query[movie_rating_key])

            # add the movie name and rating to the dictionary
            user_movie_ratings[movie_name] = movie_rating

    recommendations = NMF(query=user_movie_ratings)
    return render_template('/results.html', movies = recommendations)

#if i don't do it, then nothing would be executed and it will take only recommender name
if __name__ == '__main__':
    app.run(port=5040, debug=True)