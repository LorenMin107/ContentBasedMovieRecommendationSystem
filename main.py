import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import json
import bs4 as bs
import urllib.request
import pickle
from pymongo import MongoClient
from recommendation_engine import rcmd, create_similarity

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['movie_db']  # Replace with your database name
collection = db['movies']  # Replace with your collection name

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))

def load_data_to_mongo():
    data = pd.read_csv('main_data.csv')
    data.fillna('', inplace=True)

    # Drop duplicates before inserting into MongoDB
    data = data.drop_duplicates(subset=['movie_title'])

    # Convert to dictionary format for MongoDB
    data_dict = data.to_dict("records")

    # Insert only if there are no duplicates in the database
    for record in data_dict:
        if collection.count_documents({'movie_title': record['movie_title']}, limit=1) == 0:
            collection.insert_one(record)

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data = pd.DataFrame(list(collection.find()))
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route('/add_movie')
def add_movie():
    return render_template('add_movie.html')

@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    data, similarity_matrix = create_similarity()
    rc = rcmd(movie, data, similarity_matrix, store_predictions=True)
    if isinstance(rc, str):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    suggestions = get_suggestions()

    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in
                    range(len(cast_places))}

    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = []
    reviews_status = []
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
                           vote_count=vote_count, release_date=release_date, runtime=runtime, status=status,
                           genres=genres, movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details)

@app.route('/add_movie', methods=["POST"])
def add_movie_to_db():
    movie_title = request.form['movie_title'].strip().lower()
    director_name = request.form['director_name'].strip()
    actor_1_name = request.form['actor_1_name'].strip()
    actor_2_name = request.form['actor_2_name'].strip()
    actor_3_name = request.form['actor_3_name'].strip()
    genres = request.form['genres'].strip()

    comb = ' '.join(filter(None, [actor_1_name, actor_2_name, actor_3_name, director_name, genres]))

    movie = {
        "movie_title": movie_title,
        "director_name": director_name,
        "actor_1_name": actor_1_name,
        "actor_2_name": actor_2_name,
        "actor_3_name": actor_3_name,
        "genres": genres,
        "comb": comb
    }

    # Check if the movie already exists
    if collection.count_documents({'movie_title': movie_title}, limit=1) == 0:
        collection.insert_one(movie)
        create_similarity()  # Update the similarity matrix after adding new movie
        return jsonify("Movie added successfully!")
    else:
        return jsonify("Movie already exists in the database!")

if __name__ == '__main__':
    load_data_to_mongo()
    app.run(debug=True)
