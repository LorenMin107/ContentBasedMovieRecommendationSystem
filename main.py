import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import json
import bs4 as bs
import urllib.request
import pickle
from pymongo import MongoClient
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import signal
import sys
from recommendation_engine import clean_genre_data
from recommendation_engine import rcmd, create_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('app.log', mode='a')
                    ])

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['movie_db']  # Replace with your database name
collection = db['movies']  # Replace with your collection name
predictions_collection = db['predictions']  # Collection for storing predictions

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))

def signal_handler(sig, frame):
    print('Interrupt received, shutting down...')
    # Perform any cleanup here if necessary
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def load_data_to_mongo():
    logging.info('Loading data into MongoDB...')
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
            logging.info(f'Inserted record: {record["movie_title"]}')
        else:
            logging.info(f'Record already exists: {record["movie_title"]}')

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    logging.info('Fetching movie suggestions...')
    data = pd.DataFrame(list(collection.find()))
    return list(data['movie_title'].str.capitalize())

def offline_test():
    logging.info('Starting offline testing...')
    # Split data into training and testing sets
    data = pd.DataFrame(list(collection.find())).drop_duplicates(subset=['movie_title'])
    data = data.reset_index(drop=True)
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Create similarity matrix using training data
    train_data, similarity_matrix = create_similarity()

    # Generate recommendations for test data
    test_data['recommendations'] = test_data['movie_title'].apply(lambda x: rcmd(x, train_data, similarity_matrix))

    # Evaluation
    y_true = []
    y_pred = []

    for _, row in test_data.iterrows():
        movie_title = row['movie_title']
        true_recommendations = set(row['recommendations'])  # No split needed, it's already a list

        # Generate recommendations for the same movie
        predicted_recommendations = set(rcmd(movie_title.lower(), train_data, similarity_matrix))
        logging.info(f"Movie: {movie_title} - Predicted Recommendations: {predicted_recommendations}")

        # Calculate precision, recall, F1 score, and accuracy
        intersection = true_recommendations.intersection(predicted_recommendations)

        y_true.append(1 if intersection else 0)
        y_pred.append(1 if predicted_recommendations else 0)

    # Calculate evaluation metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F1 Score: {f1}')
    logging.info(f'Accuracy: {accuracy}')


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
    rc = rcmd(movie, data, similarity_matrix, store_predictions=True) # Removed store_predictions=True
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

    logging.info(f'Recommendations for movie: {title} - Found {len(movie_reviews)} reviews')

    return render_template('recommend.html', title=title, poster=poster, overview=overview,
                           vote_average=vote_average, vote_count=vote_count, release_date=release_date, runtime=runtime,
                           status=status, genres=genres, movie_cards=movie_cards, reviews=movie_reviews, casts=casts,
                           cast_details=cast_details)

def standardize_genres(genres):
    return ', '.join(genre.strip().replace('Science Fiction', 'Sci-Fi') for genre in genres.split(', '))

@app.route('/add_movie', methods=["POST"])
def add_movie_to_db():
    movie_title = request.form['movie_title'].strip().lower()
    director_name = request.form['director_name'].strip()
    actor_1_name = request.form['actor_1_name'].strip()
    actor_2_name = request.form['actor_2_name'].strip()
    actor_3_name = request.form['actor_3_name'].strip()
    genres = standardize_genres(request.form['genres'].strip())

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
        logging.info(f'Added movie to database: {movie_title}')
        create_similarity()  # Update the similarity matrix after adding new movie
        return jsonify("Movie added successfully!")
    else:
        logging.info(f'Movie already exists in the database: {movie_title}')
        return jsonify("Movie already exists in the database!")


if __name__ == '__main__':
    # Perform data cleaning
    clean_genre_data()

    # load_data_to_mongo()
    # offline_test()  # Call offline test
    app.run(debug=True)
