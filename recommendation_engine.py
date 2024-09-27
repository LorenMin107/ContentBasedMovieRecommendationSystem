# recommendation_engine.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('recommendation_engine.log', mode='a')
                    ])

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['movie_db']
collection = db['movies']
predictions_collection = db['predictions']  # Collection for storing predictions

def clean_genre_data():
    logging.info('Cleaning genre data...')
    data = pd.DataFrame(list(collection.find()))

    # Clean genre data
    data['genres'] = data['genres'].apply(lambda x: ' '.join(genre.strip().rstrip(',') for genre in x.split()))

    # Update MongoDB with cleaned data
    for _, row in data.iterrows():
        collection.update_one(
            {'_id': row['_id']},
            {'$set': {'genres': row['genres']}}
        )

    logging.info('Genre data cleaned.')

def create_similarity():
    logging.info('Creating similarity matrix...')

    # Clean genre data before creating similarity matrix
    clean_genre_data()

    data = pd.DataFrame(list(collection.find())).drop_duplicates(subset=['movie_title'])
    if data.empty:
        logging.error("No data found in MongoDB.")
        raise ValueError("No data found in MongoDB.")

    # Combine relevant features
    data['combined_features'] = data['genres'] * 3 + ' ' + data['director_name'] + ' ' + \
                                data['actor_1_name'] + ' ' + data['actor_2_name'] + ' ' + \
                                data['actor_3_name'] + ' ' + data['comb']

    # Apply TfidfVectorizer
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    count_matrix = tfidf.fit_transform(data['combined_features'])

    # Calculate similarity matrix
    similarity = cosine_similarity(count_matrix)

    logging.info('Similarity matrix created.')
    return data, similarity

def rcmd(m, data=None, similarity=None, store_predictions=False):
    m = m.lower()
    if data is None or similarity is None:
        data, similarity = create_similarity()

    if m not in data['movie_title'].str.lower().unique():
        logging.warning(f"Movie '{m}' not found in the database.")
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'

    i = data.loc[data['movie_title'].str.lower() == m].index[0]
    # logging.info(f'Index of movie "{m}": {i}') # uncomment to debug

    input_movie_genres = data['genres'].iloc[i].split()
    # logging.info(f'Genres of input movie: {input_movie_genres}') # uncomment to debug

    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)

    recommendations = []
    count = 0

    for j in lst:
        index = j[0]
        recommended_movie = data['movie_title'].iloc[index]
        recommended_movie_genres = data['genres'].iloc[index].split()

        if recommended_movie.lower() != m and recommended_movie not in recommendations:
            if any(genre in recommended_movie_genres for genre in input_movie_genres):
                recommendations.append(recommended_movie)
                count += 1

        if count >= 10:
            break

    # logging.info(f'Recommended movies for "{m}": {recommendations}') # uncomment to debug

    # Store predictions if required
    if store_predictions:
        prediction = {
            "movie_title": m,
            "recommended_movies": "|".join(recommendations)
        }
        # Store in MongoDB
        predictions_collection.insert_one(prediction)
        logging.info(f'Stored predictions for movie: {m}')

    return recommendations

