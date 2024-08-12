import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
import bs4 as bs
import urllib.request
import pickle
import csv
from datetime import date, datetime

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))

# Load movie data
movies_df = pd.read_csv('main_data.csv')

# Create TF-IDF matrix and similarity matrix
def create_similarity():
    global movies_df
    try:
        tfidf_matrix = vectorizer.fit_transform(movies_df['comb'])
        similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        movies_df['similarity'] = similarity_matrix.tolist()
        print(movies_df.head())  # Debugging line to check the DataFrame
        movies_df.to_csv('main_data_with_similarity.csv', index=False)
        print("Similarity matrix created and saved successfully.")
    except Exception as e:
        print(f"Error in create_similarity: {e}")


# Convert list of strings to list
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

# Convert list of numbers to list
def convert_to_list_num(my_list):
    my_list = my_list.split(',')
    my_list[0] = my_list[0].replace("[", "")
    my_list[-1] = my_list[-1].replace("]", "")
    return my_list

def get_suggestions():
    return list(movies_df['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route('/add-movie', methods=['POST'])
def add_movie():
    # Get form data
    movie_title = request.form['movie_title']
    director_name = request.form['director_name']
    actor_1_name = request.form['actor_1_name']
    actor_2_name = request.form['actor_2_name']
    actor_3_name = request.form['actor_3_name']
    genres = request.form['genres']
    overview = request.form['overview']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    rating = request.form['rating']
    vote_count = request.form['vote_count']

    # Combine relevant fields for the 'comb' column
    comb = f"{actor_1_name} {actor_2_name} {actor_3_name} {director_name} {genres}"

    # Create a new row for the CSV file
    new_row = [director_name, actor_1_name, actor_2_name, actor_3_name, genres, movie_title, comb]

    # Append the new movie data to the CSV file
    with open('main_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(new_row)

    # Recalculate similarity matrix after adding a new movie
    create_similarity()

    return redirect(url_for('admin'))

@app.route("/populate-matches", methods=["POST"])
def populate_matches():
    # Getting data from AJAX request
    res = json.loads(request.get_data())
    movies_list = res['movies_list']

    movie_cards = {
        "https://image.tmdb.org/t/p/original" + movies_list[i]['poster_path'] if movies_list[i]['poster_path'] else "/static/movie_placeholder.jpeg":
        [movies_list[i]['title'], movies_list[i]['original_title'], movies_list[i]['vote_average'],
         datetime.strptime(movies_list[i]['release_date'], '%Y-%m-%d').year if movies_list[i]['release_date'] else "N/A",
         movies_list[i]['id']] for i in range(len(movies_list))
    }

    return render_template('recommend.html', movie_cards=movie_cards)

@app.route("/recommend", methods=["POST"])
def recommend():
    # Getting data from AJAX request
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
    rel_date = request.form['rel_date']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']
    rec_movies_org = request.form['rec_movies_org']
    rec_year = request.form['rec_year']
    rec_vote = request.form['rec_vote']
    rec_ids = request.form['rec_ids']

    # Get movie suggestions for autocomplete
    suggestions = get_suggestions()

    # Call the convert_to_list function for every string that needs to be converted to list
    rec_movies_org = convert_to_list(rec_movies_org)
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # Convert string to list (e.g., "[1,2,3]" to [1,2,3])
    cast_ids = convert_to_list_num(cast_ids)
    rec_vote = convert_to_list_num(rec_vote)
    rec_year = convert_to_list_num(rec_year)
    rec_ids = convert_to_list_num(rec_ids)

    # Rendering the string to Python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    for i in range(len(cast_chars)):
        cast_chars[i] = cast_chars[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # Combining multiple lists as a dictionary
    movie_cards = {rec_posters[i]: [rec_movies[i], rec_movies_org[i], rec_vote[i], rec_year[i], rec_ids[i]] for i in range(len(rec_posters))}

    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    if imdb_id:
        # Web scraping to get user reviews from IMDb site
        sauce = urllib.request.urlopen(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()
        soup = bs.BeautifulSoup(sauce, 'lxml')
        soup_result = soup.find_all("div", {"class": "text show-more__control"})

        reviews_list = []  # List of reviews
        reviews_status = []  # List of comments (good or bad)
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # Passing the review to our model
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Positive' if pred else 'Negative')

        # Getting current date
        movie_rel_date = ""
        curr_date = ""
        if rel_date:
            today = str(date.today())
            curr_date = datetime.strptime(today, '%Y-%m-%d')
            movie_rel_date = datetime.strptime(rel_date, '%Y-%m-%d')

        # Combining reviews and comments into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

        # Passing all the data to the HTML file
        return render_template('recommend.html', title=title, poster=poster, overview=overview,
                               vote_average=vote_average, vote_count=vote_count, release_date=release_date,
                               movie_rel_date=movie_rel_date, curr_date=curr_date, runtime=runtime, status=status,
                               genres=genres, movie_cards=movie_cards, reviews=movie_reviews, casts=casts,
                               cast_details=cast_details)

    else:
        return render_template('recommend.html', title=title, poster=poster, overview=overview,
                               vote_average=vote_average, vote_count=vote_count, release_date=release_date,
                               movie_rel_date="", curr_date="", runtime=runtime, status=status, genres=genres,
                               movie_cards=movie_cards, reviews="", casts=casts, cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)
