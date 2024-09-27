# Movie Recommendation System

## Overview

This project implements a movie recommendation system using content-based filtering. The system allows users to search for movies, view detailed movie information, and receive recommendations based on the movie they search for. The system is built using **Flask**, **MongoDB**, and **machine learning** techniques for recommendations and sentiment analysis.

The project is capable of loading movie data, processing user queries, and generating recommendations based on the similarity of movie attributes such as **genres**, **actors**, and **directors**. It also provides offline evaluation of the recommendation model using precision, recall, F1 score, and accuracy metrics.

## Features

1. **Movie Search & Recommendations**: Users can search for a movie and receive recommendations based on the movie's content.
2. **Add New Movies**: Movies can be added to the system using a web interface.
3. **Movie Details & Reviews**: The system fetches and displays movie details such as the cast, overview, and user reviews from IMDb.
4. **Offline Testing**: Evaluates the performance of the recommendation system using offline data testing with metrics like precision, recall, F1 score, and accuracy.
5. **Sentiment Analysis**: Reviews fetched from IMDb are analyzed for sentiment using a pre-trained model to classify them as "Good" or "Bad."

## Project Structure

- **app.py**: The main Flask application. Handles routes, requests, and the recommendation engine.
- **templates/**: Contains HTML files for rendering the web interface (`home.html`, `add_movie.html`, `recommend.html`).
- **static/**: Contains static assets such as CSS and JavaScript files.
- **recommendation_engine.py**: Contains functions to create the similarity matrix and recommend movies based on content similarity.
- **main_data.csv**: The main dataset file containing movie information.
- **MongoDB**: Used for storing and retrieving movie data. Collections include `movies` for storing movie details and `predictions` for offline evaluation.
- **NLP Model & Vectorizer**: Pre-trained models (`nlp_model.pkl` and `transform.pkl`) are used for sentiment analysis on movie reviews.

## Requirements

To run the project, install the following dependencies:

```bash
pip install Flask
pip install pymongo
pip install numpy
pip install pandas
pip install scikit-learn==1.5.1
pip install bs4
pip install lxml
pip install pickle-mixin
```

A VPN is needed to fully load the images and reviews fetched from external sources like IMDb.

### MongoDB Setup

1. Make sure MongoDB is running locally or on a server.
2. A MongoDB collection named `movies` is used to store movie data.
3. Use the `load_data_to_mongo()` function to load data from the `main_data.csv` file into MongoDB.

## Key Components

### 1. **Flask Application**

The main application (`app.py`) provides routes to handle movie search, recommendations, and the addition of new movies.

- **Home Page (`/home`)**: Displays suggestions for movies.
- **Similarity Route (`/similarity`)**: Takes user input and generates recommendations based on the movie title.
- **Recommendation Route (`/recommend`)**: Fetches and displays movie details including cast, reviews, and similar movies.
- **Add Movie Route (`/add_movie`)**: Allows the user to add new movies to the database.

### 2. **Content-Based Recommendation Engine**

The recommendation engine uses movie attributes such as genres, actors, and directors to compute a similarity matrix and provide recommendations.

- **create_similarity()**: Creates a similarity matrix for all movies based on their combined features (genres, actors, directors).
- **rcmd()**: Recommends movies similar to the searched movie based on the similarity matrix.

### 3. **Offline Evaluation**

The system includes an offline testing function (`offline_test()`) to evaluate the performance of the recommendation engine using the following metrics:

- **Precision**
- **Recall**
- **F1 Score**
- **Accuracy**

### 4. **Sentiment Analysis**

Reviews from IMDb are analyzed using a pre-trained NLP model to classify them as either "Good" or "Bad." This functionality enhances user experience by providing sentiment-based feedback on reviews.

### 5. **Logging**

The application uses Python’s `logging` module to log significant events such as data loading, movie addition, and recommendations.

## Usage

### Running the Application

1. Make sure MongoDB is running.
2. Run the Flask application:

```bash
python main.py
```

### Adding a Movie

1. Navigate to the `/add_movie` page.
2. Enter movie details including the title, director, actors, and genres.
3. The movie will be added to the MongoDB collection if it doesn't already exist.

### Recommendations

1. Search for a movie on the home page.
2. Receive a list of similar movies based on the content filtering algorithm.

## Future Enhancements

- **User-Based Filtering**: Add a user-based collaborative filtering system to enhance recommendations by learning user preferences.
- **Real-Time Recommendations**: Implement real-time recommendation generation for users with dynamic inputs.
- **Improved Sentiment Analysis**: Train a custom sentiment analysis model for more accurate results.

## Acknowledgements

- Flask documentation for providing insights into web development.
- IMDb for providing movie reviews.
- Scikit-learn for machine learning algorithms used in the recommendation system.

---
