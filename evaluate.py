import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from recommendation_engine import rcmd

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['movie_db']  # Replace with your database name
predictions_collection = db['predictions']  # Collection where predictions are stored


def fetch_ground_truth():
    # Fetch ground truth data from MongoDB predictions collection
    data = pd.DataFrame(list(predictions_collection.find()))
    return data


def evaluate_recommendations():
    ground_truth = fetch_ground_truth()

    # Initialize lists to store results for evaluation
    y_true = []
    y_pred = []

    # Iterate over each movie title in ground truth
    for index, row in ground_truth.iterrows():
        movie_title = row['movie_title']
        true_recommendations = set(row['recommended_movies'].split('|'))

        # Generate recommendations for the same movie
        predicted_recommendations = set(rcmd(movie_title.lower()))

        # Here we're using set intersection for binary classification, modify as per your needs
        intersection = true_recommendations.intersection(predicted_recommendations)

        y_true.append(1 if intersection else 0)
        y_pred.append(1 if predicted_recommendations else 0)

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    evaluate_recommendations()
