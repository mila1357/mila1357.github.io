from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load Data
df = pd.read_csv("movies_data.csv")  # Replace with actual dataset file
df_final = pd.read_csv("movies_data_final.csv")

# Load BERT Model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/recommend", methods=["POST"])
def recommend_movies():
    data = request.json
    user_inputs = data.get("movies", [])

    input_descriptions = []
    input_genres_set = set()

    for segment in user_inputs:
        matched_rows = df[df['title'].str.contains(segment, case=False, na=False)]
        for desc in matched_rows['description'].fillna(''):
            input_descriptions.append(desc)
        for mg_list in matched_rows['genres']:
            if isinstance(mg_list, list):
                input_genres_set.update(mg_list)

    if not input_descriptions:
        return jsonify({
            "Recommended Movies": [],
            "Recommended Movies Descriptions": [],
            "Recommended Movies Genres": [],
            "Similarity Scores": [],
            "Input Genres": [],
            "All Descriptions": [],
            "All Unique Genres": [],
            "Message": "No movies matched the provided title segments."
        })

    matched_stemmed = []
    for segment in user_inputs:
        matched_rows = df[df['title'].str.contains(segment, case=False, na=False)]
        for d_stemmed in matched_rows['description_stemmed'].fillna(''):
            matched_stemmed.append(d_stemmed)

    user_desc_embs = bert_model.encode(matched_stemmed)
    user_input_emb = np.mean(user_desc_embs, axis=0)

    all_stemmed = df['description_stemmed'].fillna('').tolist()
    movie_desc_embs = bert_model.encode(all_stemmed)

    semantic_similarity = cosine_similarity([user_input_emb], movie_desc_embs).flatten()

    movie_genres_sets = df['genres'].apply(
        lambda glist: set(glist) if isinstance(glist, list) else set()
    )
    categorical_sim_list = [
        len(input_genres_set & mg_set) / len(input_genres_set | mg_set) if len(input_genres_set | mg_set) > 0 else 0
        for mg_set in movie_genres_sets
    ]
    categorical_similarity = np.array(categorical_sim_list)

    combined_similarity = 0.7 * semantic_similarity + 0.3 * categorical_similarity

    df['similarity'] = combined_similarity
    recommendations = df.nlargest(5, 'similarity')

    recommended_movies = recommendations['title'].tolist()
    recommended_descriptions = recommendations['description'].fillna('').tolist()
    recommended_genres_lists = recommendations['genres'].tolist()

    return jsonify({
        "Recommended Movies": recommended_movies,
        "Recommended Movies Descriptions": recommended_descriptions,
        "Recommended Movies Genres": recommended_genres_lists
    })

if __name__ == "__main__":
    app.run(debug=True)
