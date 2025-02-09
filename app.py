from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast

app = Flask(__name__)

# Load datasets
df_final = pd.read_csv("oneHot.csv")  
df = pd.read_csv("movieData.csv")  
merged_combined_df = pd.read_csv("News.csv")  

# Initialize the Sentence Transformer model once
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_movies(user_inputs, df, df_final):
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
        return []

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

    movie_genres_sets = df['genres'].apply(lambda glist: set(glist) if isinstance(glist, list) else set())
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

    return recommended_movies, recommended_descriptions

def recommend_news(movies, df, merged_combined_df):
    recommended_categories = set()

    for title in movies:
        index = df[df['title'] == title].index[0]
        raw_categories = df.loc[index, 'mapped_categories']
        if isinstance(raw_categories, str):
            try:
                raw_categories = ast.literal_eval(raw_categories)
            except (ValueError, SyntaxError):
                raw_categories = []
        recommended_categories.update(category for category in raw_categories if category != "Other")

    filtered_news = merged_combined_df[merged_combined_df['category'].isin(recommended_categories)]
    return filtered_news[['title', 'category', 'link']].head(5).to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    news_recommendations = []
    
    if request.method == "POST":
        user_input = request.form["keyword"]
        movies, descriptions = recommend_movies(user_input.split(), df, df_final)
        news_recommendations = recommend_news(movies, df, merged_combined_df)

        recommendations = list(zip(movies, descriptions))
    
    return render_template("index.html", recommendations=recommendations, news_recommendations=news_recommendations)

if __name__ == "__main__":
    app.run(debug=True)
