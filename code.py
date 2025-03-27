import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import ast
import numpy as np
import re
from gensim.models import Word2Vec
df_final = pd.read_csv("oneHot.csv")  
df = pd.read_csv("movieData.csv")  
merged_combined_df = pd.read_csv("News.csv")  

!pip install sentence-transformers

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize model once (outside the function for efficiency).
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_movies(user_inputs, df, df_final):

    # 1) Identify the "input movies" by partial matching on title
    input_descriptions = []
    input_genres_set = set()  # Will store all genres from matched input movies

    for segment in user_inputs:
        # Find rows where 'title' contains the segment (case-insensitive)
        matched_rows = df[df['title'].str.contains(segment, case=False, na=False)]

        # Collect their descriptions (original text, for user-friendly display)
        for desc in matched_rows['description'].fillna(''):
            input_descriptions.append(desc)

        # Collect their mapped genres
        for mg_list in matched_rows['genres']:
            if isinstance(mg_list, list):
                input_genres_set.update(mg_list)

    # If no movies were matched, return early
    if not input_descriptions:
        return {
            "Recommended Movies": [],
            "Recommended Movies Descriptions": [],
            "Recommended Movies Genres": [],
            "Similarity Scores": [],
            "Input Genres": [],
            "All Descriptions": [],
            "All Unique Genres": [],
            "Message": "No movies matched the provided title segments."
        }

    # 2) Create a combined embedding from all matched input movies
    #    We'll use the stemmed descriptions for embedding
    matched_stemmed = []
    for segment in user_inputs:
        # This time gather the 'description_stemmed' for matched rows
        matched_rows = df[df['title'].str.contains(segment, case=False, na=False)]
        for d_stemmed in matched_rows['description_stemmed'].fillna(''):
            matched_stemmed.append(d_stemmed)

    # Encode and average
    user_desc_embs = bert_model.encode(matched_stemmed)
    user_input_emb = np.mean(user_desc_embs, axis=0)  # shape: (embedding_dim,)

    # 3) Compute similarity for all movies in df
    #    (A) BERT embeddings
    all_stemmed = df['description_stemmed'].fillna('').tolist()
    movie_desc_embs = bert_model.encode(all_stemmed)  # shape: (num_movies, embedding_dim)

    semantic_similarity = cosine_similarity(
        [user_input_emb],       # shape: (1, embedding_dim)
        movie_desc_embs         # shape: (num_movies, embedding_dim)
    ).flatten()                  # shape: (num_movies,)

    #    (B) Categorical similarity
    movie_genres_sets = df['genres'].apply(
        lambda glist: set(glist) if isinstance(glist, list) else set()
    )
    categorical_sim_list = []
    for mg_set in movie_genres_sets:
        union_size = len(input_genres_set | mg_set)
        if union_size == 0:
            categorical_sim_list.append(0.0)
        else:
            intersection_size = len(input_genres_set & mg_set)
            categorical_sim_list.append(intersection_size / union_size)

    categorical_similarity = np.array(categorical_sim_list)

    # Combine them
    combined_similarity = 0.7 * semantic_similarity + 0.3 * categorical_similarity

    # 4) Find top 5 recommended movies
    df['similarity'] = combined_similarity
    recommendations = df.nlargest(5, 'similarity')

    # 5) Build the output data
    recommended_movies = recommendations['title'].tolist()
    recommended_descriptions = recommendations['description'].fillna('').tolist()
    recommended_genres_lists = recommendations['genres'].tolist()
    similarity_scores = recommendations['similarity'].tolist()

    # 6) Build the combined lists requested:
    #    (A) All descriptions: input movies + recommended movies
    all_descriptions = input_descriptions + recommended_descriptions

    #    (B) All unique genres: union of input genres + recommended genres
    recommended_genres_set = set()
    for mg_list in recommended_genres_lists:
        if isinstance(mg_list, list):
            recommended_genres_set.update(mg_list)
    all_unique_genres = list(input_genres_set.union(recommended_genres_set))

    # 7) Return everything in a dictionary
    return {
        "Recommended Movies": recommended_movies,
        "Recommended Movies Descriptions": recommended_descriptions,
        "Recommended Movies Genres": recommended_genres_lists,
        "Similarity Scores": similarity_scores,
        "Input Genres": list(input_genres_set),
        "All Descriptions": all_descriptions,
        "All Unique Genres": all_unique_genres
    }


# Example usage
user_inputs = ["Titan", "Avatar", "Furious"]
result = recommend_movies(user_inputs, df, df_final)

#all unique categories
unique_categories = merged_combined_df['category'].dropna().unique()

import ast  # Import Abstract Syntax Tree for safe string-to-list conversion

recommended_categories = set()  # Use a set to avoid duplication
recommended_movies = result["Recommended Movies"]

for title in recommended_movies:
    # Find the index of the movie in the DataFrame
    index = df[df['title'] == title].index[0]

    # Convert string representation of a list into an actual list if needed
    raw_categories = df.loc[index, 'mapped_categories']
    if isinstance(raw_categories, str):
        try:
            raw_categories = ast.literal_eval(raw_categories)  # Safely convert string to list
        except (ValueError, SyntaxError):
            raw_categories = []  # If conversion fails, use an empty list

    # Filter out 'Other' and add the rest to the set
    recommended_categories.update(
        category for category in raw_categories if category != "Other"
    )

# Optionally, convert the set back to a list if needed
recommended_categories = list(recommended_categories)

filtered_news = merged_combined_df[merged_combined_df['category'].isin(recommended_categories)]

# Print and store each news title and link
news_list = []
for _, row in filtered_news.iterrows():
    # print(f"Title: {row['title']}, Link: {row['link']}")
    news_list.append({'title': row['title'], 'category':row['category'], 'link': row['link']})

# Optionally store the filtered news in a DataFrame or save it
filtered_news_df = pd.DataFrame(news_list)

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Example user input for the news recommendation
user_input_text = "latest technology and business trends"

# 1) Filter news by recommended_categories
filtered_news = merged_combined_df[merged_combined_df['category'].isin(recommended_categories)]

# Convert filtered rows into a DataFrame
news_list = []
for _, row in filtered_news.iterrows():
    news_list.append({
        'title': row['title'],
        'category': row['category'],
        'link': row['link']
    })

filtered_news_df = pd.DataFrame(news_list)

# 2) Use BERT to compute similarity using the news titles
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# (A) Encode the user input
user_embed = bert_model.encode([user_input_text])  # shape: (1, embedding_dim)

# (B) Encode the news titles
news_titles = filtered_news_df['title'].fillna('').tolist()
news_embeds = bert_model.encode(news_titles)       # shape: (num_rows, embedding_dim)

# (C) Compute cosine similarity
similarities = cosine_similarity(user_embed, news_embeds).flatten()

# 3) Add the similarity scores to filtered_news_df
filtered_news_df['similarity'] = similarities

# 4) Sort or select top N articles by similarity
top_n = 5
recommended_news_df = filtered_news_df.nlargest(top_n, 'similarity')