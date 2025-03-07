import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
import chromadb
import os

parser = argparse.ArgumentParser(description="Movie Recommendation System with ChromaDB")
parser.add_argument("--force-save", action="store_true", help="Force saving embeddings to ChromaDB")
args = parser.parse_args()

#  Load CSV files
print("\n📥 Loading data...")
movies_df = pd.read_csv("./ml-latest-small/movies.csv")
ratings_df = pd.read_csv("./ml-latest-small/ratings.csv")
print("🔍 Loading AI model...\n")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#  Create a vector database in ChromaDB
print("💾 Connecting to ChromaDB...\n")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Colection to store movie
movies_collection = chroma_client.get_or_create_collection(name="movies")

# Colection to store user
ratings_collection = chroma_client.get_or_create_collection(name="user_ratings")

if args.force_save:
    print("🚀 Force saving embeddings to ChromaDB...\n")
    # Clear datas before populate again
    movies_collection.delete(where={})
    ratings_collection.delete(where={})

    print("⚡ Saving movies...\n")

    for index, row in movies_df.iterrows():
        movie_id = str(row["movieId"])
        movie_description = f"Title: {row['title']} | Genres: {row['genres']}"
        
        # Generate embedding
        embedding = model.encode(movie_description).tolist()

        # Save in ChromaDB
        movies_collection.add(
            ids=[movie_id],
            embeddings=[embedding],
            metadatas=[{"title": row["title"], "genres": row["genres"]}]
        )

    print("✅ Movies stored successfully!\n")

    print("⚡ Saving user ratings...\n")

    for index, row in ratings_df.iterrows():
        user_id = str(row["userId"])
        movie_id = str(row["movieId"])
        rating = row["rating"]

        # Saving relationship between user and rating in  ChromaDB
        ratings_collection.add(
            ids=[f"{user_id}_{movie_id}"],  # Unique ID combination user + movie
            embeddings=[[rating]],  # Only rating with simple embedding
            metadatas=[{"user_id": user_id, "movie_id": movie_id, "rating": rating}]
        )

    print("✅ User ratings stored successfully!\n")
else:
    print("🔄 Using existing ChromaDB embeddings without saving...\n")

# Function to recommend movies for a specific user
def recommend_movies(user_id):
    user_id = str(user_id)

    # Retrieve user rating
    user_ratings = ratings_collection.get(where={"user_id": user_id})

    if not user_ratings["metadatas"]:
        print("User not found.")
        return

    # Sort movies by best ratings
    user_movies = sorted(
        user_ratings["metadatas"],
        key=lambda x: x["rating"],
        reverse=True
    )

    print("\nTop 10 movies rated by user:\n")

    top_movies = user_movies[:10]

    for idx, movie_data in enumerate(top_movies, start=1):
        movie_id = movie_data["movie_id"]
        rating = movie_data["rating"]

        # Search for movie information
        movie_info = movies_collection.get([movie_id])["metadatas"][0]
        title = movie_info.get("title", "Unknown")
        genres = movie_info.get("genres", "Unknown")

        print(f" **Movie {idx}:**")
        print(f" Title: {title}")
        print(f" Genres: {genres}")
        print(f" Rating: {rating}")
        print("-" * 60)
        print("\n")

def recommend_similar_movies(movie_title):
    # Verify if movie is present in dataset
    movie_data = movies_df[movies_df["title"].str.lower() == movie_title.lower()]
    
    if movie_data.empty:
        print(" Movie not found.")
        return
    
    # movie_id = str(movie_data["movieId"].values[0])
    movie_genres = movie_data["genres"].values[0]

    # Criar uma descrição baseada no gênero
    query_description = f"Genres: {movie_genres}"
    
    # Generate movie embedding
    query_embedding = model.encode([query_description]).tolist()

    # Search on ChromaDB the 10 movies more similars
    results = movies_collection.query(
        query_embeddings=query_embedding,
        n_results=10
    )

    print("\n Top 10 recommended movies based on genre similarity:\n")

    for idx, res in enumerate(results["metadatas"][0], start=1):
        title = res.get("title", "Unknown")
        # description = res.get("description", "Unknown")
        genres = res.get("genres", "Unknown")

        print(f" **Recommendation {idx}:**\n")
        print(f" Title: {title}")
        print(f" Genres: {genres}")
        print("-" * 60)
        print("\n")

#  Exemple of consult for a specific movie
movie_title_input = input("\n🎬 Enter the movie title for recommendations: ")
recommend_similar_movies(movie_title_input)

# Exemple of consult for a specific user
user_id_input = input("\n👤 Enter the user ID for recommendations: ")
recommend_movies(user_id_input)
