"""
generate_data.py
Generates a realistic synthetic movie-ratings dataset (MovieLens-style),
including tags and a short overview per movie so the content and semantic
models have real text to work with (not just a genre string).

Runs fully offline — no external downloads. To use a REAL dataset (e.g.
MovieLens 25M) instead, replace data/movies.csv and data/ratings.csv with
files matching the same schema; no other code needs to change. This
project does NOT auto-download external datasets.

Produces:
  data/movies.csv   -> movieId, title, genres, tags, overview
  data/ratings.csv  -> userId, movieId, rating, timestamp
"""
import numpy as np
import pandas as pd
import os
import time

SEED = 42
np.random.seed(SEED)

os.makedirs("data", exist_ok=True)

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance",
          "Horror", "Thriller", "Animation", "Documentary", "Fantasy"]

THEME_WORDS = {
    "Action": ["explosive", "chase", "heist", "combat", "rescue"],
    "Comedy": ["hilarious", "awkward", "misadventure", "satire", "quirky"],
    "Drama": ["heartbreaking", "family", "struggle", "redemption", "loss"],
    "Sci-Fi": ["space", "future", "alien", "robot", "dystopia"],
    "Romance": ["love", "heartfelt", "reunion", "longing", "wedding"],
    "Horror": ["haunted", "terrifying", "supernatural", "survival", "curse"],
    "Thriller": ["suspenseful", "conspiracy", "chase", "mystery", "betrayal"],
    "Animation": ["colorful", "whimsical", "family-friendly", "adventure", "magical"],
    "Documentary": ["true story", "investigative", "historical", "interviews", "archival"],
    "Fantasy": ["mythical", "kingdom", "magic", "quest", "dragon"],
}

N_MOVIES = 500
N_USERS = 300
N_LATENT = 8

movie_titles = [f"Movie_{i:03d}" for i in range(N_MOVIES)]
movie_genres_list, movie_tags_list, movie_overview_list = [], [], []

for _ in range(N_MOVIES):
    k = np.random.randint(1, 3)
    genres = list(np.random.choice(GENRES, size=k, replace=False))
    movie_genres_list.append("|".join(genres))

    theme_pool = []
    for g in genres:
        theme_pool.extend(THEME_WORDS[g])
    tags = list(np.random.choice(theme_pool, size=min(4, len(theme_pool)), replace=False))
    movie_tags_list.append(", ".join(tags))

    overview_words = list(np.random.choice(theme_pool, size=min(6, len(theme_pool)), replace=False))
    overview = f"A {genres[0].lower()} story about " + " and ".join(overview_words[:3]) + \
               f", exploring themes of {overview_words[-1]}."
    movie_overview_list.append(overview)

movie_latent = np.random.normal(0, 1, size=(N_MOVIES, N_LATENT))

movies_df = pd.DataFrame({
    "movieId": range(1, N_MOVIES + 1),
    "title": movie_titles,
    "genres": movie_genres_list,
    "tags": movie_tags_list,
    "overview": movie_overview_list,
})
movies_df.to_csv("data/movies.csv", index=False)

user_latent = np.random.normal(0, 1, size=(N_USERS, N_LATENT))

rows = []
base_time = int(time.time()) - 60 * 60 * 24 * 365
for u in range(N_USERS):
    n_ratings = np.random.randint(15, 60)
    rated_movies = np.random.choice(N_MOVIES, size=n_ratings, replace=False)
    ts_offsets = np.sort(np.random.randint(0, 60 * 60 * 24 * 365, size=n_ratings))
    for m, ts_off in zip(rated_movies, ts_offsets):
        score = np.dot(user_latent[u], movie_latent[m]) / np.sqrt(N_LATENT)
        noise = np.random.normal(0, 0.4)
        rating = 3 + score + noise
        rating = np.clip(round(rating * 2) / 2, 0.5, 5.0)
        rows.append((u + 1, m + 1, rating, base_time + int(ts_off)))

ratings_df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
ratings_df.to_csv("data/ratings.csv", index=False)

print(f"Generated {len(movies_df)} movies and {len(ratings_df)} ratings for {N_USERS} users.")
print(movies_df.head(3))
print(ratings_df.head(3))
