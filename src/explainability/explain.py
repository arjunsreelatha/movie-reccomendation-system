"""
src/explainability/explain.py
Generates explanations strictly from actual per-component scores and
real user history — never a templated guess disconnected from the model.

Two things this version fixes over a naive "cite the user's top-2 liked
movies" approach:
  1. The cited titles are chosen per-recommendation (genre-overlap first,
     then a rotation fallback) instead of always being the same fixed two —
     otherwise every card ends up citing identical evidence regardless of
     which movie is being explained.
  2. The runner-up signal is mentioned when it's close to the dominant one,
     so explanations reflect that recommendations are usually a blend, not
     a single winner-take-all model.
"""


def _genre_set(genres_str):
    return set(g.strip() for g in genres_str.split("|")) if genres_str else set()


def _pick_supporting_titles(movie_id, target_genres, liked_df, movies_df, k=2):
    """
    Choose which of the user's liked movies to cite as evidence for THIS
    specific recommendation. Prefers genre overlap with the target movie;
    falls back to a deterministic rotation (keyed on movie_id) through the
    user's liked list so different recommended movies cite different
    evidence, instead of the same two titles every time.
    """
    liked_movies = movies_df[movies_df.movieId.isin(liked_df.movieId)].copy()
    if liked_movies.empty:
        return []

    liked_movies["overlap"] = liked_movies.genres.apply(
        lambda g: len(target_genres & _genre_set(g))
    )
    liked_movies = liked_movies.merge(
        liked_df[["movieId", "rating"]], on="movieId", how="left"
    )
    liked_movies = liked_movies.sort_values(["overlap", "rating"], ascending=False)

    top_overlap = liked_movies[liked_movies.overlap > 0]
    if len(top_overlap) >= k:
        return top_overlap.title.tolist()[:k]

    # Not enough genre-overlapping evidence — rotate through the full liked
    # list based on movie_id so different target movies get different
    # citations instead of all collapsing to the same top-2.
    titles = liked_movies.title.tolist()
    if not titles:
        return []
    start = movie_id % len(titles)
    rotated = titles[start:] + titles[:start]
    return rotated[:k]


def explain_recommendation(user_id, movie_id, movies_df, ratings_df, component_scores: dict,
                            component_signals: dict, top_genre_overlap=None):
    """
    component_scores: {model_name: score_for_this_item}
    component_signals: {model_name: bool} — whether that model actually had
                        signal for this user (False for cold-start components)
    """
    movie_row = movies_df[movies_df.movieId == movie_id].iloc[0]
    genres = movie_row.genres
    target_genres = _genre_set(genres)

    liked = ratings_df[(ratings_df.userId == user_id) & (ratings_df.rating >= 4.0)]

    # Only consider components that actually had signal for this user
    usable = {k: v for k, v in component_scores.items() if component_signals.get(k, True)}
    if not usable:
        return f"Popular pick with strong overall ratings; matches the {genres} genre."

    ranked = sorted(usable.items(), key=lambda kv: kv[1], reverse=True)
    dominant, dominant_score = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    if liked.empty:
        # Genuine cold-start: no rating history exists to point to, so the
        # explanation must not claim otherwise.
        return f"Recommended based on general popularity and the {genres} genre — " \
               f"we don't have enough of your ratings yet to personalize further."

    reasons = {
        "collaborative": "users with similar rating patterns rated this highly",
        "neural": "deep interaction modeling of your rating history",
        "content": f"it closely matches the {genres} genre profile of movies you rated highly",
        "semantic": "it's semantically similar in theme/description to movies you rated highly",
        "popularity": "it's broadly well-rated across the platform",
    }
    base = reasons.get(dominant, "your overall preference profile")

    sample_titles = _pick_supporting_titles(movie_id, target_genres, liked, movies_df, k=2)
    sample = ", ".join(sample_titles) if sample_titles else "your rating history"

    explanation = f"Recommended because {base} (e.g. you rated {sample} highly)."

    # If the runner-up signal is genuinely close (within ~10%), name it too —
    # this is honest about the fact that most recommendations are a blend,
    # not a single model's clean win.
    if runner_up and dominant_score > 0 and (runner_up[1] / dominant_score) > 0.9:
        runner_up_name, _ = runner_up
        runner_up_reason = {
            "collaborative": "similar users' ratings",
            "neural": "deep interaction patterns",
            "content": "genre similarity",
            "semantic": "thematic similarity",
            "popularity": "overall popularity",
        }.get(runner_up_name)
        if runner_up_reason and runner_up_name != dominant:
            explanation += f" ({runner_up_reason} also contributed.)"

    return explanation