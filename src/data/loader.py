"""
src/data/loader.py
Loads ratings/movies and performs a CHRONOLOGICAL split (not random).

Why chronological, not random:
Recommender systems are deployed to predict future behavior from past
behavior. A random split lets ratings a user made "later" leak into the
training set while "earlier" ratings of the same user sit in test — a
pattern the model would never see live. That inflates offline metrics vs.
what you'd actually get in production. A chronological, per-user split
(earliest interactions -> train, most recent -> test) mirrors real
deployment and is the standard practice for offline RecSys evaluation.
"""
import pandas as pd
from typing import Tuple


def load_data(movies_path: str = "data/movies.csv",
              ratings_path: str = "data/ratings.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def chronological_split(ratings: pd.DataFrame, test_fraction: float = 0.2,
                         min_train_per_user: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user chronological split: for each user, sort their ratings by
    timestamp, keep the earliest (1 - test_fraction) as train and the most
    recent test_fraction as test. Users with too few ratings to split
    meaningfully go entirely into train (so they still contribute to CF/NCF
    fitting) and are naturally absent from test.
    """
    train_rows, test_rows = [], []
    for user_id, group in ratings.groupby("userId"):
        group = group.sort_values("timestamp")
        n = len(group)
        n_test = int(round(n * test_fraction))
        n_train = n - n_test
        if n_train < min_train_per_user or n_test == 0:
            train_rows.append(group)
            continue
        train_rows.append(group.iloc[:n_train])
        test_rows.append(group.iloc[n_train:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else \
        ratings.iloc[0:0].copy()
    return train_df, test_df
