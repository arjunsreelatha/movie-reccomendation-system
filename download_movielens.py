"""Download and convert the official MovieLens 1M dataset."""
import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def convert_dataset(movie_data: bytes, rating_data: bytes):
    movies = pd.read_csv(
        BytesIO(movie_data),
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )
    movies["tags"] = ""
    movies["overview"] = ""

    ratings = pd.read_csv(
        BytesIO(rating_data),
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    return movies, ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "movielens-1m",
        help="Directory for converted CSV files (default: data/movielens-1m)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite converted CSV files if they already exist",
    )
    args = parser.parse_args()

    movies_path = args.output_dir / "movies.csv"
    ratings_path = args.output_dir / "ratings.csv"
    if not args.force and (movies_path.exists() or ratings_path.exists()):
        parser.error(
            f"output already exists in {args.output_dir}; use --force to overwrite"
        )

    with urlopen(DATASET_URL, timeout=60) as response:
        archive_data = response.read()

    with ZipFile(BytesIO(archive_data)) as archive:
        movies, ratings = convert_dataset(
            archive.read("ml-1m/movies.dat"),
            archive.read("ml-1m/ratings.dat"),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    movies.to_csv(movies_path, index=False)
    ratings.to_csv(ratings_path, index=False)
    print(
        f"Converted {len(movies):,} movies and {len(ratings):,} ratings "
        f"from {DATASET_URL} into {args.output_dir}."
    )


if __name__ == "__main__":
    main()
