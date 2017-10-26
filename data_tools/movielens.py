import pandas as pd


def get_ratings_df(path):
    df = pd.read_csv(path)
    # indexing from zero
    df.userId -= 1
    df.movieId -= 1
    return df


def get_movies_df(path):
    df = pd.read_csv(path)
    # indexing from zero
    df.movieId -= 1
    return df


def get_tags_df(path):
    df = pd.read_csv(path)
    # indexing from zero
    df.movieId -= 1
    return df
