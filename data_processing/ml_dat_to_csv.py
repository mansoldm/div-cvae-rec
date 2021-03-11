import argparse
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from utils.path_utils import get_orig_path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml')
parser.add_argument('--variation', type=str, default='10m')

args = parser.parse_args()
orig_path = get_orig_path(args.dataset, args.variation)

movies_cols = ['itemId', 'title', 'genres']
ratings_cols = ['userId', 'itemId', 'rating', 'timestamp']


def read_file(path):
    return open(path, encoding='ISO-8859-1')


def main():
    # Movies CSV
    print('Processing movies...')
    movies_dict = defaultdict(list)
    movie_file = read_file(os.path.join(orig_path, 'movies.dat'))
    for line in tqdm(movie_file):
        line = line.strip().split('::')

        # line is movieId, title, genres
        for key, value in zip(movies_cols, line):
            movies_dict[key].append(value)

    movies_df = pd.DataFrame.from_dict(movies_dict)
    movies_df.to_csv(os.path.join(orig_path, 'items.csv'), index=False)

    # Ratings CSV
    print('Processing ratings...')
    ratings_dict = defaultdict(list)
    rating_file = read_file(os.path.join(orig_path, 'ratings.dat'))
    for line in tqdm(rating_file):
        line = line.strip().split('::')

        for key, value in zip(ratings_cols, line):
            ratings_dict[key].append(value)

    ratings_df = pd.DataFrame.from_dict(ratings_dict)
    ratings_df.to_csv(os.path.join(orig_path, 'ratings.csv'), index=False)


if __name__ == '__main__':
    main()