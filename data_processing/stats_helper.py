import argparse
import os

import numpy as np
import pandas as pd

from utils.load_write_utils import load_df_from_csv
from utils.path_utils import get_processed_path


def item_stats_per_user(df: pd.DataFrame):
    user_gby = df.groupby('userId')['itemId'].apply(lambda x: len(list(x)))
    num_item_per_user = user_gby.values
    return np.mean(num_item_per_user), np.var(num_item_per_user)


def print_df_stats(df):
    num_records = len(df)
    num_users, num_items = get_num_users_and_items_from_df(df)
    avg_n_mpu = item_stats_per_user(df)
    min_user_occ = sorted(df.groupby('userId').apply(len).values)[0]
    min_item_occ = sorted(df.groupby('itemId').apply(len).values)[0]

    print(f'num records: {num_records}')
    print(f'user IDs: {num_users }')
    print(f'item IDs: {num_items}')
    print(f'min user occurrences: {min_user_occ}')
    print(f'min item occurrences: {min_item_occ}')
    print(f'avg item/user: {avg_n_mpu}')
    print('')


def get_num_users_and_items_from_df(df):
    num_users = len(df['userId'].unique())
    num_items = len(df['itemId'].unique())

    return num_users, num_items


def print_stats(train_df, val_df, test_df):
    print('Full dataset')
    full_df = pd.concat([train_df, val_df, test_df])
    print_df_stats(full_df)
    print('Train')
    print_df_stats(train_df)
    print('Val')
    print_df_stats(val_df)
    print('Test')
    print_df_stats(test_df)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--variation', type=str)
    parser.add_argument('--switch', type=str, choices={'original', 'processed'})

    args = parser.parse_args()
    dataset = args.dataset
    variation = args.variation
    switch = args.switch # 'processed' or 'original'
    if switch == 'processed':
        load_path = get_processed_path(dataset, variation)

        train_df = load_df_from_csv(load_path, 'train.csv')
        val_df = load_df_from_csv(load_path, 'val.csv')
        test_df = load_df_from_csv(load_path, 'test.csv')

        print_stats(train_df, val_df, test_df)
    if switch == 'original':
        load_path = os.path.join('data', dataset, variation, 'original')
        ratings_df = load_df_from_csv(load_path, 'ratings.csv')
        items_df = load_df_from_csv(load_path, 'items.csv')
        full_df = pd.merge(ratings_df, items_df, on='itemId')

        print_df_stats(full_df)


if __name__ == "__main__":
    main()