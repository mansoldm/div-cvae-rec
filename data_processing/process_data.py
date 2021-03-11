import argparse
import os

import scipy as sp
import scipy.sparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.load_write_utils import write_to_csv, write_sparse_to_npz, load_df_from_csv, write_dict_to_csv
from utils.path_utils import get_processed_path, get_dataset_variation_data_path
from data_processing.stats_helper import get_num_users_and_items_from_df


def train_test_split(df: pd.DataFrame, split: float):
    ''' df must be sorted by timestamps ascendingly '''
    train_items = round(len(df) * (1 - split))
    train_df = df.iloc[:train_items]
    test_df = df.iloc[train_items:]

    return train_df, test_df


def train_val_test_split_by_user(merged_df, test_split, val_split):
    #  train/val/test split - for each user_id, perform timestamped train_test_split
    # user_ids = merged_df['userId'].unique().tolist()
    user_gby = merged_df.groupby(['userId'])
    train_lst, val_lst, test_lst = [], [], []
    for user_id, entries in tqdm(user_gby):
        # perform split
        trainval, test = train_test_split(entries, test_split)
        train, val = train_test_split(trainval, val_split)

        train_lst.append(train)
        val_lst.append(val)
        test_lst.append(test)

    train_df = pd.concat(train_lst)
    val_df = pd.concat(val_lst)
    test_df = pd.concat(test_lst)

    return train_df, val_df, test_df


def df_to_user_item_matrix(df: pd.DataFrame, shape=None):
    users, items = df['userId'].to_numpy(), df['itemId'].to_numpy()
    entries = np.ones(len(df))
    if shape:
        csr_mat = sp.sparse.csr_matrix((entries, (users, items)), shape=shape)
    else:
        csr_mat = sp.sparse.csr_matrix((entries, (users, items)))

    return csr_mat


def to_binary(df: pd.DataFrame, threshold=0):
    # make 'ratings' binary
    df['rating'] = (df['rating'].astype(float) >= threshold).astype(int)
    return df


def rm_column_threshold(df: pd.DataFrame, col_name: str, threshold: int):
    return df.groupby(col_name).filter(lambda x: len(x) >= threshold)


def process_data(path: str, val_split: float, test_split: float, min_items: int, min_users: int):
    '''
    This method processes the CSVs and generates a train/val/test split
    path: the path where the CSVs are located
    :param min_items:
    :param min_users:
    '''

    # load dataframes
    load_path = os.path.join(path, 'original')
    print('Loading dataframes')
    items_df = load_df_from_csv(load_path, f'items.csv')
    ratings_df = load_df_from_csv(load_path, 'ratings.csv')

    print(items_df.head())
    print(items_df.shape)
    print(ratings_df.head())
    print(ratings_df.shape)

    # merge userids, ratings, items, etc in unique dataframe
    # Pandas merge() == inner join
    merged_df = pd.merge(ratings_df, items_df, on='itemId')
    print(merged_df.head())
    print(merged_df.shape)

    # remove users and items which don't appear very often
    print('Removing rare users and items')
    merged_df = rm_column_threshold(merged_df, 'itemId', min_items)
    merged_df = rm_column_threshold(merged_df, 'userId', min_users)
    print(merged_df.shape)

    # make userids and itemids categorical, numerical, contiguous
    # handle numeric fields
    print('Make categorical')
    merged_df['userId'] = pd.factorize(merged_df['userId'])[0]
    merged_df['itemId'] = pd.factorize(merged_df['itemId'])[0]
    merged_df['timestamp'] = pd.to_numeric(merged_df['timestamp'])
    merged_df['rating'] = pd.to_numeric(merged_df['rating'])

    # sort values by timestamp, get split dataframes and resulting sizes
    print('Sort by timestamp and split')
    merged_df.sort_values(by=['timestamp'], inplace=True,  ascending=[True])
    split = train_val_test_split_by_user(merged_df, test_split, val_split)

    # concat, refactorize itemIds, split back
    print('Ensure contiguous itemIds in training set')
    merged_df = pd.concat(split, axis=0)
    merged_df['itemId'] = pd.factorize(merged_df['itemId'])[0]
    # split back
    train_lim = len(split[0])
    val_lim = train_lim + len(split[1])
    train_df = merged_df[:train_lim]
    val_df = merged_df[train_lim:val_lim]
    test_df = merged_df[val_lim:]

    # delete items from validation/test which don't appear in training set
    print('Delete items from validation/test missing from train')
    val_df = val_df[val_df['itemId'].isin(train_df['itemId'].unique())]
    test_df = test_df[test_df['itemId'].isin(train_df['itemId'].unique())]

    return train_df, val_df, test_df


def write_sparse_matrix(df: pd.DataFrame, path: str, data_type: str):
    '''
    make and save sparse matrices for df
    df: Dataframe with userIds, itemIds from which sparse matrix is created
    path: path to save sparse matrix at
    data_type: i.e. the set name (train, val, test)
    '''
    sparse_csr_mat = df_to_user_item_matrix(df)
    write_sparse_to_npz(path, data_type, sparse_csr_mat)
    return sparse_csr_mat


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--variation', type=str)
    parser.add_argument('--min_users', type=int, default=10)
    parser.add_argument('--min_items', type=int, default=20)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)

    args = parser.parse_args()

    dataset = args.dataset
    variation = args.variation
    val_split = args.val_split
    test_split = args.test_split

    path = get_dataset_variation_data_path(dataset, variation)
    proc_path = get_processed_path(dataset, variation)

    data_sets = process_data(path, val_split, test_split, args.min_items, args.min_users)
    full_df = pd.concat(data_sets)
    train_df, val_df, test_df = data_sets

    # write to CSVs
    write_to_csv(proc_path, 'train', train_df)
    write_to_csv(proc_path, 'val', val_df)
    write_to_csv(proc_path, 'test', test_df)
    write_to_csv(proc_path, 'full', full_df)

    num_users, num_items = get_num_users_and_items_from_df(full_df)
    to_write = {'num_users': num_users, 'num_items': num_items}
    write_dict_to_csv(proc_path, 'num_users_items', to_write=to_write,)

    sparse_path = os.path.join(proc_path, 'sparse')
    # make and save sparse matrices
    sparse_train = write_sparse_matrix(train_df, sparse_path, 'train')
    write_sparse_matrix(val_df, sparse_path, 'val')
    write_sparse_matrix(test_df, sparse_path, 'test')


if __name__ == "__main__":
    main()
