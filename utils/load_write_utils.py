import csv
import os

import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse


def load_df_from_csv(path, filename_csv):
    if not filename_csv.endswith('.csv'):
        filename_csv += '.csv'

    csv_path = os.path.join(path, filename_csv)
    df = pd.read_csv(csv_path, index_col=False)
    return df


def load_arr_from_npz(path: str, filename_npz: str):
    if not filename_npz.endswith('.npz'):
        filename_npz += '.npz'

    fpath = os.path.join(path, filename_npz)
    npload = np.load(fpath, allow_pickle=True)
    return npload['arr_0']


def write_arr_to_npz(path: str, filename_npz: str, np_arr, ignore_if_existing=False):
    if not filename_npz.endswith('.npz'):
        filename_npz += '.npz'
    if not os.path.exists(path):
        os.mkdir(path)

    output_path = os.path.join(path, filename_npz)
    if ignore_if_existing and os.path.exists(output_path):
        return

    np.savez(output_path, np_arr)


def write_to_csv(path: str, filename_csv: str, df: pd.DataFrame):
    if not filename_csv.endswith('.csv'):
        filename_csv += '.csv'
    if not os.path.exists(path):
        os.mkdir(path)

    output_path = os.path.join(path, filename_csv)
    df.to_csv(output_path, header=True, index=False)


def load_sparse_from_npz(path: str, filename_npz: str) -> sp.sparse.csr_matrix:
    if not filename_npz.endswith('.npz'):
        filename_npz += '.npz'

    fpath = os.path.join(path, filename_npz)
    spload = sp.sparse.load_npz(fpath)
    return spload


def write_sparse_to_npz(path: str, filename_npz: str, sparse_mat):
    if not filename_npz.endswith('.npz'):
        filename_npz += '.npz'
    if not os.path.exists(path):
        os.mkdir(path)

    output_path = os.path.join(path, filename_npz)
    sp.sparse.save_npz(output_path, sparse_mat)


def write_dict_to_csv_file(f, file_exists, to_write, verbose=True):
    writer = csv.DictWriter(f, fieldnames=list(to_write.keys()))
    if not file_exists:
        writer.writeheader()
    if verbose:
        print(to_write)
    writer.writerow(to_write)


def write_dict_to_csv(path, filename_csv, to_write, verbose=True, mode='a+'):
    if not filename_csv.endswith('.csv'):
        filename_csv += '.csv'

    if not os.path.exists(path):
        os.makedirs(path)
    respath_str = os.path.join(path, filename_csv)
    file_exists = os.path.exists(respath_str)
    with open(respath_str, mode) as f:
        write_dict_to_csv_file(f, file_exists, to_write, verbose)


def load_dataframes(load_processed_path):
    train_df = load_df_from_csv(load_processed_path, 'train.csv')
    val_df = load_df_from_csv(load_processed_path, 'val.csv')
    test_df = load_df_from_csv(load_processed_path, 'test.csv')
    full_df = load_df_from_csv(load_processed_path, 'full.csv')
    return train_df, val_df, test_df, full_df
