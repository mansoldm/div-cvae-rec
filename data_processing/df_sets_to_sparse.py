import argparse
import os

from data_processing.process_data import write_sparse_matrix
from utils.load_write_utils import load_dataframes
from utils.path_utils import get_processed_path

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--variation', type=str)

    args = parser.parse_args()
    proc_path = get_processed_path(args.dataset, args.variation)

    train_df, val_df, test_df, _ = load_dataframes(proc_path)
    sparse_path = os.path.join(proc_path, 'sparse')
    # make and save sparse matrices
    sparse_train = write_sparse_matrix(train_df, sparse_path, 'train')
    shape = sparse_train.shape
    write_sparse_matrix(val_df, sparse_path, 'val')
    write_sparse_matrix(test_df, sparse_path, 'test')

