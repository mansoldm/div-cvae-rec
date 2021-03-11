import argparse
import os

import numpy as np

from utils.load_write_utils import load_sparse_from_npz, write_arr_to_npz
from utils.path_utils import get_processed_path


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--variation', type=str)
parser.add_argument('--name', type=str, help='Train or test. The name of the sparse matrix to use')
parser.add_argument('--task', type=str, help='Specify the task for which the similarities are saved i.e. val or test')
args = parser.parse_args()


def cos_div(A):
    similarity = A.dot(A.T).toarray()

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag

    return 1 - cosine


def main():
    processed_path = get_processed_path(args.dataset, args.variation)
    path = os.path.join(processed_path, 'sparse')
    print('Loading interactions...')
    if args.name == 'test':
        train_matrix = load_sparse_from_npz(path, 'train')
        val_matrix = load_sparse_from_npz(path, 'val')
        user_item_matrix = train_matrix._add_sparse(val_matrix)
    else:
        user_item_matrix = load_sparse_from_npz(path, args.name)

    print('Computing pointwise diversities...')
    pointwise_diversities = cos_div(user_item_matrix.transpose())
    num_items = pointwise_diversities.shape[0]
    zeros = np.zeros((num_items+1, num_items+1))
    zeros[:num_items, :num_items] = pointwise_diversities
    zeros[num_items, num_items] = 1
    pointwise_diversities = zeros
    print(type(pointwise_diversities))
    print(pointwise_diversities.shape)
    save_path = os.path.join(processed_path, args.task)
    print(f'Saving pointwise diversities to {os.path.join(save_path, "item_diversities.npz")}')
    write_arr_to_npz(save_path, 'item_diversities.npz', pointwise_diversities,)
    print('Success.')


if __name__ == '__main__':
    main()
    