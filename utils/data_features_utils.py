import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.load_write_utils import load_arr_from_npz, load_df_from_csv
from utils.path_utils import get_processed_path


def get_item_item_diversities(dataset, variation, task, num_items) -> torch.FloatTensor:
    proc_path = get_processed_path(dataset, variation)
    load_path = os.path.join(proc_path, task)
    similarities = load_arr_from_npz(load_path, 'item_diversities.npz')

    assert similarities.shape == (num_items, num_items) or similarities.shape == (num_items + 1, num_items + 1), f'num_items: {num_items}, shape: {similarities.shape}'

    return torch.from_numpy(similarities).float()


def get_histories(dataset, variation, set_name: str):
    load_processed_path = get_processed_path(dataset, variation)
    load_set_path = os.path.join(load_processed_path, set_name)

    histories = load_arr_from_npz(load_set_path, f'histories')

    # ensure unique
    new_histories = []
    for i, history in enumerate(histories):
        idx = list(sorted(np.unique(history, return_index=True)[1]))
        new_histories.append(history[idx])
    return new_histories


def get_targets(dataset, variation, set_name: str, truncate_targets, target_size):
    load_processed_path = get_processed_path(dataset, variation)
    load_set_path = os.path.join(load_processed_path, set_name)
    targets = load_arr_from_npz(load_set_path, f'targets')

    if truncate_targets:
        new_targets = []
        for target in targets:
            new_targets.append(target[:target_size])
        new_targets = np.array(new_targets)
        assert len(new_targets) == len(targets)

        return new_targets

    return targets


def pad_batch(batch: list, pad_token: int):
    """
    :param batch: a jagged list of tensors (i.e. tensors of different lengths)
    :param pad_token: padding token to add extra entries
    :return:
    """

    lengths = [len(tensor) for tensor in batch]
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=pad_token)
    padded_batch = padded_batch.to(torch.int64)

    return padded_batch, lengths


def pad_tensor_list(lst, pad_token):
    lengths = torch.as_tensor([item.shape[0] for item in lst])
    ## pad
    result = pad_sequence(lst, batch_first=True, padding_value=pad_token)
    ## compute mask
    mask = (result != pad_token)

    return result.long(), lengths, mask


def get_num_users_items(dataset, variation):
    proc_path = get_processed_path(dataset, variation)
    df = load_df_from_csv(proc_path, 'num_users_items')
    stats_dict = df.to_dict()

    return stats_dict['num_users'][0], stats_dict['num_items'][0]