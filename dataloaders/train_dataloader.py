import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_features_utils import pad_tensor_list, get_histories


class RandomDataset(Dataset):
    def __init__(self, num_users, histories_arr, slates_arr):
        print('Setting up Random dataset...')
        self.history = histories_arr
        self.slates = slates_arr

        # set total number of datapoints
        self.len_data = min(len(self.history), num_users)

        # randomly shuffle entire dataset in the same way
        randomize = np.arange(self.len_data)
        np.random.shuffle(randomize)

        self.history = self.history[randomize]
        self.slates = self.slates[randomize]

        history = [torch.as_tensor(hist) for hist in self.history]
        self.history = history

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        history = self.history[idx]
        slate = torch.LongTensor(self.slates[idx])

        return slate, history


def training_collate_fn_pad(batch, shuffle_slates: bool, num_items):
    slates = [example[0] for example in batch]
    if shuffle_slates:
        for i, slate in enumerate(slates):
            slates[i] = slate[torch.randperm(len(slate))]

    slates = torch.stack(slates).type(torch.LongTensor)
    history = [torch.as_tensor(example[1]) for example in batch]
    history, history_lengths, history_mask = pad_tensor_list(history, pad_token=num_items)

    return slates, history, history_lengths, history_mask


def get_training_collate_fn_pad(shuffle_slates, num_items):
    return lambda batch: training_collate_fn_pad(batch, shuffle_slates, num_items)


def get_training_dataloader(dataset, variation, batch_size, slate_size, shuffle_slates, num_users, num_items, set_name):
    full_histories = get_histories(dataset, variation, set_name)
    history_arr, slates_arr = get_histories_slates(training_interactions=full_histories, slate_size=slate_size)
    dataset = RandomDataset(num_users, history_arr, slates_arr)

    loader = DataLoader(dataset, batch_size,
                        shuffle=True,
                        collate_fn=get_training_collate_fn_pad(shuffle_slates, num_items)
                        )

    return loader


def get_user_history_slate(user_history, slate_size):
    truncated_history = user_history[:-slate_size].copy()
    slate = user_history[-slate_size:].copy()

    return truncated_history, slate


def get_histories_slates(training_interactions, slate_size):
    history_arr = []
    slates_arr = []
    for interaction in training_interactions:
        history, slate = get_user_history_slate(interaction, slate_size)
        if len(history) > 0 and len(slate) == slate_size:
            history_arr.append(history)
            slates_arr.append(slate)

    return np.array(history_arr), np.array(slates_arr)
