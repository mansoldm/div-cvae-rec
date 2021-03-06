import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_features_utils import pad_tensor_list, get_histories, get_targets


class ValidationTestDataset(Dataset):
    def __init__(self, user_full_histories, targets, cond_diversity):
        self.cond_diversity = cond_diversity

        histories = [torch.as_tensor(hist) for hist, target in zip(user_full_histories, targets) if len(target) > 0]
        self.full_histories = histories
        targets = [torch.as_tensor(target) for target in targets if len(target) > 0]
        self.targets = targets
        self.len_data = len(self.full_histories)

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        cond_diversity = self.cond_diversity

        history = self.full_histories[idx]
        target = self.targets[idx]

        return cond_diversity, history, target


def validation_test_collate_fn_pad(batch, num_items):
    cond_diversity = torch.stack([torch.as_tensor(example[0]) for example in batch])

    history = [torch.as_tensor(example[1]) for example in batch]
    history, history_lengths, history_mask = pad_tensor_list(history, pad_token=num_items)

    targets = [torch.as_tensor(example[2]) for example in batch]
    targets, targets_lengths, _ = pad_tensor_list(targets, num_items)

    return cond_diversity, history, history_lengths, history_mask, targets, targets_lengths


def get_validation_test_collate_fn_pad(num_items: int):
    return lambda batch: validation_test_collate_fn_pad(batch, num_items=num_items)


def get_validation_test_dataloader(num_items, set_name, dataset, variation, truncate_targets, target_size, batch_size,
                                   slate_size=None, cond_diversity=None):
    if cond_diversity is None:
        cond_diversity = torch.FloatTensor([0.7] * slate_size)

    histories = get_histories(dataset, variation, set_name)
    targets = get_targets(dataset, variation, set_name, truncate_targets, target_size)
    dataset = ValidationTestDataset(histories, targets, cond_diversity)

    loader = DataLoader(dataset, batch_size,
                        shuffle=False,
                        collate_fn=get_validation_test_collate_fn_pad(num_items)
                        )

    return loader