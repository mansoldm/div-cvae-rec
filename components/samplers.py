import torch
from torch import nn


class SlateSampler(nn.Module):
    def __init__(self, slate_size, *args):
        super(SlateSampler, self).__init__()
        self.slate_size = slate_size

    def forward(self, batch_k_head_softmax):
        raise NotImplementedError


class RandomSlateSampler(SlateSampler):
    # randomly samples from each distribution head
    def forward(self, batch_k_head_softmax):
        batch_size = batch_k_head_softmax.shape[0]
        batch_samples = []
        for i in range(batch_size):
            samples = torch.multinomial(batch_k_head_softmax[i], num_samples=1)
            batch_samples.append(samples)

        # stack to get (batch_size, slate_size, 1) tensor, squeeze last dimension
        batch_samples = torch.stack(batch_samples).squeeze(-1)

        return batch_samples


class ArgmaxSlateSampler(SlateSampler):
    # get highest probability item from each distribution
    def forward(self, batch_k_head_softmax):
        batch_samples = torch.argmax(batch_k_head_softmax, dim=2)
        return batch_samples


class ArgmaxDeduplicateSlateSampler(SlateSampler):
    # get highest probability item from each distribution
    # items sampled at 1..i cannot be sampled at i+1
    def forward(self, batch_k_head_softmax):
        device = next(self.parameters()).device
        batch_size = batch_k_head_softmax.shape[0]
        batch_samples = []
        for i in range(batch_size):
            slate_dists = batch_k_head_softmax[i, :, :]
            slate = []
            for j in range(self.slate_size):
                slate_dist = slate_dists[j]
                slate_dist = slate_dist.scatter_(dim=-1, index=torch.LongTensor(slate).to(device),
                                                 src=torch.FloatTensor(0))
                item = torch.argmax(slate_dist)
                slate.append(item)

            slate = torch.tensor(slate)
            batch_samples.append(torch.tensor(slate))

        batch_samples = torch.stack(batch_samples).to(device)
        return batch_samples


class TopKDeduplicateSlateSampler(SlateSampler):
    # treat each distribution independently
    # i.e. order of items in slate doesn't matter
    # sum log distributions to obtain a single distributions
    # sample top k items
    def forward(self, batch_k_head_softmax):
        device = batch_k_head_softmax.device
        eps = torch.tensor(1e-7).to(device)  # ensure that log arg is never zero
        log_sm = torch.log(batch_k_head_softmax + eps)
        scores = torch.sum(log_sm, dim=1)
        # batch_samples are the resulting 'argmax indices' of topk operation, return
        _, batch_samples = torch.topk(scores, k=self.slate_size)
        return batch_samples


def get_slate_sampler(sample_type, slate_size) -> SlateSampler:

    if sample_type == 'random':
        return RandomSlateSampler(slate_size)

    elif sample_type == 'argmax':
        return ArgmaxSlateSampler(slate_size)

    elif sample_type == 'argmax_deduplicate':
        return ArgmaxDeduplicateSlateSampler(slate_size)

    elif sample_type == 'topk_deduplicate':
        return TopKDeduplicateSlateSampler(slate_size)

    raise NotImplementedError(f'Sampler type \'{sample_type}\' not implemented!')