import itertools

import numpy as np
import torch

SLATE_METRIC_KEYS = ['pk', 'rk', 'f1k', 'cos_ild',]
AVG_METRIC_KEYS = [f'avg_{metric}' for metric in SLATE_METRIC_KEYS]
VAR_METRIC_KEYS = [f'var_{metric}' for metric in SLATE_METRIC_KEYS]
STD_ERR_METRIC_KEYS = [f'std_err_{metric}' for metric in SLATE_METRIC_KEYS]
METRIC_KEYS = AVG_METRIC_KEYS + VAR_METRIC_KEYS + STD_ERR_METRIC_KEYS + ['cc']


def precision_recall_f1_at_k(slate, target, target_length):
    num_hits = float(len(np.intersect1d(slate, target, assume_unique=False)))

    pk = num_hits / len(slate)
    rk = 0
    if target_length > 0:
        rk = num_hits / target_length

    denominator = pk + rk
    if denominator == 0:
        f1k = 0
    else:
        f1k = 2 * (pk * rk) / denominator

    return pk, rk, f1k


def avg_pairwise_diversity_from_matrix(slates: torch.Tensor, item_item_scores: torch.FloatTensor, truncate_last=False) -> torch.FloatTensor:
    batch_size = slates.shape[0]
    K = slates.shape[1]

    pairs = torch.Tensor(list(itertools.permutations(torch.arange(0, K), 2))).long()
    slate_item_rows, slate_item_cols = slates[:, pairs[:, 0]], \
                                       slates[:, pairs[:, 1]]
    scores = item_item_scores[slate_item_rows, slate_item_cols]
    scores_res = torch.reshape(scores, (batch_size, K, K-1))
    mean_scores = scores_res.mean(dim=-1)
    if truncate_last:
        mean_scores = mean_scores[:, :-1]

    return mean_scores


def avg_emb_pairwise_diversity(emb_slates: torch.Tensor, diversity_type) -> torch.FloatTensor:
    batch_size = emb_slates.shape[0]
    K = emb_slates.shape[1]
    pairs = torch.Tensor(list(itertools.permutations(torch.arange(0, emb_slates.shape[1]), 2))).long()
    m1, m2 = emb_slates[:, pairs[:, 0], :], emb_slates[:, pairs[:, 1], :]
    if diversity_type == 'cosine':
        norm_cossim = (1 / 2) * (1 + torch.cosine_similarity(m1, m2, dim=-1))  # cossim between 0, 1
        distance = 1 - norm_cossim

    elif diversity_type == 'euclidean':
        distance = torch.linalg.norm((m1 - m2), dim=-1)

    distance = torch.reshape(distance, (batch_size, K, K-1))
    distance = distance.mean(dim=-1)  # mean diversity of item with all other items (due to reshape)

    return distance


def avg_pairwise_diversity(slates: torch.Tensor, item_embeddings, diversity_type):
    emb_slates = item_embeddings(slates)
    return avg_emb_pairwise_diversity(emb_slates, diversity_type)


def intra_list_diversity_from_matrix(slates, item_item_scores: torch.FloatTensor):
    diversities = avg_pairwise_diversity_from_matrix(slates, item_item_scores)
    diversities = diversities.mean(dim=-1, keepdim=True)    
    return diversities


def item_coverage(slates, num_items):
    # item coverage
    unique_items = set(np.array(slates.flatten()))
    ic = len(unique_items) / num_items

    return ic


def generate_catalog_metrics(slates, num_items):
    ic = item_coverage(slates, num_items)
    return torch.FloatTensor([ic])


def generate_slate_metrics(slates, targets, targets_lengths, item_item_scores) -> torch.FloatTensor:
    ild = intra_list_diversity_from_matrix(slates, item_item_scores).cpu()
    slates = slates.cpu()
    targets = targets.cpu()
    targets_lengths = targets_lengths.cpu()
    metrics = torch.FloatTensor([precision_recall_f1_at_k(*metric_input)
                                 for metric_input in zip(slates, targets, targets_lengths)])
    metrics = torch.cat([metrics, ild], dim=1)
    return metrics
