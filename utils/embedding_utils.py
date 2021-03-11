import os

import numpy as np
import torch
from torch import nn

from utils.load_write_utils import load_sparse_from_npz, load_arr_from_npz
from utils.path_utils import get_processed_path


def get_training_embeddings(args, num_items):
    if args.use_pretrained_embeddings:
        training_embeddings = get_item_embeddings(args.dataset, args.variation, num_items, args.item_embedding_size,
                                                  args.pretrained_embeddings_name, args.finetune_pretrained_embeddings)
    else:
        training_embeddings = nn.Embedding(num_items + 1, args.item_embedding_size, padding_idx=num_items)
    return training_embeddings


def get_item_embeddings(dataset, variation, num_items, embedding_size, embedding_type, finetune=False, sparse=False) -> nn.Embedding:
    load_processed_path = get_processed_path(dataset, variation)
    load_embeddings_path = os.path.join(load_processed_path, 'item_embeddings')
    embeddings_filename = f'{embedding_type}_embeddings'
    if sparse:
        # load sparse embeddings
        embeddings = load_sparse_from_npz(load_embeddings_path, embeddings_filename)
        embeddings = embeddings.tocoo()
        embeddings = torch.sparse.FloatTensor(
            torch.LongTensor([embeddings.row.tolist(), embeddings.col.tolist()]),
            torch.FloatTensor(embeddings.data.astype(np.int32))
        )
        sparse_flag = True
    else:
        embeddings = load_arr_from_npz(load_embeddings_path, embeddings_filename)
        embeddings = torch.from_numpy(embeddings).float()
        sparse_flag = False

    # handle rotation
    if embeddings.shape[1] in (num_items, num_items + 1):
        embeddings = embeddings.t()

    # truncate to given embedding size
    if embedding_size > -1:
        embeddings = embeddings[:, :embedding_size]

    # handle missing padding
    if num_items == embeddings.shape[0]:
        new_row = torch.zeros(1, embeddings.shape[1])
        if sparse_flag:
            indices = torch.nonzero(new_row).t()
            values = new_row[indices[0], indices[1]]  # modify this based on dimensionality
            embeddings = torch.cat([embeddings, torch.sparse.FloatTensor(indices, values, new_row.size())])
        else:
            embeddings = torch.cat([embeddings, new_row])

    embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=num_items, freeze=not finetune, sparse=sparse_flag)

    return embeddings