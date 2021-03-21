import torch

from torch import nn
from torch.nn.utils.rnn import PackedSequence

from components.encoders.transformer_encoder_components import TransformerEncoderBlock
from utils.training_evaluation_utils import avg_emb_pairwise_diversity, avg_pairwise_diversity_from_matrix


class EmbeddingEncoder(nn.Module):
    def __init__(self, nargs, item_embeddings, *args):
        """
        :param item_embeddings:
        :param nargs: number of arguments required to call forward()
        :param args:
        """
        super(EmbeddingEncoder, self).__init__()
        self.nargs = nargs
        self.item_embeddings = item_embeddings
        self.item_embedding_size = self.item_embeddings.embedding_dim
        self.input_size = self.item_embedding_size
        self.output_size = self.item_embedding_size

    def forward(self, *args):
        emb_item = self.item_embeddings(args[0])
        return self.encode(emb_item, *args[1:])

    def encode(self, *args):
        raise NotImplementedError


class HistoryEncoder(EmbeddingEncoder):
    def __init__(self, item_embeddings, nargs=3):
        super(HistoryEncoder, self).__init__(nargs, item_embeddings=item_embeddings)
        self.num_items = self.item_embeddings.num_embeddings - 1

    def encode(self, embedded_history, history_lengths, history_mask):
        raise NotImplementedError


class SumHistoryEncoder(HistoryEncoder):
    def encode(self, embedded_history, history_lengths, history_mask):
        return torch.sum(embedded_history, dim=1)


class MeanHistoryEncoder(SumHistoryEncoder):
    def encode(self, embedded_history, history_lengths, history_mask):
        history_sum = super(MeanHistoryEncoder, self).forward(embedded_history, history_lengths, history_mask)
        return history_sum / history_lengths.unsqueeze(1)


class LSTMHistoryEncoder(HistoryEncoder):
    def __init__(self, *args):
        super(LSTMHistoryEncoder, self).__init__(*args)
        self.lstm = nn.LSTM(input_size=self.item_embedding_size,
                            hidden_size=self.item_embedding_size,
                            num_layers=2,
                            batch_first=True,)

    def encode(self, embedded_history, history_lengths, history_mask):
        device = embedded_history.device
        X: PackedSequence = torch.nn.utils.rnn.pack_padded_sequence(embedded_history, history_lengths.cpu(), batch_first=True,
                                                                    enforce_sorted=False)
        X, _ = self.lstm.forward(X)

        packed = X
        lengths = history_lengths
        sum_batch_sizes = torch.cat((
            torch.zeros(2, dtype=torch.int64),
            torch.cumsum(packed.batch_sizes, 0)
        ))
        sorted_lengths = lengths[packed.sorted_indices]
        last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0))
        last_seq_items = packed.data[last_seq_idxs]
        last_seq_items = last_seq_items[packed.unsorted_indices]

        if self.lstm.bidirectional:
            last_seq_items = last_seq_items[:, :self.item_embedding_size] + last_seq_items[:, self.item_embedding_size:]

        return last_seq_items.to(device)


class TransformerHistoryEncoder(HistoryEncoder):
    def __init__(self, *args):
        super(TransformerHistoryEncoder, self).__init__(*args)
        teb = TransformerEncoderBlock(self.num_items, self.item_embedding_size)
        cls = nn.Embedding(1, self.item_embedding_size)
        self.transformer_encoder_block = teb
        self.cls_token = cls

    def encode(self, embedded_history, history_lengths, history_mask):
        batch_size = embedded_history.shape[0]
        cls_history = torch.cat([self.cls_token.weight.unsqueeze(0).repeat(batch_size, 1, 1), embedded_history], dim=1)
        cls_bool_mask = torch.BoolTensor([True]).repeat(batch_size, 1).to(cls_history.device)
        cls_history_mask = torch.cat([cls_bool_mask, history_mask], dim=1)
        all_encoding = self.transformer_encoder_block.forward(cls_history, cls_history_mask)
        history_encoding = all_encoding[0]

        return history_encoding


class SlateDiversityEncoder(EmbeddingEncoder):
    def __init__(self, item_embeddings, slate_size, diversity_type):
        super(SlateDiversityEncoder, self).__init__(nargs=1, item_embeddings=item_embeddings)
        self.diversity_type = diversity_type
        self.input_size = slate_size
        self.output_size = slate_size

    def encode(self, emb_slate, *args):
        slate_diversities = avg_emb_pairwise_diversity(emb_slate, self.diversity_type)
        slate_diversities.requires_grad = False
        return slate_diversities


class SlateDiversityEncoderFromDiversities(nn.Module):
    def __init__(self, slate_size, item_item_similarities):
        super(SlateDiversityEncoderFromDiversities, self).__init__()
        self.input_size = slate_size
        self.output_size = slate_size
        self.item_item_scores = item_item_similarities

    def forward(self, slate, *args):
        slate_diversities = avg_pairwise_diversity_from_matrix(slate, self.item_item_scores)
        return slate_diversities


def get_embedding_encoder(embedding_encoding_type, item_embeddings) -> EmbeddingEncoder:
    if embedding_encoding_type == 'sum':
        return SumHistoryEncoder(item_embeddings)

    elif embedding_encoding_type == 'mean':
        return MeanHistoryEncoder(item_embeddings)

    elif embedding_encoding_type == 'lstm':
        return LSTMHistoryEncoder(item_embeddings)

    elif embedding_encoding_type == 'transformer':
        return TransformerHistoryEncoder(item_embeddings)

    raise NotImplementedError(f'Embedding encoding type \'{embedding_encoding_type}\' not implemented!')


def get_slate_conditioning_encoder_from_matrix(slate_size, item_item_scores):
    return SlateDiversityEncoderFromDiversities(slate_size, item_item_scores)


def get_slate_conditioning_embedding_encoder(diversity_embedding_encoding_type, slate_size, item_embeddings):
    if diversity_embedding_encoding_type == 'cosine':
        return SlateDiversityEncoder(item_embeddings, slate_size, 'cosine')

    elif diversity_embedding_encoding_type == 'euclidean':
        return SlateDiversityEncoder(item_embeddings, slate_size, 'euclidean')




