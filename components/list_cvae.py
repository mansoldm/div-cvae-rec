from typing import List

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from components.cvae import CVAE
from components.encoders.embedding_encoders import HistoryEncoder, EmbeddingEncoder
from components.samplers import SlateSampler


class ListCVAE(pl.LightningModule):
    def __init__(self, slate_size: int, cvae: CVAE, slate_conditioning_encoder: EmbeddingEncoder,
                 diversity_encoder: EmbeddingEncoder, history_encoder: HistoryEncoder, slate_sampler: SlateSampler,
                 item_embedding_size: int, num_items: int, item_embeddings: nn.Embedding):

        super(ListCVAE, self).__init__()

        self.slate_size = slate_size
        self.cvae = cvae
        self.slate_conditioning_encoder = slate_conditioning_encoder
        self.diversity_encoder = diversity_encoder
        self.history_encoder = history_encoder
        self.slate_sampler = slate_sampler
        self.item_embedding_size = item_embedding_size
        self.num_items = num_items
        self.item_embeddings = item_embeddings

        self.bn = nn.BatchNorm1d(self.cvae.conditioning_size)

    def forward(self, *fwd_input: torch.Tensor):

        """
        :param fwd_input: slate, [metric], history, history_lengths, history_mask
        note that 'diversity' is a tensor of pointwise average diversitu scores
        :return: s_reconstruction: reconstructed slate
        """

        slate = fwd_input[0]
        history_inputs = fwd_input[-3:]

        if self.slate_conditioning_encoder is not None and self.diversity_encoder is not None:
            # we encode the slate with conditioning metric and put it in the conditioning input
            slate_conditioning_encoding = self.slate_conditioning_encoder(slate)
            conditioning_input = [slate_conditioning_encoding, *history_inputs]

        elif self.diversity_encoder is not None and len(fwd_input) > 4:
            # we have a separate metric at fwd_input[1] ready to be encoded
            conditioning_input = [fwd_input[1], *history_inputs]

        elif len(fwd_input) == 4:
            # just condition on the history i.e. non-diverse model
            conditioning_input = history_inputs

        else:
            raise NotImplementedError('Model not implemented')

        # encode history, response into conditioning
        history, conditioning = self.get_history_and_encoded_conditioning(*conditioning_input)
        slate_encoding = self.encode_slate(slate)
        s_embed_recon, Q_mean, Q_log_var, h_encoder, P_mean, P_log_var, h_decoder = self.cvae.forward(slate_encoding,
                                                                                                      conditioning)

        # reconstruction is (input_dim * embedding, 1) dimensional, want to reshape to (input_dim, embedding)
        # batched dimension is (batch, input_dim, embedding)
        match = self.compute_masked_match(history, s_embed_recon)

        return match, Q_mean, Q_log_var, P_mean, P_log_var

    def predict(self, *conditioning_input: List[torch.Tensor]) -> torch.Tensor:
        """
        when predicting, the metric will be given (if we are conditioning on it)
        """
        # drop userIds
        history, conditioning = self.get_history_and_encoded_conditioning(*conditioning_input)

        s_embed_recon = self.cvae.predict(conditioning)
        assert s_embed_recon.shape[0] == history.shape[0]

        match = self.compute_masked_match(history, s_embed_recon)
        batch_k_head_softmax = F.softmax(match, dim=-1)
        batch_samples = self.slate_sampler.forward(batch_k_head_softmax)

        return batch_samples

    def encode_slate(self, slate):
        """
        Encode slate by concatenating its tensors
        """
        # get tensor of (batch_size, input_size * item_embedding_size)
        emb_slate = self.item_embeddings(slate)
        batch_of_flattened_embedded_slates = torch.flatten(emb_slate, start_dim=1)
        return batch_of_flattened_embedded_slates

    def encode_conditioning(self, conditioning_input):
        """
        :param conditioning_input: slate_diversity, history, history_lengths, history_mask
        slate_diversity is the K-dimensional encoding of the slate
        :return:
        """
        # encode history (we always have the history
        history_inputs = conditioning_input[-3:]
        history_encoding = self.history_encoder(*history_inputs)

        if self.diversity_encoder is not None and len(conditioning_input) >= 4:
            # we have the encoded slate as average diversities: encode it
            encoded_slate = conditioning_input[0]
            diversity_encoding = self.diversity_encoder(encoded_slate).to(history_encoding.device)
            conditioning = torch.cat([history_encoding, diversity_encoding], dim=1)

        else:
            conditioning = history_encoding

        return conditioning

    def get_history_and_encoded_conditioning(self, *conditioning_input):
        history = conditioning_input[-3]
        conditioning = self.encode_conditioning(conditioning_input)
        conditioning = self.bn(conditioning)

        return history, conditioning

    def compute_masked_match(self, history: torch.Tensor, s_embed_recon: torch.Tensor):
        batch_size = s_embed_recon.shape[0]
        s_embed_recon_reshape = torch.reshape(s_embed_recon, (batch_size, self.slate_size, self.item_embedding_size))

        # match is a (batch_size, slate_size, num_items + 1) tensor
        emb = self.item_embeddings.weight
        # normalize
        match = torch.matmul(s_embed_recon_reshape, emb.t())
        
        match = self.mask_match(match, history)

        match = match[:, :, :self.num_items]
        return match

    def mask_match(self, match: torch.Tensor, history: torch.Tensor):
        # make history (batch_size, slate_size, -1) and use for masking
        device = next(self.parameters()).device
        hist = history.unsqueeze(1).repeat(1, self.slate_size, 1).type(torch.LongTensor).to(device)

        # mask out items already seen in the history
        match = match.scatter_(dim=2, index=hist, value=float('-inf'))

        return match
