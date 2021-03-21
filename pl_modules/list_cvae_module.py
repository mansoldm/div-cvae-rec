from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from components.cvae import CVAE
from components.encoders.embedding_encoders import get_embedding_encoder
from components.encoders.encoder_helper import get_slate_conditioning_encoders
from components.samplers import get_slate_sampler
from loss_functions import loss_function_posterior_and_prior
from utils.training_evaluation_utils import METRIC_KEYS, generate_slate_metrics, generate_catalog_metrics


class ListCVAEModule(pl.LightningModule):
    def __init__(self, lr, weight_decay, sample_type, embedding_encoder_type, diversity_encoder_type, is_diverse_model,
                 slate_size, item_embedding_size, latent_size, encoder_hidden_size, decoder_hidden_size,
                 prior_hidden_size, item_item_scores, num_users, num_items):
        """

        @param lr: learning rate during training
        @param weight_decay: weight decay during training
        @param sample_type: what sampler to use when generating recommendations from K-headed distribution
        @param embedding_encoder_type: how to encode embedding-based conditioning (e.g. interaction history)
        @param diversity_encoder_type: how to encode diversity, or other metric we condition on (e.g. sum)
        @param is_diverse_model: whether to use the diverse model or only condition on history
        @param slate_size: dimension of slate to train on/generate
        @param item_embedding_size:
        @param latent_size:
        @param encoder_hidden_size:
        @param decoder_hidden_size:
        @param prior_hidden_size:
        @param item_item_scores: item-item matrix of diversity scores
        @param num_users: number of users in dataset
        @param num_items: number of items in dataset
        """

        super(ListCVAEModule, self).__init__()
        self.sample_type = sample_type
        self.embedding_encoder_type = embedding_encoder_type
        self.diversity_encoder_type = diversity_encoder_type
        self.is_diverse_model = is_diverse_model

        # architecture
        self.slate_size = slate_size
        self.item_embedding_size = item_embedding_size
        self.latent_size = latent_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.prior_hidden_size = prior_hidden_size

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_users = num_users
        self.num_items = num_items
        self.item_item_scores = item_item_scores

        self.val_metrics = None

        self.diversity_encoder = None
        self.slate_conditioning_encoder = None
        self.history_encoder = None

        self.build_module()

    def build_module(self):
        self.item_embeddings = nn.Embedding(self.num_items + 1, self.item_embedding_size, padding_idx=self.num_items)

        self.slate_sampler = get_slate_sampler(self.sample_type, slate_size=self.slate_size)
        self.history_encoder = get_embedding_encoder(self.embedding_encoder_type, self.item_embeddings)

        cvae_conditioning_size, \
        self.diversity_encoder, \
        self.slate_conditioning_encoder = get_slate_conditioning_encoders(
            self.history_encoder, self.item_item_scores, self.is_diverse_model, self.diversity_encoder_type,
            self.slate_size)

        cvae_input_size = self.slate_size * self.item_embedding_size
        self.cvae = CVAE(cvae_input_size, cvae_conditioning_size,
                         self.latent_size, self.encoder_hidden_size, self.prior_hidden_size, self.decoder_hidden_size)

        self.bn = nn.BatchNorm1d(cvae_conditioning_size)

    def forward(self, *fwd_input: torch.Tensor):

        """
        @param fwd_input: slate, [metric], history, history_lengths, history_mask
        note that 'diversity' is a tensor of pointwise average diversitu scores
        @return: unnormalized predictive distribution, parameters of encoder/prior distributions
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
        s_embed_recon, Q_mean, Q_log_var, h_encoder, P_mean, P_log_var, h_decoder = self.cvae(slate_encoding,
                                                                                              conditioning)

        # reconstruction is (input_dim * embedding, 1) dimensional, want to reshape to (input_dim, embedding)
        # batched dimension is (batch, input_dim, embedding)
        match = self.compute_masked_match(history, s_embed_recon)

        return match, Q_mean, Q_log_var, P_mean, P_log_var

    def predict(self, *conditioning_input: List[torch.Tensor]) -> torch.Tensor:
        """
        @param conditioning_input: [metric], history, history_lengths, history_mask
        @return: batch of sampled slates
        """
        # drop userIds
        history, conditioning = self.get_history_and_encoded_conditioning(*conditioning_input)

        s_embed_recon = self.cvae.predict(conditioning)
        assert s_embed_recon.shape[0] == history.shape[0]

        match = self.compute_masked_match(history, s_embed_recon)
        batch_k_head_softmax = F.softmax(match, dim=-1)
        batch_samples = self.slate_sampler(batch_k_head_softmax)

        return batch_samples

    def encode_slate(self, slate):
        """
        @param: slate
        @return: encoded_slate - slate encoded with model's embeddings and concatenated lengthwise
        """
        # get tensor of (batch_size, input_size * item_embedding_size)
        emb_slate = self.item_embeddings(slate)
        batch_of_flattened_embedded_slates = torch.flatten(emb_slate, start_dim=1)
        return batch_of_flattened_embedded_slates

    def encode_conditioning(self, conditioning_input):
        """
        @param conditioning_input: [metric] history, history_lengths, history_mask
        [metric] is a K-dimensional encoding of the slate (e.g. avg diversity of each item relative to all other items)
        @return: conditioning - encoded version of conditioning to be fed to the CVAE
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

        match = torch.matmul(s_embed_recon_reshape, self.item_embeddings.weight.t())
        masked_match = self.mask_match(match, history)

        return masked_match[:, :, :self.num_items]

    def mask_match(self, match: torch.Tensor, history: torch.Tensor):
        """
        @param match: the unnormalized K-headed distribution over catalog items
        @param history: user interaction history
        @return: unnormalized K-headed distribution with -inf score set for items in history
        """
        # make history (batch_size, slate_size, -1) and use for masking
        device = next(self.parameters()).device
        hist = history.unsqueeze(1).repeat(1, self.slate_size, 1).type(torch.LongTensor).to(device)

        # mask out items already seen in the history
        match = match.scatter_(dim=2, index=hist, value=float('-inf'))

        return match

    # PyTorch Lightning Module training/validation methods
    def training_step(self, batch, batch_idx):
        slates = batch[0]
        forward_result = self(*batch)
        d = loss_function_posterior_and_prior(slates, *forward_result)
        return d

    def training_epoch_end(self, batch_parts):
        lst = []
        keys = sorted(list(batch_parts[0].keys()))
        for part in batch_parts:
            metrics = []
            for key in keys:
                metrics.append(part[key].detach().cpu().numpy())
            lst.append(metrics)

        arr = np.stack(lst)
        avg_results = arr.mean(axis=0)

        print()
        if self.val_metrics is not None:
            d = {key: value for key, value in zip(keys, avg_results)}
            train_metrics = [('train_loss', d['loss'])] + \
                            sorted([(k, v) for k, v in list(d.items()) if k != 'loss'], key=lambda x: x[0])

            print(train_metrics)
            print(list(self.val_metrics.items()))

            # log merged dictionary
            self.log_dict({**self.val_metrics, **d})
            self.val_metrics = None

    def val_or_test_step(self, batch, stage: str):
        targets, target_lengths = batch[-2:]
        preds = self.predict(*batch[:-2])

        eval_in = [targets, target_lengths, self.item_item_scores]
        metrics = generate_slate_metrics(preds, *eval_in)
        if stage == 'val':
            res = torch.cat([preds.cpu(), metrics], dim=1)

        elif stage == 'test':
            # assumption: condition on the same diversity for all examples
            cond_div = batch[0][:, 0].unsqueeze(1)
            res = torch.cat([cond_div.cpu(), preds.cpu(), metrics], dim=1)

        return res

    def val_or_test_epoch_end(self, validation_step_output: List[torch.Tensor], stage: str):
        outputs_tensor = torch.cat(validation_step_output)
        if stage == 'test':
            cond_div = outputs_tensor[:, 0][0]
            outputs_tensor = outputs_tensor[:, 1:]

        slates, slate_metrics = outputs_tensor[:, :self.slate_size], outputs_tensor[:, self.slate_size:]
        avg_slate_metrics = torch.mean(slate_metrics, dim=0)
        var_slate_metrics = torch.var(slate_metrics, dim=0)
        std_err_slate_metrics = torch.sqrt(var_slate_metrics / self.num_users)

        # calculate catalog_level metrics
        catalog_metrics = generate_catalog_metrics(slates, self.num_items)

        avg_metrics = torch.cat([avg_slate_metrics, var_slate_metrics, std_err_slate_metrics, catalog_metrics])
        d = {f'{key}': float(value) for key, value in zip(METRIC_KEYS, avg_metrics)}

        if stage == 'val':
            self.val_metrics = d

        elif stage == 'test':
            d['div'] = cond_div.item()
            self.log_dict(d)

    def validation_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, stage='val')

    def validation_epoch_end(self, validation_step_output: List[torch.Tensor]):
        self.val_or_test_epoch_end(validation_step_output, stage='val')

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, stage='test')

    def test_epoch_end(self, test_step_output: List[torch.Tensor]):
        self.val_or_test_epoch_end(test_step_output, stage='test')

    def configure_optimizers(self, ):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
