from typing import List

import numpy as np
import torch

from components.list_cvae import ListCVAE
from components.cvae import CVAE
from components.encoders.embedding_encoders import get_embedding_encoder
from components.encoders.encoder_helper import get_slate_conditioning_encoders
from components.samplers import get_slate_sampler
from loss_functions import loss_function_posterior_and_prior
from utils.data_features_utils import get_item_item_diversities
from utils.embedding_utils import get_training_embeddings
from utils.training_evaluation_utils import METRIC_KEYS, generate_slate_metrics, generate_catalog_metrics


class ListCVAEModule(ListCVAE):
    def __init__(self, args, num_users, num_items):
        """
        :param args: args from command line
        :param num_users: number of users in dataset
        :param num_items: number of items in dataset
        """

        training_embeddings = get_training_embeddings(args, num_items)
        item_item_scores = get_item_item_diversities(args.dataset, args.variation, args.task, num_items)

        slate_sampler = get_slate_sampler(args.sample_type, slate_size=args.K)
        history_encoder = get_embedding_encoder(args.embedding_encoder, training_embeddings, item_item_scores)
        cvae_conditioning_size, diversity_encoder, slate_conditioning_encoder = get_slate_conditioning_encoders(
            history_encoder, item_item_scores, args.diverse_model, args.diversity_encoder, args.K)

        cvae_input_size = args.K * args.item_embedding_size
        cvae = CVAE(cvae_input_size, cvae_conditioning_size,
                    args.latent_size, args.encoder_hidden_size, args.prior_hidden_size, args.decoder_hidden_size)

        super(ListCVAEModule, self).__init__(args.K, cvae,
                                             slate_conditioning_encoder, diversity_encoder, history_encoder,
                                             slate_sampler, args.item_embedding_size, num_items, training_embeddings)

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_users = num_users
        self.item_item_scores = item_item_scores

        self.val_metrics = None

    def training_step(self, batch, batch_idx):
        slates = batch[1]
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
            cond_div = batch[1][:, 0].unsqueeze(1)
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
