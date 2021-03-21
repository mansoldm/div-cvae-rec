import itertools
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch

from utils.dpp import dpp
from utils.training_evaluation_utils import METRIC_KEYS, generate_slate_metrics, generate_catalog_metrics


class DPPReRanker(pl.LightningModule):

    def __init__(self, slate_size, rec_module: pl.LightningModule, num_users, num_items,
                 item_item_diversities: torch.FloatTensor):
        """
        @param slate_size: size of output recommendation. Note that this should be smaller than the size of the candidate
        set used by the underlying module rec_module
        @param item_item_diversities:
        @param rec_module: A ListCVAE instance
        @param num_users: number of users in dataset
        @param num_items: number of items in dataset
        """
        super(DPPReRanker, self).__init__()
        self.__dict__.update(args.__dict__)

        self.num_users = num_users
        self.num_items = num_items

        self.rec_module = rec_module
        self.slate_size = slate_size
        self.item_item_diversities = item_item_diversities
        self.kernel_matrix = 1 - item_item_diversities
        self.val_metrics = None

    def predict(self, *pred_args):
        slates = self.rec_module.predict(*pred_args)
        batch_size = slates.shape[0]
        K = slates.shape[1]
        item_item_scores = self.kernel_matrix

        # generate row-column pairs
        pairs = torch.Tensor(list(itertools.product(torch.arange(0, K), repeat=2))).long()
        # get corresponding items in generated slates
        slate_item_rows, slate_item_cols = slates[:, pairs[:, 0]], slates[:, pairs[:, 1]]
        scores = item_item_scores[slate_item_rows, slate_item_cols]
        kernel_matrices = torch.reshape(scores, (batch_size, K, K))

        rec_idxs = torch.stack([torch.from_numpy(dpp(kernel.cpu().numpy(), max_length=self.slate_size)) for kernel in kernel_matrices])
        recs = torch.gather(slates, dim=-1, index=rec_idxs.to(slates.device))
        return recs

    def test_step(self, batch, batch_idx):
        targets, target_lengths = batch[-2:]
        preds = self.predict(*batch[:-2])

        eval_in = [targets, target_lengths, self.item_item_diversities]
        metrics = generate_slate_metrics(preds, *eval_in)
        res = torch.cat([preds.cpu(), metrics], dim=1)
        return res

    def test_epoch_end(self, test_step_output: List[torch.Tensor]):
        outputs_tensor = torch.cat(test_step_output)
        slates, slate_metrics = outputs_tensor[:, :self.slate_size], outputs_tensor[:, self.slate_size:]
        avg_slate_metrics = torch.mean(slate_metrics, dim=0)
        var_slate_metrics = torch.var(slate_metrics, dim=0)
        std_err_slate_metrics = torch.sqrt(var_slate_metrics / self.num_users)

        # calculate catalog_level metrics
        catalog_metrics = generate_catalog_metrics(slates, self.num_items)

        avg_val_metrics = torch.cat([avg_slate_metrics, var_slate_metrics, std_err_slate_metrics, catalog_metrics])
        d = {f'{key}': float(value) for key, value in zip(METRIC_KEYS, avg_val_metrics)}

        self.log_dict(d)

    def forward(self, *batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, batch_parts):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_output: List[torch.Tensor]):
        pass

    def configure_optimizers(self,):
        pass
