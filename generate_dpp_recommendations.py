import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger

from dataloaders.valtest_dataloader import get_validation_test_dataloader
from pl_modules.dpp_reranker_module import DPPReRanker
from pl_modules.list_cvae_module import ListCVAEModule
from utils.arg_extractor import get_args
from utils.data_features_utils import get_num_users_items, get_item_item_diversities


def main():
    args = get_args()
    pl.seed_everything(args.seed)

    num_users, num_items = get_num_users_items(args.dataset, args.variation)
    result_dir = os.path.join('results', f'{args.exp_name}')
    result_dir_name = f'dpp-{args.exp_name}' if args.exp_name.endswith(str(args.K)) else f'dpp-{args.exp_name}-{args.K}'
    result_dir_dpp = os.path.join('results', result_dir_name)
    test_checkpoint_path = os.path.join(result_dir, 'model', 'test_checkpoint.ckpt')
    curr_device = None if not torch.cuda.is_available() else -1
    test_csv_logger = CSVLogger(save_dir=result_dir_dpp, name=f'test')

    trainer = pl.Trainer(gpus=curr_device, auto_select_gpus=True, logger=test_csv_logger)

    rec_module = ListCVAEModule.load_from_checkpoint(test_checkpoint_path,
                                                 args=args, num_users=num_users, num_items=num_items)

    item_item_diversities = get_item_item_diversities(args.dataset, args.variation, args.task, num_items)
    dpp_reranker_module = DPPReRanker(args, rec_module, num_users, num_items, item_item_diversities)

    test_loader = get_validation_test_dataloader(args, num_items, args.task)
    trainer.test(model=dpp_reranker_module, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
