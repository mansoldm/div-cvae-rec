import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataloaders.train_dataloader import get_training_dataloader
from dataloaders.valtest_dataloader import get_validation_test_dataloader
from pl_modules.list_cvae_module import ListCVAEModule
from utils.arg_extractor import get_args
from utils.data_features_utils import get_num_users_items, get_item_item_diversities

TEST_MODEL_NAME = 'test_checkpoint.ckpt'
LAST_CHECKPOINT_NAME = 'last.ckpt'
TRAINING_LOG_DIR = 'train'

granularity = 0.01
DIVERSITIES = torch.arange(start=0, end=1 + granularity, step=granularity)


def train(args, num_users, num_items, result_dir, model_dir, checkpoint_dir):
    # dataloaders
    train_loader = get_training_dataloader(args, num_users, num_items, args.task)
    val_loader = get_validation_test_dataloader(args, num_items, args.task)

    # training module
    item_item_scores = get_item_item_diversities(args.dataset, args.variation, args.task, num_items)
    module = ListCVAEModule(args.lr, args.weight_decay, args.sample_type, args.embedding_encoder,
                            args.diversity_encoder, args.diverse_model, args.K, args.item_embedding_size,
                            args.latent_size, args.encoder_hidden_size, args.decoder_hidden_size,
                            args.prior_hidden_size, item_item_scores, num_users, num_items)
    print(module)

    eval_every_n = (args.num_epochs, args.evaluate_every_n)[args.evaluate_every_n != 0]
    curr_device = None if not torch.cuda.is_available() else -1

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
    )

    train_csv_logger = CSVLogger(result_dir, name=TRAINING_LOG_DIR)
    last_checkpoint_path = os.path.join(checkpoint_dir, LAST_CHECKPOINT_NAME)

    # train from last checkpoint
    if os.path.isfile(last_checkpoint_path):
        # [HACK] use version from the previous training that we are resuming
        train_csv_logger = CSVLogger(result_dir, name=TRAINING_LOG_DIR, version=train_csv_logger.version - 1)
        trainer = pl.Trainer(resume_from_checkpoint=last_checkpoint_path,
                             num_sanity_val_steps=0, max_epochs=args.num_epochs, check_val_every_n_epoch=eval_every_n,
                             gpus=curr_device, auto_select_gpus=True, callbacks=[checkpoint_callback, ],
                             logger=train_csv_logger)

    else:  # training from scratch
        trainer = pl.Trainer(num_sanity_val_steps=0, max_epochs=args.num_epochs, check_val_every_n_epoch=eval_every_n,
                             gpus=curr_device, auto_select_gpus=True, callbacks=[checkpoint_callback, ],
                             logger=train_csv_logger)
    # train!
    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=val_loader)

    if args.task == 'test':
        trainer.save_checkpoint(os.path.join(model_dir, TEST_MODEL_NAME))


def test(args, curr_device, num_items, num_users, test_checkpoint_path, test_csv_logger):
    # load pretrained model for inference
    item_item_scores = get_item_item_diversities(args.dataset, args.variation, args.task, num_items)
    module = ListCVAEModule.load_from_checkpoint(test_checkpoint_path,
                                                 lr=args.lr, weight_decay=args.weight_decay,
                                                 sample_type=args.sample_type,
                                                 embedding_encoder_type=args.embedding_encoder,
                                                 diversity_encoder_type=args.diversity_encoder,
                                                 is_diverse_model=args.diverse_model,
                                                 slate_size=args.K,
                                                 item_embedding_size=args.item_embedding_size,
                                                 latent_size=args.latent_size,
                                                 encoder_hidden_size=args.encoder_hidden_size,
                                                 decoder_hidden_size=args.decoder_hidden_size,
                                                 prior_hidden_size=args.prior_hidden_size,
                                                 item_item_scores=item_item_scores, num_users=num_users,
                                                 num_items=num_items)
    trainer = pl.Trainer(gpus=curr_device, auto_select_gpus=True, logger=test_csv_logger)

    # test
    if args.diverse_model:
        # test list-cvae with diversity conditioning
        for test_diversity in DIVERSITIES:
            print(f'Diversity: {test_diversity}')
            cond_diversity = torch.FloatTensor([test_diversity] * args.K)
            test_loader = get_validation_test_dataloader(args, num_items, args.task, cond_diversity)
            trainer.test(model=module, test_dataloaders=test_loader)

    else:  # model without variable diversity
        test_loader = get_validation_test_dataloader(args, num_items, args.task)
        trainer.test(model=module, test_dataloaders=test_loader)


def main():
    args = get_args()
    pl.seed_everything(args.seed)
    num_users, num_items = get_num_users_items(args.dataset, args.variation)
    result_dir = os.path.join('results', args.exp_name)
    model_dir = os.path.join(result_dir, 'model')
    checkpoint_dir = os.path.join(result_dir, 'checkpoint')
    if args.task == 'val':
        train(args, num_users, num_items, result_dir, model_dir, checkpoint_dir)

    elif args.task == 'test':
        test_checkpoint_path = os.path.join(model_dir, TEST_MODEL_NAME)
        curr_device = None if not torch.cuda.is_available() else -1
        test_csv_logger = CSVLogger(save_dir=result_dir, name=f'test')
        if args.train_before_testing:
            train(args, num_users, num_items, result_dir, model_dir, checkpoint_dir)

        test(args, curr_device, num_items, num_users, test_checkpoint_path, test_csv_logger)


if __name__ == "__main__":
    main()
