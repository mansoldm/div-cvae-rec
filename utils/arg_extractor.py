import argparse
import json


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args(preset_args: list=None, print=True):
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_predictions', nargs="?", type=str2bool,
                        default=False, help='During testing, whether to save model predictions to file')
    parser.add_argument('--task', nargs="?", type=str,
                        choices={'val', 'test'},
                        default='test',
                        help='The stage of experimentation: hyperparameter tuning using validation set,'
                             ' or final model training with metrics computed on test set')
    parser.add_argument('--train_before_testing', type=str2bool, default=False, help='Train model on final set before computing performance on test set')
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False, help='Use a pretrained model or train one from scratch')
    parser.add_argument('--resume_training', type=str2bool, help='Resume training from last recorded epoch')
    parser.add_argument('--dataset', nargs="?", type=str,
                        default='ml', help='Name of dataset to use in experiment (e.g. \'ml\' for MovieLens)')
    parser.add_argument('--variation', nargs="?", type=str,
                        default='25m-trainval', help='Variation of dataset to use in experiment (e.g. \'25m\' for MovieLens)')
    parser.add_argument('--algorithm', nargs="?", type=str,
                        default='list-cvae', help='Type of algorithm to use')
    parser.add_argument('--diverse_model', nargs="?", type=str2bool,
                        default=True, help='Whether to use normal list-CVAE or diverse list-CVAE')
    parser.add_argument('--save_model', nargs="?", type=str2bool,
                        default=False,
                        help='Save best val model to disk (use in subsequent training, after finding best config from grid search)')
    parser.add_argument('--save_when', nargs="?", type=str,
                        default='end')
    parser.add_argument('--diversity_embeddings_name', nargs="?", type=str,
                        default='training', choices={'training', 'evaluation'},
                        help='Embeddings to calculate conditioning slate diversity during training. By design we either'
                             'choose the ones also used in evaluation or those learned during training.')
    parser.add_argument('--evaluation_embeddings_name', nargs="?", type=str,
                        default='val',
                        help='Embeddings to calculate conditioning slate diversity during inference. It is assumed that'
                             'one would store one or more sets of item embeddings in the dataset directory.')
    parser.add_argument('--use_pretrained_embeddings', nargs="?", type=str2bool,
                        default=False,
                        help='Whether to use pretrained embeddings (same size as item_embeddings_size)')
    parser.add_argument('--finetune_pretrained_embeddings', nargs="?", type=str2bool,
                        default=False,
                        help='When using pretrained embeddings, whether to fine tune them or keep them fixed')
    parser.add_argument('--pretrained_embeddings_name', nargs="?", type=str,
                        default='feature',
                        help='Name of pretrained embeddings to be used (if previous flags permit this)')
    parser.add_argument('--embedding_encoder', nargs='?', default='lstm', help='how to encode embedded history')
    parser.add_argument('--diversity_encoder', nargs='?', default='identity', help='how to encode diversity feature')
    parser.add_argument('--sample_type', nargs="?", type=str,
                        default='topk_deduplicate',
                        choices={'sample', 'argmax', 'argmax_deduplicate', 'topk_deduplicate'},
                        help='Type of sampling when creating slates')
    parser.add_argument('--shuffle_slates', nargs='?', default=True, type=str2bool, help='Shuffle slates during training'
                                                                          'Use when diversity encoding == identity or similar')

    parser.add_argument('--lr', nargs="?", type=float,
                        default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', nargs="?", type=float, default=1e-4,
                        help='Weight decay to use for Adam')
    parser.add_argument('--batch_size', nargs="?", type=int,
                        default=128, help='Batch_size for experiment')
    parser.add_argument('--num_epochs', nargs="?", type=int,
                        default=60, help='The experiment\'s epoch budget')
    parser.add_argument('--evaluate_every_n', nargs="?", type=int, default=5, help='Run validation after n epochs')
    parser.add_argument('--dataloader_type', nargs="?", type=str,
                        default='random', help='Which generator to use')
    parser.add_argument('--evaluation_embedding_size', nargs="?", type=int,
                        default=32, help='Dimension of eveluation item embeddings (for computing diversity)')
    parser.add_argument('--item_embedding_size', nargs="?", type=int,
                        default=64, help='Dimension of training item embeddings embedding')
    parser.add_argument('--latent_size', nargs="?", type=int,
                        default=16, help='Dimension of latent space z')
    parser.add_argument('--encoder_hidden_size', nargs="?", type=int,
                        default=256, help='Size of hidden layer in encoder')
    parser.add_argument('--decoder_hidden_size', nargs="?", type=int,
                        default=256, help='Size of hidden layer in decoder')
    parser.add_argument('--prior_hidden_size', nargs="?", type=int,
                        default=32, help='Size of hidden layer in prior MLP')

    parser.add_argument('--K', nargs="?", type=int,
                        default=10, help='Size K of recommendation slate i.e. number of items to be recommended')
    parser.add_argument('--K_rerank', nargs="?", type=int,
                        default=10, help='Size K of recommendation slate in DPP reranking')
    parser.add_argument('--truncate_targets', nargs="?",
                        type=str2bool,
                        default=False,
                        help='Whether user targets should be truncated (by --num_user_target_movies argument)')
    parser.add_argument('--t', nargs="?", type=int,
                        default=20, help='Number of items in user target set (for evaluation)')
    parser.add_argument('--num_factors', nargs="?", type=int,
                        default=-1, help='Dimensionality of user/item vectors')

    parser.add_argument('--num_neg', nargs='?', type=int, default=5, 
                          help='Number of negative samples (e.g. in BPR)')
    parser.add_argument('--bpr_lambda', nargs='?', type=float, default=0.001,
                          help='Regularization constant for BPR')

    parser.add_argument('--seed', nargs="?", type=int, default=1,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--exp_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--filepath_to_arguments_json_file', nargs="?", type=str, default=None,
                        help='')

    args = parser.parse_args()
    if preset_args is not None:
        new_args = extract_from_preset_args(preset_args, args)
        return print_and_return_args(new_args)

    if args.filepath_to_arguments_json_file is not None:
        args = extract_args_from_json(
            json_file_path=args.filepath_to_arguments_json_file, existing_args_dict=args)
        return print_and_return_args(args)

    if print:
        return print_and_return_args(args)
    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__ = adict


def print_and_return_args(my_args):
    arg_str = [(str(key), str(value)) for (key, value) in vars(my_args).items()]
    print(arg_str)
    return my_args


def extract_from_preset_args(preset_args: list, existing_args_dict):
    preset_dict = {}
    for k, v in preset_args:
        preset_dict[k] = v

    return join_arg_dicts(preset_dict, existing_args_dict)


def extract_args_from_json(json_file_path, existing_args_dict=None):

    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)

    return join_arg_dicts(arguments_dict, existing_args_dict)


def join_arg_dicts(arguments_dict, existing_args_dict):
    for key, value in vars(existing_args_dict).items():
        if key not in arguments_dict:
            arguments_dict[key] = value
        else :
            arguments_dict[key] = type(value)(arguments_dict[key])

    arguments_dict = AttributeAccessibleDict(arguments_dict)
    return arguments_dict
