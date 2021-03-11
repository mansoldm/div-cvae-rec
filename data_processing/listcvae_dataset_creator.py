import argparse

from tqdm import tqdm

from utils.load_write_utils import *
from utils.path_utils import *


def get_user_interactions(train_df, userId):
    user_history = train_df[train_df['userId'] == userId]['itemId'].values
    return user_history


def get_all_user_interactions(df, userIds):
    interactions = []
    for userId in tqdm(userIds):
        user_interactions = get_user_interactions(df, userId)
        interactions.append(user_interactions)

    return interactions


def generate_cvae_dataset(training_df, target_df, userIds, target_save_name='val'):
    training_interactions = get_all_user_interactions(training_df, userIds)
    target_interactions = get_all_user_interactions(target_df, userIds)

    proc_path = get_processed_path(dataset, variation)

    target_path = os.path.join(proc_path, target_save_name)
    write_arr_to_npz(target_path, 'histories', training_interactions, ignore_if_existing=True)
    write_arr_to_npz(target_path, 'targets', target_interactions, ignore_if_existing=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--variation', type=str)

    args = parser.parse_args()

    dataset = args.dataset
    variation = args.variation

    dv_path = get_dataset_variation_data_path(dataset, variation)
    proc_path = get_processed_path(dataset, variation)
    full_df = load_df_from_csv(proc_path, 'full.csv')

    itemIds = full_df['itemId'].unique()
    userIds = full_df['userId'].unique()

    print('Loading data...')
    train_df = load_df_from_csv(proc_path, 'train.csv')
    val_df = load_df_from_csv(proc_path, 'val.csv')
    test_df = load_df_from_csv(proc_path, 'test.csv')

    print('Generating dataset...')
    # train/val histories, slates, full_histories and targets
    # for hyperparameter search
    generate_cvae_dataset(train_df, val_df, userIds, 'val')

    # trainval/test histories for model evaluation
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    generate_cvae_dataset(trainval_df, test_df, userIds, 'test')
