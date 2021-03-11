import os

DATA_DIR = 'data'


def get_dataset_variation_data_path(dataset: str, variation: str) -> str:
    return os.path.join(DATA_DIR, dataset, variation)


def get_orig_path(dataset, variation) -> str:
    dv_path = get_dataset_variation_data_path(dataset, variation)
    return os.path.join(dv_path, 'original')


def get_processed_path(dataset, variation) -> str:
    dv_path = get_dataset_variation_data_path(dataset, variation)
    return os.path.join(dv_path, 'processed')
