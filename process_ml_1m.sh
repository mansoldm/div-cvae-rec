#!/usr/bin/env bash

# Run script specific to dataset to create ratings.csv and items.csv
python data_processing/ml_dat_to_csv.py --dataset ml --variation 1m

# Clean up dataset, create CSVs, sparse matrices and cvae numpy arrays
python data_processing/process_data.py --dataset ml --variation 1m --min_users 10 --min_items 20
python data_processing/listcvae_dataset_creator.py --dataset ml --variation 1m

# Generate diversities from sparse matrices
python generate_item_item_diversities.py --dataset ml --variation 1m --name train  --task val
python generate_item_item_diversities.py --dataset ml --variation 1m --name test  --task test
