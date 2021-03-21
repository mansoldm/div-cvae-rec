## Info 

This repository provides an implementation of the direct slate optimization paradigm based on Conditional VAE, which models distributions over slates and accepts a diversity feature as input. As a result, individual diversity can be controlled at inference time. 
Further experiments can easily be done using other metrics and/or different encoding strategies. In this repo we provide some alternative encodings for both user history and diversity input. 
Finally, this framework can easily be extended to support architectures such as a GAN or a sequence to sequence model.

## Data 

We define the dataset directory `data/[DATASET_NAME]/[DATASET_VARIATION]`, the idea being that we may want to handle datasets from the same source but of different version (e.g. number of interactions). For example, MovieLens 1M can be stored in the directory `data/ml/1m`.
The repo provides some scripts to experiment witn Netflix and MovieLens datasets. To use a different dataset, you must write a script which creates under `[DATASET_DIRECTORY]/original` the following files:
* `ratings.csv`: the set of all user-item interactions with columns `userId, itemId, rating, timestamp`.
* `items.csv`: the set of all items in the catalog with at least the column `itemId`.

The included data processing scripts expect the two files with the above specifications, and can be used to process the dataset in the format required by the architecture. These scripts will create files in `[DATASET_DIRECTORY]/processed`.
See the sample script `process_ml_1m.sh` for more context.
