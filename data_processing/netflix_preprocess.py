import argparse
import os
from collections import defaultdict

import pandas as pd
import datetime as dt

from utils.path_utils import get_orig_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='netflix')
    parser.add_argument('--variation', type=str, default='prize')

    args = parser.parse_args()

    keys = ['userId', 'rating', 'itemId', 'timestamp']
    filenames = [f'combined_data_{i}.txt' for i in range(1, 5)]
    orig_path = get_orig_path(args.dataset, args.variation)

    rat_path = os.path.join(orig_path, 'ratings.csv')
    if not os.path.exists(rat_path): 
        ratings = open(os.path.join(orig_path, 'ratings.csv'), 'a+')
        ratings.write(','.join(keys))

        joint_dict = defaultdict(list)
        for filename in filenames:
            path = os.path.join(orig_path, filename)
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line: continue
                    if ':' == line[-1]:
                        itemId = str(line[:-1])
                    else:
                        userId, rating, date = line.strip().split(',')
                        timestamp = str(dt.datetime.timestamp(dt.datetime.strptime(date, '%Y-%m-%d')))
                        print(','.join([userId, rating, itemId, timestamp]), file=ratings)

    df = pd.read_csv(os.path.join(orig_path, 'ratings.csv'),)
    df.drop_duplicates(subset=['userId', 'itemId'], keep='last', inplace=True)

    df.to_csv(os.path.join(orig_path, 'ratings.csv'), index=False)
    item_dict = {'itemId': df['itemId'].unique()}
    pd.DataFrame.from_dict(item_dict).to_csv(os.path.join(orig_path, 'items.csv'), index=False)


if __name__ == "__main__":
    main()
