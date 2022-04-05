import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from argparse import ArgumentParser
from src.utils.sampling import sample
from params import parser_add_data_arguments


def build_neg(df_full, df_test, neg_num) -> pd.DataFrame:
    """
    :param df_full: full review file,
    :param df_test: test revie file,
    :param neg_num: number of negtives,
    """
    users = df_full['userID'].unique()
    items = df_full['asin'].unique()

    # temporary build dicts
    item_map    = dict(zip(items, range(len(items))))
    ui_matrix   = sp.dok_matrix((len(users), len(item_map)), dtype=np.float32)

    item_info   = dict()
    user_map    = dict()
    for _, entry in tqdm(df_full.iterrows(),
                         desc='iter interactions',
                         total=len(df_full),
                         ncols=117, unit_scale=True):
        user = entry['userID']
        item = item_map[entry['asin']]

        ui_matrix[user, item] = 1.0
        user_map[user] = entry['reviewerID']

        # for building new dataframe later
        if item not in item_info:
            item_info[item] = {
                'asin': entry['asin'],
                'query': eval(entry['query']),
                'queryWords': eval(entry['queryWords'])
            }

    # sample negatives for each user
    cands_user = dict()

    for _, entry in tqdm(df_test.iterrows(),
                         desc='iter test',
                         total=len(df_test),
                         ncols=117, unit_scale=True):
        user = entry['userID']
        item = item_map[entry['asin']]

        negatives = sample(ui_matrix, user, 0, len(item_map), neg_num)
        cands_user[user] = [item] + [i for i in negatives]

    # build the new test dataframe
    data_frame = []
    for user in tqdm(cands_user, desc='iter user',
                     total=len(cands_user),
                     ncols=117, unit_scale=True):
        gt_item     = cands_user[user][0]
        query       = item_info[gt_item]['query']
        query_words = item_info[gt_item]['queryWords']
        for item in cands_user[user]:
            info = {
                'userID': user,
                'reviewerID': user_map[user],
                'asin': item_info[item]['asin'],
                'query': query,
                'queryWords': query_words
            }
            data_frame.append(info)
    return pd.DataFrame(data_frame)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    args = parser.parse_args()
    np.random.seed(args.seed)

    dset_path   = os.path.join(args.processed_path, args.dataset)
    df_full     = pd.read_csv(os.path.join(dset_path, 'full.csv'))
    df_test     = df_full[df_full['filter'] == 'Test'].reset_index(drop=True)

    df_test     = build_neg(df_full, df_test, args.candidate - 1)
    df_test.to_csv(os.path.join(dset_path, 'test.csv'), index=False)
