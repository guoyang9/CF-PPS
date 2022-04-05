import os
import json
import pandas as pd


def data_preparation(args):
    dset_path   = os.path.join(args.processed_path, args.dataset)
    full_df     = pd.read_csv(os.path.join(dset_path, 'full.csv'))
    train_df    = full_df[full_df['filter'] == 'Train'].reset_index(drop=True)
    # test_df     = pd.read_csv(os.path.join(dset_path, 'test.csv'))
    test_df     = full_df[full_df['filter'] == 'Test'].reset_index(drop=True)

    word_dict_path  = os.path.join(dset_path, 'word_dict.json')
    word_dict       = json.load(open(word_dict_path, 'r'))

    return train_df, test_df, full_df, word_dict
