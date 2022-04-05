import os
import json
from argparse import ArgumentParser

from AmazonDataset import *
from params import parser_add_data_arguments


if __name__ == '__main__':
    parser = ArgumentParser()

    parser_add_data_arguments(parser)
    args = parser.parse_args()
    np.random.seed(args.seed)

    dset_path   = os.path.join(args.processed_path, args.dataset)

    full_df     = pd.read_csv(os.path.join(dset_path, 'full.csv'))
    train_df    = full_df[full_df['filter'] == 'Train'].reset_index(drop=True)

    word_dict   = json.load(open(os.path.join(dset_path, 'word_dict.json'), 'r'))

    item_map    = init(full_df)
    tf, u_words = traverse_corpus(train_df, len(word_dict), item_map)

    ql_path     = os.path.join(dset_path, 'ql')
    if not os.path.exists(ql_path):
        os.makedirs(ql_path)

    np.save(os.path.join(ql_path, 'tf.npy'), tf)
    json.dump(item_map, open(os.path.join(ql_path, 'item_map.json'), 'w'))
    json.dump(u_words, open(os.path.join(ql_path, 'u_words.json'), 'w'))
