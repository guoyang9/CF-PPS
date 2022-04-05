import os
import time
import random
from argparse import ArgumentParser
from tools.transform_module import *
from params import parser_add_data_arguments


if __name__ == '__main__':
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    start_time = time.time()

    # --------------Create Paths-------------- #
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    if not os.path.exists(args.processed_path):
        os.makedirs(args.processed_path)

    # parse raw files from disk
    meta_path   = os.path.join(args.data_path, "meta_{}.json.gz".format(args.dataset))
    review_path = os.path.join(args.data_path, "reviews_{}_5.json.gz".format(args.dataset))
    review_df   = parse_review(review_path)
    categories, also_viewed = parse_meta(meta_path)

    # pre-extraction steps
    df, word_dict = parse_words(review_df, args.word_count, categories, args.dataset.split('_'))
    df = df.drop(['reviewerName', 'reviewTime', 'summary', 'overall', 'helpful'], axis=1)
    df = reindex(df)
    df = split_data(df)

    # write processed results to disk
    processed_path = os.path.join(args.processed_path, args.dataset)
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    json.dump(word_dict, open(os.path.join(processed_path, 'word_dict.json'), 'w'))
    df = remove_review(df, word_dict)  # remove the reviews from test set
    df.to_csv(os.path.join(processed_path, 'full.csv'), index=False)

    print("The number of {} users is {:d}; items is {:d}; feedbacks is {:d}; words is {:d}.".format(
        args.dataset, len(df.reviewerID.unique()), len(df.asin.unique()), len(df), len(word_dict)),
        "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
