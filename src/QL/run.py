import os
import json
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser

from utils.metrics import display
from utils.metrics import hit, mrr, ndcg
from params import parser_add_data_arguments

from AmazonDataset import *


def ql(query, tf, prior_p, mu, doc_len_items, bought):
    """
    :param query: query words,
    :param tf: term freq for each item, [word_num, item_num]
    :param prior_p: word dist over the whole corpus, [word_num, ]
    :param mu: hyper-param,
    :param doc_len_items: document length for all items,
    :param bought: the bought items for this user,
    :return: prob for each item, [item_num, ]
    """

    # query side: term frequency
    word_count  = Counter(query)
    query       = list(word_count.keys())
    freq_query  = np.array(list(word_count.values()), dtype=np.float32) # [word_num, ]

    # log side
    freq_doc    = (tf[query] + np.expand_dims(mu * prior_p[query], axis=1)) / (doc_len_items + mu)
    freq_doc    = np.log(freq_doc + 1e-6)
    prob        = np.sum(np.expand_dims(freq_query, axis=1) * freq_doc, axis=0)

    # mask bought items
    items_buy   = np.ones_like(prob)
    items_buy[bought] = 0.0 # for whole set test
    prob        = prob * items_buy
    prob        = prob / (prob.sum() + 1e-6)
    return prob


def uql(user, query, tf, prior_p, mu, _lambda, doc_len_items, bought):
    prob_q_i    = ql(query, tf, prior_p, mu, doc_len_items, bought)
    prob_u_i    = ql(user, tf, prior_p, mu, doc_len_items, bought)
    prob        = _lambda * prob_q_i + (1 - _lambda) * prob_u_i
    return prob / (prob.sum() + 1e-6)


def run(model_str, mu, lam):
    parser = ArgumentParser()
    parser_add_data_arguments(parser)
    args = parser.parse_args()

    # ------------------------prepare for data------------------------ #
    dset_path   = os.path.join(args.processed_path, args.dataset)
    ql_path     = os.path.join(dset_path, 'ql')
    full_path   = os.path.join(dset_path, 'full.csv')
    test_path   = os.path.join(dset_path, 'test.csv')

    full_df     = pd.read_csv(full_path)
    train_df    = full_df[full_df['filter'] == 'Train'].reset_index(drop=True)
    # test_df     = pd.read_csv(test_path)
    test_df     = full_df[full_df['filter'] == 'Test'].reset_index(drop=True)

    # ------------------------load from disk-------------------------- #
    tf          = np.load(os.path.join(ql_path, 'tf.npy'))
    u_words     = json.load(open(os.path.join(ql_path, 'u_words.json'), 'r'))
    item_map    = json.load(open(os.path.join(ql_path, 'item_map.json'), 'r'))

    doc_len_items       = tf.sum(axis=0) # doc length for each item
    word_distribution   = tf.sum(axis=1) / tf.sum(axis=1).sum(keepdims=True)

    # ---------------------------------------------------------------- #
    test_dataset    = AmazonDataset(test_df, item_map)
    user_buy        = get_buy(train_df, item_map)

    Mrr, Hr, Ndcg   = [], [], []

    candidate_items = []
    for user, item, query in tqdm(test_dataset, desc='test',
                                  total=test_dataset.__len__(),
                                  ncols=117, unit_scale=True):
        # append all tested items for the current user
        # candidate_items.append(item)
        # if len(candidate_items) < args.candidate:
        #     continue

        # loaded all candidate items for the current user
        bought  = user_buy[user]
        user    = u_words[str(user)]
        if model_str == 'ql':
            prob = ql(query,
                      # np.take(tf, candidate_items, axis=1),
                      tf,
                      word_distribution,
                      mu,
                      # np.take(doc_len_items, candidate_items, axis=0),
                      doc_len_items,
                      bought)
        elif model_str == 'uql':
            prob = uql(user,
                       query,
                       # np.take(tf, candidate_items, axis=1),
                       tf,
                       word_distribution,
                       mu,
                       lam,
                       # np.take(doc_len_items, candidate_items, axis=0),
                       doc_len_items,
                       bought)

        else:
            raise NotImplementedError

        ranking_list = np.argsort(prob)[::-1][:args.top_k].tolist()

        # for candidate test
        # return_list = np.array(candidate_items)[ranking_list].tolist()
        # Hr.append(hit(candidate_items[0], return_list))
        # Mrr.append(mrr(candidate_items[0], return_list))
        # Ndcg.append(ndcg(candidate_items[0], return_list))

        Hr.append(hit(item, ranking_list))
        Mrr.append(mrr(item, ranking_list))
        Ndcg.append(ndcg(item, ranking_list))

        candidate_items = []  # reset for next user
    display(0, 1, 0, np.mean(Hr), np.mean(Mrr), np.mean(Ndcg))


def run_ql():
    for mu in [2000, 6000, 10000]:
        print('++++++++++++++QL, mu: {}++++++++++++++'.format(mu))
        run('ql', mu, None)


def run_uql():
    for mu in [2000, 6000, 10000]:
        for lam in [0.2, 0.4, 0.6, 0.8, 1.0]:
            print('++++++++++++++UQL, mu: {}, lambda: {}++++++++++++++'.format(mu, lam))
            run('uql', mu, lam)


if __name__ == '__main__':
    run_ql()
    run_uql()
