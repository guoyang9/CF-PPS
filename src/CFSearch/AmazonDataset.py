import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.sampling import sample
from utils import building_progress


def pad_collate(batch, value):
    items, words, queries = zip(*batch)
    items   = torch.stack(items)
    queries = torch.stack(queries)
    words   = pad_sequence(words, batch_first=True, padding_value=value)
    return items, words, queries


class AmazonDataset(Dataset):
    def __init__(self, df, users, item_map: dict,
                 word_num, query_max_length, sent_max_length,
                 mode, debug, user_buy=None):
        self.mode       = mode
        self.users      = users
        self.item_map   = item_map
        self.user_buy   = user_buy
        self.data       = []
        self.user_num   = len(users)
        item_num        = len(item_map)

        def sent_extract(sentence, max_len):
            sentence = sentence[: max_len] if len(sentence) > max_len else \
                sentence + [word_num] * (max_len - len(sentence))
            return torch.tensor(sentence, dtype=torch.long)

        if mode == 'train':
            progress        = building_progress(df, debug, desc='iter train')
            self.user_buy   = dict()

            # load u-i interactions as a dok matrix - for negative sampling
            self.ui_matrix  = sp.dok_matrix((self.user_num, item_num), dtype=np.float32)

            # adjacency matrix - for GCN estimation
            self.adj_matrix = sp.dok_matrix((self.user_num + item_num,
                                             self.user_num + item_num),
                                            dtype=np.float32)

            for _, entry in progress:
                user  = entry['userID']
                item  = item_map[entry['asin']]
                words = eval(entry['reviewWords'])
                self.ui_matrix[user, item] = 1.0

                user += item_num  # entity - [item; user]
                if user not in self.user_buy:
                    self.user_buy[user] = []
                if item in self.user_buy[user]:
                    self.user_buy[user].remove(item)
                self.user_buy[user].append(item)  # note that items are appended chronologically

                self.adj_matrix[user, item] = 1.0
                self.adj_matrix[item, user] = 1.0

                words = sent_extract(words, sent_max_length)
                query = sent_extract(eval(entry['queryWords']), query_max_length)

                self.data.append({
                    'user': user,
                    'item': item,
                    'word': words,
                    'query': query
                })

        elif mode == 'test':
            progress = building_progress(df, debug, desc='iter test')
            for _, entry in progress:
                user    = entry['userID']
                item    = item_map[entry['asin']]

                user    += item_num
                query   = sent_extract(eval(entry['queryWords']), query_max_length)

                self.data.append({
                    'user': user,
                    'item': item,
                    'query': query
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        if self.mode == 'train':
            return entry['user'], entry['item'], entry['query'], \
                   entry['word'], entry['item_neg'],
        else:
            return entry['user'], entry['item'], entry['query']

    def item_sampling(self, neg_num, start=0, end=None):
        for idx, entry in enumerate(tqdm(self.data,
                                         desc='item sampling',
                                         total=len(self.data),
                                         ncols=117, unit_scale=True)):
            neg_items = sample(self.ui_matrix,
                               entry['user'] - len(self.item_map),
                               start, end, neg_num)
            self.data[idx]['item_neg'] = torch.tensor(neg_items, dtype=torch.long)

    def non_personalized_sampling(self, neg_num, start=0, end=None):
        """ Negative sampling without personalization. """
        for idx, entry in enumerate(tqdm(self.data,
                                         desc='item sampling',
                                         total=len(self.data),
                                         ncols=117, unit_scale=True)):
            negatives = []
            for _ in range(neg_num):
                j = np.random.randint(start, end)
                while j == entry['item'] or j in negatives:
                    j = np.random.randint(start, end)
                negatives.append(j)
            self.data[idx]['item_neg'] = torch.tensor(negatives, dtype=torch.long)

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length
