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
                 word_num,
                 query_max_length, sent_max_length, user_buy_max_length,
                 mode, debug, user_buy=None):
        self.mode       = mode
        self.users      = users
        self.item_map   = item_map
        self.word_num   = word_num
        self.user_buy   = user_buy
        user_num        = len(users)
        item_num        = len(item_map)
        self.data       = []

        def query_extract(query):
            query = query[:query_max_length] if len(query) > query_max_length else \
                query + [word_num] * (query_max_length - len(query))
            return torch.tensor(query, dtype=torch.long)

        def hist_extract(item_hist):
            if len(item_hist) > user_buy_max_length:
                mask_hist = [0] * user_buy_max_length
                item_hist = np.random.choice(item_hist, size=user_buy_max_length, replace=False)
            else:
                mask_hist = [0] * len(item_hist) + [-10e6] * (user_buy_max_length - len(item_hist))
                item_hist = item_hist + [item_num] * (user_buy_max_length - len(item_hist))

            item_hist = torch.tensor(item_hist, dtype=torch.long)
            mask_hist = torch.tensor(mask_hist, dtype=torch.float32)
            return item_hist, mask_hist

        if mode == 'train':
            progress        = building_progress(df, debug, desc='iter train')
            self.user_buy   = dict()

            # load interactions as a dok matrix - for negative sampling
            self.ui_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
            self.iw_matrix = sp.dok_matrix((item_num, word_num), dtype=np.float32)

            for _, entry in progress:
                user = entry['userID']
                item = item_map[entry['asin']]
                words = eval(entry['reviewWords'])

                self.ui_matrix[user, item] = 1.0
                for word in words:
                    self.iw_matrix[item, word] = 1.0

                if user not in self.user_buy:
                    self.user_buy[user] = []
                if item in self.user_buy[user]:
                    self.user_buy[user].remove(item)
                self.user_buy[user].append(item)  # note that items are appended chronologically

                # select purchased items for each user
                item_hist, mask_hist = hist_extract(self.user_buy[user])

                query = query_extract(eval(entry['queryWords']))

                # words
                for word in eval(entry['reviewWords'])[:sent_max_length]:
                    self.data.append({
                        'user': user,
                        'item': item,
                        'word': word,
                        'query': query,
                        'item_hist': item_hist,
                        'mask_hist': mask_hist,
                    })
        elif mode == 'test':
            progress = building_progress(df, debug, desc='iter test')
            for _, entry in progress:
                user = entry['userID']
                item = item_map[entry['asin']]

                # select  purchased items for each user
                item_hist, mask_hist = hist_extract(self.user_buy[user])

                query = query_extract(eval(entry['queryWords']))

                self.data.append({
                    'user': user,
                    'item': item,
                    'query': query,
                    'item_hist': item_hist,
                    'mask_hist': mask_hist
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        if self.mode == 'train':
            return entry['user'], entry['item'], entry['query'], entry['word'], \
                   entry['item_hist'], entry['mask_hist'], \
                   entry['item_neg'], entry['word_neg_item'],
        else:
            return entry['user'], entry['item'], entry['query'], \
                   entry['item_hist'], entry['mask_hist']

    def item_sampling(self, neg_num, start=0, end=None):
        for idx, entry in enumerate(tqdm(self.data,
                                         desc='item sampling',
                                         total=len(self.data),
                                         ncols=117, unit_scale=True)):
            neg_items = sample(self.ui_matrix,
                               entry['user'] - len(self.item_map),
                               start, end, neg_num)
            self.data[idx]['item_neg'] = torch.tensor(neg_items, dtype=torch.long)

    def word_sampling(self, matrix, name: str, neg_num, start=0, end=None):
        """ Sample negative for each entity-word.
        :param matrix: matrix and name must match,
        :param name: ['user', 'item'],
        """
        for idx, entry in enumerate(tqdm(self.data,
                                         desc='word sampling for {}'.format(name),
                                         total=len(self.data),
                                         ncols=117, unit_scale=True)):
            entity = entry[name]
            neg_words = sample(matrix, entity, start, end, neg_num)
            self.data[idx]['word_neg_{}'.format(name)] = torch.tensor(neg_words, dtype=torch.long)

    @staticmethod
    def init(full_df: pd.DataFrame, user_buy_max_length):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length
