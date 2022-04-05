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
                 sub_rate,
                 mode, debug, user_buy=None):
        self.df         = df
        self.mode       = mode
        self.users      = users
        self.word_num   = word_num
        self.item_map   = item_map
        self.user_buy   = user_buy
        self.data       = []
        self.user_num   = len(users)
        item_num        = len(item_map)

        def query_extract(query):
            query = query[:query_max_length] if len(query) > query_max_length else \
                query + [word_num] * (query_max_length - len(query))
            return torch.tensor(query, dtype=torch.long)

        if mode == 'train':
            # estimate word sub-sampling first
            self.sub_sampling_rate = self.subsample(sub_rate)

            progress        = building_progress(df, debug, desc='iter train')
            self.user_buy   = dict()

            # load u-i interactions as a dok matrix - for negative sampling
            self.ui_matrix = sp.dok_matrix((self.user_num, item_num), dtype=np.float32)
            self.uw_matrix = sp.dok_matrix((self.user_num, word_num), dtype=np.float32)
            self.iw_matrix = sp.dok_matrix((item_num, word_num), dtype=np.float32)

            for _, entry in progress:
                user  = entry['userID']
                item  = item_map[entry['asin']]
                words = eval(entry['reviewWords'])

                self.ui_matrix[user, item] = 1.0
                for word in words:
                    self.uw_matrix[user, word] = 1.0
                    self.iw_matrix[item, word] = 1.0

                # entities
                user += item_num  # entity - [item; user]
                if user not in self.user_buy:
                    self.user_buy[user] = []
                if item in self.user_buy[user]:
                    self.user_buy[user].remove(item)
                self.user_buy[user].append(item)  # note that items are appended chronologically

                query = query_extract(eval(entry['queryWords']))

                # words
                for word in eval(entry['reviewWords'])[:sent_max_length]:
                    if sub_rate == 0. or np.random.random() < self.sub_sampling_rate[word]:
                        self.data.append({
                            'user': user,
                            'item': item,
                            'word': word,
                            'query': query
                        })
        elif mode == 'test':
            progress = building_progress(df, debug, desc='iter test')
            for _, entry in progress:
                user    = entry['userID']
                item    = item_map[entry['asin']]
                user    += item_num
                query   = query_extract(eval(entry['queryWords']))

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
            return entry['user'], entry['item'], entry['query'], entry['word'], \
                entry['item_neg'], entry['word_neg_user'], entry['word_neg_item']
        else:
            return entry['user'], entry['item'], entry['query']

    def subsample(self, rate):
        """ Follow the original implementation. """
        sub_sampling_rate = np.ones(self.word_num)
        if not rate == 0.:
            word_dist = np.zeros(self.word_num)
            # estimate word distribution first
            for words in self.df['reviewWords']:
                for word in eval(words):
                    word_dist[word] += 1
            word_dist = word_dist * rate / (word_dist.sum() + 1e-6)

            sub_sampling_rate = np.clip(np.sqrt(word_dist + 1) * (1 / (word_dist + 1e-6)), a_min=0., a_max=1.)
        return sub_sampling_rate

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
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length
