import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import building_progress


def pad_collate(batch, value):
    items, words, queries = zip(*batch)
    items   = torch.stack(items)
    queries = torch.stack(queries)
    words   = pad_sequence(words, batch_first=True, padding_value=value)
    return items, words, queries


class AmazonDataset(Dataset):
    def __init__(self, df, item_map: dict,
                 word_num,
                 query_max_length, sent_max_length, window_size,
                 mode, debug, user_buy=None):
        self.mode       = mode
        self.item_map   = item_map
        self.user_buy   = user_buy
        self.data       = []

        def query_extract(query):
            query = query[:query_max_length] if len(query) > query_max_length else \
                query + [word_num] * (query_max_length - len(query))
            return torch.tensor(query, dtype=torch.long)

        if mode == 'train':
            progress    = building_progress(df, debug, desc='iter train')
            for _, entry in progress:
                item = item_map[entry['asin']]

                query = query_extract(eval(entry['queryWords']))

                # words
                words = eval(entry['reviewWords'])[:sent_max_length]
                padded_words = [word_num] * (window_size // 2) + \
                               words + [word_num] * (window_size // 2)
                for index in range(len(words)):
                    word = torch.tensor(padded_words[
                                        index: index + window_size], dtype=torch.long)
                    self.data.append({
                        'item': item,
                        'word': word,
                        'query': query
                    })
        elif mode == 'test':
            progress = building_progress(df, debug, desc='iter test')
            for _, entry in progress:
                item    = item_map[entry['asin']]
                query   = query_extract(eval(entry['queryWords']))

                self.data.append({
                    'item': item,
                    'query': query
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        if self.mode == 'train':
            return entry['item'], entry['word'], entry['query'], entry['item_neg']
        else:
            return entry['item'], entry['query']

    def item_sampling(self, neg_num, start=0, end=None):
        for idx, entry in enumerate(tqdm(self.data,
                                         desc='item sampling',
                                         total=len(self.data),
                                         ncols=117, unit_scale=True)):
            item = entry['item']

            neg_items = []
            for _ in range(neg_num):
                j = np.random.randint(start, end)
                while j == item or j in neg_items:
                    j = np.random.randint(start, end)
                neg_items.append(j)

            self.data[idx]['item_neg'] = torch.tensor(neg_items, dtype=torch.long)

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length
