import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset
from utils import building_progress
from utils.sampling import sample


class AmazonDataset(Dataset):
    def __init__(self, df, user_num,
                 item_map: dict, query2id: dict,
                 doc2vecs: dict,
                 debug, is_training: bool, user_buy=None):
        self.user_buy       = user_buy
        self.doc2vecs       = doc2vecs
        self.is_training    = is_training
        self.index2item     = {index: item for item, index in item_map.items()}

        self.data           = []

        if self.is_training:
            progress        = building_progress(df, debug, desc='iter train')
            self.user_buy   = dict()

            # load interactions as a dok matrix - for negative sampling
            self.ui_matrix = sp.dok_matrix((user_num, len(item_map)), dtype=np.float32)

            for _, entry in progress:
                user    = entry['userID']
                asin    = entry['asin']
                item = item_map[asin]
                query   = query2id[entry['queryWords']]

                self.ui_matrix[user, item] = 1.0
    
                if user not in self.user_buy:
                    self.user_buy[user] = []
                if item in self.user_buy[user]:
                    self.user_buy[user].remove(item)
                self.user_buy[user].append(item)  # note that items are appended chronologically

                # entities
                item = torch.tensor(self.doc2vecs[asin])

                # queries
                query = torch.tensor(self.doc2vecs[query])

                self.data.append({
                    'user': user,
                    'item': item,
                    'query': query,
                })
        else:
            progress = building_progress(df, debug, desc='iter test')
            for _, entry in progress:
                user    = entry['userID']
                asin    = entry['asin']
                query   = query2id[entry['queryWords']]

                # entities
                # item = torch.tensor(self.doc2vecs[asin])
                item = item_map[asin]

                # queries
                query = torch.tensor(self.doc2vecs[query])

                self.data.append({
                    'user': user,
                    'item': item,
                    'query': query,
                })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        if self.is_training:
            return entry['user'], entry['item'], entry['query'],\
                   entry['item_neg']
        else:
            return entry['user'], entry['item'], entry['query']

    def item_sampling(self, neg_num, start=0, end=None):
        for index, entry in enumerate(tqdm(self.data,
                                           desc='item sampling',
                                           total=len(self.data),
                                           ncols=117, unit_scale=True)):
            neg_items = sample(self.ui_matrix, entry['user'], start, end, neg_num)

            neg_items = torch.tensor(np.array([self.doc2vecs[self.index2item[index]]
                                      for index in neg_items]), dtype=torch.float32)
            self.data[index]['item_neg'] = neg_items

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        return users, item_map
