import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain


def init(full_df: pd.DataFrame):
    items       = full_df['asin'].unique()
    item_map    = dict(zip(items, range(len(items))))
    return item_map


def get_buy(train_df: pd.DataFrame, item_map) -> dict:
    user_buy = dict()
    for _, entry in tqdm(train_df.iterrows(),
                         desc='iter train',
                         total=len(train_df),
                         ncols=117, unit_scale=True):
        user = entry['userID']
        item = item_map[entry['asin']]

        if user not in user_buy:
            user_buy[user] = []
        if item in user_buy[user]:
            user_buy[user].remove(item)
        user_buy[user].append(item) # note that items are appended chronologically
    return user_buy


def traverse_corpus(train_df: pd.DataFrame, word_num, item_map):
    users = train_df.groupby('userID')
    items = train_df.groupby('asin')

    tf      = np.zeros((word_num, len(item_map)), dtype=np.int32) # counted words for each product
    u_words = {} # counted words for each user

    def concat_words(asin):
        item = item_map[asin[0]]
        for review in asin[1]['reviewWords']:
            for word in eval(review):
                tf[word][item] += 1

    def most_frequent_words(user):
        """ Extract frequencies large than 20 words for each user. """
        userID      = user[0]
        word_count  = {}
        for review in user[1]['reviewWords']:
            for word in eval(review):
                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1

        u_words[userID] = [[w] * word_count[w] for w in word_count if word_count[w] > 50]
        u_words[userID] = list(chain.from_iterable(u_words[userID]))
        if len(u_words[userID]) == 0:
            u_words[userID] = [np.random.randint(0, word_num)] # the word will be ignored anyway

    for u in tqdm(users, desc='iter users', total=len(users), unit_scale=True):
        most_frequent_words(u)
    for i in tqdm(items, desc='iter items', total=len(items), unit_scale=True):
        concat_words(i)

    return tf, u_words


class AmazonDataset(object):
    def __init__(self, test_df: pd.DataFrame, item_map):
        self.test_df    = test_df
        self.item_map   = item_map

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, index):
        if index == self.__len__():
            raise IndexError
        return self.test_df['userID'][index], \
                    self.item_map[self.test_df['asin'][index]], \
                    eval(self.test_df['queryWords'][index])
