import gzip
import json
import random
import itertools
import pandas as pd
from tqdm import tqdm
from tools import text_module


def parse_review(path):
    """ Apply raw data to pandas DataFrame. """
    idx, df = 0, dict()
    with gzip.open(path, 'rb') as fd:
        progress = tqdm(fd, desc='parsing reviews to dataframe',
                        unit_scale=True,
                        total=len(gzip.open(path, 'rb').readlines()))
        for line in progress:
            df[idx] = json.loads(line)
            idx    += 1
    return pd.DataFrame.from_dict(df, orient='index')


def reindex(df):
    """ Reindex the reviewID from 0 to total length. """
    reviewer        = df['reviewerID'].unique()
    reviewer_map    = dict(zip(reviewer, range(len(reviewer))))

    userIDs         = [reviewer_map[df['reviewerID'][i]] for i in range(len(df))]
    df['userID']    = userIDs
    return df


def parse_meta(meta_path):
    """
    Extract useful information (i.e., categories, related) from meta file.
    """
    with gzip.open(meta_path, 'rb') as fd:
        categories, also_viewed = {}, {}
        progress = tqdm(fd, desc='parsing meta',
                        unit_scale=True,
                        total=len(gzip.open(meta_path, 'rb').readlines()))
        for line in progress:
            line = eval(line)
            asin = line['asin']
            if 'category' in line:
                categories[asin] = line['category']
            elif 'categories' in line:
                categories[asin] = random.choice(line['categories'])
            else:
                raise Exception('categories tag not in metadata')
            related = line['related'] if 'related' in line else None

            # fill the also_related dictionary
            also_viewed[asin] = []
            relations = ['also_viewed', 'buy_after_viewing']
            if related:
                also_viewed[asin] = [related[r] for r in relations if r in related]
                also_viewed[asin] = itertools.chain.from_iterable(also_viewed[asin])
    return categories, also_viewed


def parse_words(review_df, min_num, categories, default_query):
    """ Parse words for both reviews and queries. """
    queries, reviews = [], []
    word_set = set()
    progress = tqdm(range(len(review_df)),
                    desc='parsing review and query words',
                    total=len(review_df), unit_scale=True)
    for i in progress:
        asin        = review_df['asin'][i]
        review      = review_df['reviewText'][i]
        category    = categories[asin] if categories.get(asin) else default_query
        category    = ' '.join(map(str, category))

        # process queries
        query = text_module.remove_dup(text_module.tokenizer(category))
        for word in query:
            word_set.add(word)

        # process reviews
        review = text_module.tokenizer(review)

        queries.append(query)
        reviews.append(review)

    review_df['query'] = queries  # write query result to dataframe

    # filtering words counts less than min_num
    reviews = text_module.filter_words(reviews, min_num)
    for review in reviews:
        for word in review:
            word_set.add(word)
    review_df['reviewText'] = reviews
    word_dict = dict(zip(word_set, range(len(word_set))))  # start from 0
    review_df['queryWords'] = [[word_dict[word] for word in query] for query in queries]

    return review_df, word_dict


def split_data(df):
    """ Splitting data into training and testing."""
    split_indicator = []

    df = df.sort_values(by=['userID', 'unixReviewTime'])
    user_length = df.groupby('userID').size().tolist()
    progress = tqdm(range(len(user_length)), desc='splitting data', total=len(user_length), unit_scale=True)
    for index in progress:
        length = user_length[index]
        # tag = ['Train' for _ in range(int(length * 0.7))]
        # tag_test = ['Test' for _ in range(length - int(length * 0.7))]
        tag         = ['Train' for _ in range(length - 1)]
        tag_test    = ['Test']
        tag.extend(tag_test)
        if length == 1:
            tag = ['Train']
        # np.random.shuffle(tag)
        split_indicator.extend(tag)

    df['filter'] = split_indicator
    return df


def remove_review(df, word_dict):
    """ Remove test review data and remove duplicate reviews."""
    review_text, review_words, review_train_set = [], [], set()

    df      = df.reset_index(drop=True)
    df_test = df[df['filter'] == 'Test'].reset_index(drop=True)

    # review in test
    review_test = set(repr(df_test['reviewText'][i]) for i in range(len(df_test)))

    progress = tqdm(range(len(df)), desc='removing reviews', total=len(df), unit_scale=True)
    for i in progress:
        r = repr(df['reviewText'][i])
        if r not in review_train_set and r not in review_test:
            review_train_set.add(r)
            review_text.append(eval(r))
            review_words.append([word_dict[w] for w in eval(r)])
        else:
            review_text.append([])
            review_words.append([])
    df['reviewText']    = review_text
    df['reviewWords']   = review_words
    return df
