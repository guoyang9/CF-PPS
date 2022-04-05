import os
import json
import random
import argparse

import pandas as pd
from tqdm import tqdm
from gensim.models import doc2vec
from params import parser_add_data_arguments


parser = argparse.ArgumentParser()
parser_add_data_arguments(parser)
parser.add_argument("--window_size",
                    type=int,
                    default=3,
                    help="sentence window size")
args = parser.parse_args()

random.seed(args.seed)

# -------------------------- Data Preparation ------------------------- #
dset_path = os.path.join(args.processed_path, args.dataset)
full_data = pd.read_csv(os.path.join(dset_path, 'full.csv'))

qidx, query2id, documents = 0, dict(), dict()
for _, entry in tqdm(full_data.iterrows(),
                     desc='iter data',
                     total=len(full_data),
                     ncols=117, unit_scale=True):
    item    = entry['asin']
    query   = entry['query']
    q_words = entry['queryWords']
    review  = entry['reviewText']

    # concatenate words for each item
    if item not in documents:
        documents[item] = []
    documents[item].extend(eval(review))

    # we compromise to index queries with abuse
    if q_words not in query2id:
        query2id[q_words] = qidx
        qidx += 1
        documents[str(qidx)] = eval(query)
print("The query number is {}.".format(len(query2id)))

# crops too long reviews
documents = {key: value[:1000] for key, value in documents.items()}

# --------------------------- Model Training --------------------------- #
tagged_docs = [doc2vec.TaggedDocument(
    words=doc, tags=[key]) for key, doc in documents.items()]

alpha_val, min_alpha_val, passes = 0.025, 1e-4, 40
alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)
model = doc2vec.Doc2Vec(
    min_count=2,
    workers=4,
    epochs=10,
    vector_size=args.doc2vec_size,
    window=args.window_size)

model.build_vocab(tagged_docs)  # building vocabulary
for epoch in tqdm(range(passes),
                  desc='iter train',
                  total=passes,
                  ncols=117, unit_scale=True):
    random.shuffle(tagged_docs)
    model.alpha = alpha_val
    model.train(tagged_docs, total_examples=len(tagged_docs), epochs=model.epochs)
    alpha_val -= alpha_delta

# --------------------------- Save to Disk --------------------------- #
model.save(os.path.join(dset_path, 'doc2vec'))
json.dump(query2id, open(os.path.join(dset_path, 'query2id.json'), 'w'))
