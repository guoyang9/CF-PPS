import os
import json
import numpy as np
from argparse import ArgumentParser
from gensim.models.doc2vec import Doc2Vec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Model import Model
from AmazonDataset import AmazonDataset
from evaluate import evaluate, eval_candidates
from params import parser_add_data_arguments
from utils.metrics import display
from utils.loss import triplet_loss
from utils.data_preparation import data_preparation
from utils import training_progress, testing_progress


def run(args):
    dset_path       = os.path.join(args.processed_path, args.dataset)
    query2id        = json.load(open(os.path.join(dset_path, 'query2id.json'), 'r'))
    doc2vec_model   = Doc2Vec.load(os.path.join(dset_path, 'doc2vec'))

    train_df, test_df, full_df, word_dict = data_preparation(args)
    users, item_map = AmazonDataset.init(full_df)

    # make numpy ndarray in doc2vec writeable
    doc2vecs    = dict()
    keys        = list(item_map.keys()) + list(query2id.values())
    for key in keys:
        vectors = doc2vec_model.dv[key]
        vectors.flags.writeable = True
        doc2vecs[key] = vectors

    # --------------------------------------Data Loaders-------------------------------------- #
    train_dataset   = AmazonDataset(train_df, len(users), item_map,
                                    query2id, doc2vecs,
                                    args.debug, is_training=True)
    test_dataset    = AmazonDataset(test_df, len(users), item_map,
                                    query2id, doc2vecs,
                                    args.debug, is_training=False,
                                    user_buy=train_dataset.user_buy)

    train_loader    = DataLoader(train_dataset, drop_last=False, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.worker_num)
    # test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=args.candidate,
    #                              shuffle=False, num_workers=0)
    test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=1,
                                 shuffle=False, num_workers=0)

    # -----------------------------------Model Construction----------------------------------- #
    model = Model(0, args.text_size, args.embedding_size, len(users), args.mode, args.dropout, True)
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # ------------------------------------Train------------------------------------ #
    loss = 0

    for epoch in range(args.epochs):
        model.train()
        model.is_training = True

        epoch_loss = step = 0

        train_dataset.item_sampling(args.neg_sample_num, start=0, end=len(item_map))
        progress = training_progress(train_loader, epoch, args.epochs, loss, args.debug)

        for _, (users, items, queries, items_neg) in enumerate(progress):
            users, items, queries, items_neg, = \
                users.cuda(), items.cuda(), queries.cuda(), items_neg.cuda()

            model.zero_grad()

            pred, positive, negative = model(users, queries, items, items_neg)
            loss = triplet_loss(pred, positive, negative)
            if args.debug:
                progress.set_postfix({"loss": "{:.3f}".format(float(loss))})
            epoch_loss += loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optim.step()
            step += 1

        model.eval()
        model.is_training = False
        # Hr, Mrr, Ndcg = eval_candidates(model, test_dataset,
        #                                 testing_progress(test_loader, epoch, args.epochs, args.debug),
        #                                 args.top_k, args.candidate)
        Hr, Mrr, Ndcg = evaluate(model, test_dataset,
                                 testing_progress(test_loader, epoch, args.epochs, args.debug),
                                 args.top_k)
        display(epoch, args.epochs, epoch_loss / step, Hr, Mrr, Ndcg)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser_add_data_arguments(parser)

    # ------------------------------------Experiment Setup------------------------------------ #
    parser.add_argument('--mode',
                        default='text',
                        type=str,
                        help='the model mode')
    parser.add_argument('--text_size',
                        default=512,
                        type=int)
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="the dropout rate")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run(args)
