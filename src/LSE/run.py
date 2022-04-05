import os
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader

from Model import Model
from AmazonDataset import AmazonDataset, pad_collate
from evaluate import evaluate, eval_candidates
from params import parser_add_data_arguments
from utils.metrics import display
from utils import training_progress, testing_progress
from utils.data_preparation import data_preparation


def run():
    parser = ArgumentParser()
    parser_add_data_arguments(parser)

    # ------------------------------------Experiment Setup------------------------------------ #
    parser.add_argument('--window_size',
                        default=9,
                        help='n-gram, should be odd')

    # ------------------------------------Data Preparation------------------------------------ #
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_df, test_df, full_df, word_dict = data_preparation(args)

    users, item_map, query_max_length = AmazonDataset.init(full_df)
    query_max_length = min(query_max_length, args.max_query_len)

    # --------------------------------------Data Loaders-------------------------------------- #
    train_dataset   = AmazonDataset(train_df, item_map, len(word_dict),
                                    query_max_length, args.max_sent_len, args.window_size,
                                    'train', args.debug)
    test_dataset    = AmazonDataset(test_df, item_map, len(word_dict),
                                    query_max_length, args.max_sent_len, args.window_size,
                                    'test', args.debug)

    train_loader    = DataLoader(train_dataset, drop_last=False, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.worker_num)
    # test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=args.candidate,
    #                              shuffle=False, num_workers=0)  # for candidate setting
    test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=1,
                                 shuffle=False, num_workers=0) # for whole set

    # -----------------------------------Model Construction----------------------------------- #
    model = Model(len(word_dict), len(item_map), args.embedding_size)
    model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)

    # ------------------------------------Train------------------------------------ #
    loss = 0

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = step = 0
        train_dataset.item_sampling(args.neg_sample_num, start=0, end=len(item_map))
        progress = training_progress(train_loader, epoch, args.epochs, loss, args.debug)

        for _, (items, words, query_words, items_neg) in enumerate(progress):
            items, words, query_words, items_neg = \
                items.cuda(), words.cuda(), query_words.cuda(), items_neg.cuda()

            model.zero_grad()
            loss = model(items, query_words, 'train', words, items_neg)
            if args.debug:
                progress.set_postfix({"loss": "{:.3f}".format(float(loss))})
            epoch_loss += loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optim.step()
            step += 1

        model.eval()
        # Hr, Mrr, Ndcg = eval_candidates(model, test_dataset,
        #                                 testing_progress(test_loader, epoch, args.epochs, args.debug),
        #                                 args.top_k, args.candidate)
        Hr, Mrr, Ndcg = evaluate(model, test_dataset,
                                 testing_progress(test_loader, epoch, args.epochs, args.debug),
                                 args.top_k)
        display(epoch, args.epochs, epoch_loss / step, Hr, Mrr, Ndcg)


if __name__ == '__main__':
    run()
