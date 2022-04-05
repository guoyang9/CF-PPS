import os
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader

from Model import Model
from AmazonDataset import AmazonDataset
from evaluate import evaluate, eval_candidates
from params import parser_add_data_arguments
from utils.metrics import display
from utils import training_progress, testing_progress
from utils.data_preparation import data_preparation


def run(args):
    train_df, test_df, full_df, word_dict = data_preparation(args)

    users, item_map, query_max_length = AmazonDataset.init(full_df)
    query_max_length = min(query_max_length, args.max_query_len)

    # --------------------------------------Data Loaders-------------------------------------- #
    train_dataset   = AmazonDataset(train_df, users, item_map,
                                    len(word_dict), query_max_length, args.max_sent_len,
                                    args.sub_sampling_rate,
                                    'train', args.debug)
    test_dataset    = AmazonDataset(test_df, users, item_map,
                                    len(word_dict), query_max_length, args.max_sent_len,
                                    args.sub_sampling_rate,
                                    'test', args.debug, train_dataset.user_buy)

    train_loader    = DataLoader(train_dataset, drop_last=False, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.worker_num)
    # test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=args.candidate,
    #                              shuffle=False, num_workers=0)  # for candidate setting
    test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=1,
                                 shuffle=False, num_workers=0) # for whole set

    # -----------------------------------Model Construction----------------------------------- #
    item_num, user_num = len(item_map), len(users)
    model = Model(len(word_dict), user_num + item_num, args.embedding_size, args.factor)
    model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)

    # ------------------------------------Train------------------------------------ #
    loss = 0

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = step = 0
        train_dataset.item_sampling(args.neg_sample_num, start=0, end=item_num)
        train_dataset.word_sampling(train_dataset.uw_matrix, 'user',
                                    args.neg_sample_num, start=0, end=len(word_dict))
        train_dataset.word_sampling(train_dataset.ui_matrix, 'item',
                                    args.neg_sample_num, start=0, end=len(word_dict))
        progress = training_progress(train_loader, epoch, args.epochs, loss, args.debug)

        for _, (users, items, query_words, words,
                items_neg, words_neg_user, words_neg_item) in enumerate(progress):
            users, items, query_words, words, items_neg, words_neg_user, words_neg_item = \
                users.cuda(), items.cuda(), query_words.cuda(), words.cuda(), \
                items_neg.cuda(), words_neg_user.cuda(), words_neg_item.cuda()

            model.zero_grad()
            loss = model(users, items, query_words, 'train', words,
                         items_neg, words_neg_user, words_neg_item)
            if args.debug:
                progress.set_postfix({"loss": "{:.3f}".format(float(loss))})
            epoch_loss += loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optim.step()
            step += 1

        model.eval()
        # Hr, Mrr, Ndcg = eval_candidates(model, test_dataset,
        #                          testing_progress(test_loader, epoch, args.epochs, args.debug),
        #                          args.top_k, args.candidate)
        Hr, Mrr, Ndcg = evaluate(model, test_dataset,
                                 testing_progress(test_loader, epoch, args.epochs, args.debug),
                                 args.top_k)
        display(epoch, args.epochs, epoch_loss / step, Hr, Mrr, Ndcg)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser_add_data_arguments(parser)

    # ------------------------------------Experiment Setup------------------------------------ #
    parser.add_argument('--sub_sampling_rate',
                        default=1e4,
                        type=float,
                        help='sub-sampling rate')
    parser.add_argument('--factor',
                        default=0.5,
                        type=float,
                        help='the query weight of personalized model')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run(args)
