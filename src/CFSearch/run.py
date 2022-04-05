import os, sys
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader

from AmazonDataset import AmazonDataset
from evaluate import evaluate, eval_candidates
from Model import VanillaSearch, MFSearch, NCFSearch, GraphSearch
from params import parser_add_data_arguments
from utils.metrics import display
from utils import training_progress, testing_progress
from utils.data_preparation import data_preparation


def run(model_name: str, args):
    train_df, test_df, full_df, word_dict = data_preparation(args)
    users, item_map, query_max_length = AmazonDataset.init(full_df)
    query_max_length = min(query_max_length, args.max_query_len)

    # --------------------------------------Data Loaders-------------------------------------- #
    train_dataset   = AmazonDataset(train_df, users, item_map,
                                    len(word_dict), query_max_length, args.max_sent_len,
                                    'train', args.debug)
    test_dataset    = AmazonDataset(test_df, users, item_map,
                                    len(word_dict), query_max_length, args.max_sent_len,
                                    'test', args.debug, train_dataset.user_buy)

    train_loader    = DataLoader(train_dataset, drop_last=False, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.worker_num)
    # test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=args.candidate,
    #                              shuffle=False, num_workers=0)  # for candidate setting
    test_loader     = DataLoader(test_dataset, drop_last=False, batch_size=1,
                                 shuffle=False, num_workers=0) # for whole set

    # -----------------------------------Model Construction----------------------------------- #
    if model_name == 'VanillaSearch':
        model = VanillaSearch(len(word_dict),
                              len(item_map), args.embedding_size, args.head_num)
    elif model_name == 'MFSearch':
        model = MFSearch(len(word_dict),
                         len(users) + len(item_map), args.embedding_size, args.head_num)
    elif model_name == 'NCFSearch':
        model = NCFSearch(len(word_dict),
                          len(users) + len(item_map), args.embedding_size, args.head_num)
    elif model_name == 'GraphSearch':
        model = GraphSearch(train_dataset.adj_matrix,
                            len(word_dict), len(users) + len(item_map),
                            args.embedding_size, args.head_num, args.conv_num)
    else:
        raise NotImplementedError
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)

    # ------------------------------------Train------------------------------------ #
    loss = 0

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = step = 0

        if model_name == 'VanillaSearch':
            train_loader.dataset.non_personalized_sampling(args.neg_sample_num, start=0, end=len(item_map))
        else:
            train_loader.dataset.item_sampling(args.neg_sample_num, start=0, end=len(item_map))
        progress = training_progress(train_loader, epoch, args.epochs, loss, args.debug)

        for _, (users, items, query_words, words, items_neg) in enumerate(progress):
            users, items, query_words, words, items_neg = \
                users.cuda(), items.cuda(), query_words.cuda(), words.cuda(), items_neg.cuda()

            model.zero_grad()
            loss = model(users, items, query_words, words, 'train', items_neg)
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
    parser = ArgumentParser()
    parser_add_data_arguments(parser)

    # ------------------------------------Experiment Setup------------------------------------ #
    parser.add_argument('--head_num',
                        default=4,
                        type=int,
                        help='the number of heads used in multi-head self-attention layer')
    parser.add_argument('--conv_num',
                        default=2,
                        type=int,
                        help='the number of convolution layers')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.model in ['VanillaSearch', 'MFSearch', 'NCFSearch', 'GraphSearch']:
        run(args.model, args)
    else:
        run('GraphSearch', args)
