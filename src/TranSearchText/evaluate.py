import numpy as np

import torch
import torch.nn.functional as F
from AmazonDataset import AmazonDataset
from utils.metrics import hit, mrr, ndcg


def chunk_test(dataset, interval=512):
    count, end = 0, len(dataset.index2item)
    chunk_embeds = []

    while count < end:
        chunk_ids   = [dataset.index2item[count + i]
                       for i in range(interval) if count + i < end]
        embeds      = np.array([dataset.doc2vecs[item] for item in chunk_ids])
        count += len(chunk_ids)
        chunk_embeds.append(torch.tensor(embeds).cuda())
    return chunk_embeds


def eval_candidates(model, test_dataset, test_loader, top_k, num_cands):
    """ Evaluate on num_cands items. """
    with torch.no_grad():
        Hr, Mrr, Ndcg = [], [], []

        for _, (user, item, query) in enumerate(test_loader):
            assert len(user) == num_cands and all(user == user[0])
            user        = user.cuda()
            item        = item.cuda()
            query       = query.cuda()
            item_embed  = model(None, None, item, None, test_first=True)
            item_pred   = model(user, query, None, None, test_first=False)

            scores      = F.pairwise_distance(item_embed, item_pred)

            _, indices  = scores.topk(top_k, largest=False)
            indices     = indices.cpu().numpy().tolist()

            # we test with abuse use of ground-truth index
            Hr.append(hit(0, indices))
            Mrr.append(mrr(0, indices))
            Ndcg.append(ndcg(0, indices))
        return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)


def evaluate(model, test_dataset: AmazonDataset, progress, top_k):
    Hr, Mrr, Ndcg       = [], [], []
    chunk_embeds        = chunk_test(test_dataset)
    chunk_items_embed   = []
    for items in chunk_embeds:
        chunk_items_embed.append(model(None, None, items, None, test_first=True))

    user_buy = test_dataset.user_buy

    for _, (user, item, query) in enumerate(progress):
        assert len(user) == len(item) == len(query) == 1
        item    = item.item()
        user    = user.cuda()
        query   = query.cuda()

        # ---------rank all--------- #
        pred = model(user, query, None, None, test_first=False)

        scores = []
        for item_embeds in chunk_items_embed:
            scores.append(F.pairwise_distance(item_embeds, pred))
        scores = torch.cat(scores)

        _, ranking_list = scores.sort(descending=True)
        ranking_list    = ranking_list.tolist()

        bought      = user_buy[user.item()]
        return_list = []
        while len(return_list) < top_k:
            if len(ranking_list) == 0:
                break

            candidate_item = ranking_list.pop()
            if candidate_item not in bought or candidate_item == item:
                return_list.append(candidate_item)

        Hr.append(hit(item, return_list))
        Mrr.append(mrr(item, return_list))
        Ndcg.append(ndcg(item, return_list))

    return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)
