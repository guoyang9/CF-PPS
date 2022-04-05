import numpy as np

import torch
from AmazonDataset import AmazonDataset
from utils.metrics import hit, mrr, ndcg


def chunk_test(start, end, interval=512):
    count   = start
    all_ids = []
    while count < end:
        chunk_ids = [count + i for i in range(interval) if count + i < end]
        count += len(chunk_ids)
        all_ids.append(torch.tensor(chunk_ids, dtype=torch.long).cuda())
    return all_ids


def eval_candidates(model, test_dataset, test_loader, top_k, num_cands):
    """ Evaluate on num_cands items. """
    with torch.no_grad():
        Hr, Mrr, Ndcg = [], [], []

        for _, (user, item, query, item_hist, mask_hist) in enumerate(test_loader):
            assert len(user) == num_cands and all(user == user[0])
            item        = item.cuda()
            query       = query.cuda()
            item_hist   = item_hist.cuda()
            mask_hist   = mask_hist.cuda()
            item_embed  = model(None, None, item, None, mode='output_embedding')
            pred        = model(item_hist, mask_hist, None, query, mode='test')
            scores      = (pred * item_embed).sum(dim=-1)

            _, indices  = scores.topk(top_k, largest=True)
            indices     = indices.cpu().numpy().tolist()

            # we test with abuse use of ground-truth index
            Hr.append(hit(0, indices))
            Mrr.append(mrr(0, indices))
            Ndcg.append(ndcg(0, indices))

        return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)


def evaluate(model, test_dataset: AmazonDataset, progress, top_k):
    Hr, Mrr, Ndcg       = [], [], []
    chunk_items_ids     = chunk_test(0, len(test_dataset.item_map))
    chunk_items_embed   = []
    for items in chunk_items_ids:
        chunk_items_embed.append(model(None, None, items, None, mode='output_embedding'))

    user_buy = test_dataset.user_buy

    for _, (user, item, query, item_hist, mask_hist) in enumerate(progress):
        assert len(user) == len(item) == len(query) == 1
        item        = item.item()
        query       = query.cuda()
        item_hist   = item_hist.cuda()
        mask_hist   = mask_hist.cuda()

        # ---------rank all--------- #
        pred = model(item_hist, mask_hist, None, query, 'test')

        scores = []
        for item_embeds in chunk_items_embed:
            scores.append(torch.sum(pred * item_embeds, dim=-1))
        scores = torch.cat(scores)

        _, ranking_list = scores.sort(descending=False)
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
