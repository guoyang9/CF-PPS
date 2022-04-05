import numpy as np

import torch
from utils.metrics import hit, mrr, ndcg


def chunk_test(start, end, interval=512):
    """ For GPU memory reasons only. """
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

        for _, (item, query) in enumerate(test_loader):
            assert len(item) == num_cands
            item        = item.cuda()
            query       = query.cuda()
            item_embed  = model(item, None, mode='output_embedding')
            pred        = model(None, query, mode='test')
            scores      = (pred * item_embed).sum(dim=-1)

            _, indices  = scores.topk(top_k, largest=True)
            indices     = indices.cpu().numpy().tolist()

            # we test with abuse use of ground-truth index
            Hr.append(hit(0, indices))
            Mrr.append(mrr(0, indices))
            Ndcg.append(ndcg(0, indices))

        return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)


def evaluate(model, test_dataset, test_loader, top_k):
    with torch.no_grad():
        Hr, Mrr, Ndcg       = [], [], []
        chunk_items_ids     = chunk_test(0, len(test_dataset.item_map))
        chunk_items_embed   = []
        for items in chunk_items_ids:
            chunk_items_embed.append(model(items, None, mode='output_embedding'))

        for _, (item, query) in enumerate(test_loader):
            assert len(item) == len(query) == 1
            item    = item.item()
            query   = query.cuda()
            # ---------rank all--------- #
            pred    = model(None, query, 'test')

            scores  = []
            for item_embeds in chunk_items_embed:
                scores.append(torch.sum(pred * item_embeds, dim=-1))
            scores = torch.cat(scores)

            _, ranking_list = scores.topk(top_k, largest=True)
            return_list     = ranking_list.cpu().numpy().tolist()

            Hr.append(hit(item, return_list))
            Mrr.append(mrr(item, return_list))
            Ndcg.append(ndcg(item, return_list))

    return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)
