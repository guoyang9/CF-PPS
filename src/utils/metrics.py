import numpy as np


best_hr, best_mrr, best_ndcg = 0.0, 0.0, 0.0


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def mrr(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(float(index + 1))
    else:
        return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def display(epoch, epoch_num, loss, hr, mrr, ndcg):
    print(
        "Running Epoch {:03d}/{:03d}".format(epoch + 1, epoch_num),
        "loss:{:.3f}".format(float(loss)),
        "Hr {:.3f}, Mrr {:.3f}, Ndcg {:.3f}".format(hr, mrr, ndcg),
        flush=True)
    global best_hr, best_mrr, best_ndcg
    if hr >= best_hr:
        best_hr     = hr
        best_mrr    = mrr
        best_ndcg   = ndcg
    if epoch + 1 == epoch_num:
        print('-----------Best Result:-----------')
        print('Hr: {:.3f}, Mrr: {:.3f}, Ndcg: {:.3f}'.format(best_hr, best_mrr, best_ndcg))
        print('----------------------------------')
