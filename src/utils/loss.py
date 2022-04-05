import torch
import torch.nn.functional as F


def nce_loss(anchor, positive, negatives):
    """
    :param anchor: query embed - [batch, embed_size]
    :param positive: [batch, embed_size]
    :param negatives: [batch, k, embed_size]
    """
    pos = torch.sum(anchor * positive, dim=-1).sigmoid().log()
    neg = (- torch.sum(anchor.unsqueeze(dim=-2) * negatives, dim=-1)).sigmoid().log()
    neg = neg.sum(dim=-1)
    return - (pos + neg).mean()


def triplet_loss(anchor, positive, negatives):
    """
    We found that add all the negative ones together can
    yeild relatively better performance.
    :param anchor: user, [batch, embed]
    :param positive: item, [batch, embed]
    :param negatives: item_negs, [batch, num, embed]
    """
    negatives = negatives.permute(1, 0, 2)

    losses = torch.tensor(0., device=anchor.device)
    for negative in negatives:
        losses += torch.mean(
            F.triplet_margin_loss(anchor, positive, negative))
    return losses / len(negatives)


def triplet_match_loss(anchor, positive, negatives):
    """
    :param anchor: user, [batch, embed]
    :param positive: item, [batch, embed]
    :param negatives: item_negs, [batch, num, embed]
    """
    anchor      = anchor / anchor.norm(dim=-1, keepdim=True)
    positive    = positive / positive.norm(dim=-1, keepdim=True)
    negatives   = negatives / negatives.norm(dim=-1, keepdim=True)

    pos_scores  = torch.sum(anchor * positive, dim=-1)
    neg_scores  = torch.sum(anchor.unsqueeze(1) * negatives, dim=-1)

    pos_targets = torch.ones_like(pos_scores, device=pos_scores.device)
    neg_targets = torch.zeros_like(neg_scores, device=neg_scores.device)

    loss        = F.mse_loss(pos_scores, pos_targets) + F.mse_loss(neg_scores, neg_targets)
    return loss


def ncf_bce_loss(positive, negatives):
    """
    :param positive: scores, [batch, 1]
    :param negatives: scores, [batch, num, 1]
    """
    positive    = positive.squeeze(-1)
    negatives   = negatives.squeeze(-1)

    pos_targets = torch.ones_like(positive, device=positive.device)
    neg_targets = torch.zeros_like(negatives, device=negatives.device)

    loss        = F.binary_cross_entropy_with_logits(positive, pos_targets) + \
                  F.binary_cross_entropy_with_logits(negatives, neg_targets)
    return loss
