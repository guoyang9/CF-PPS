import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import nce_loss


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, head_num, model_name: str):
        super().__init__()
        self.input_dim  = input_dim
        self.head_num   = head_num
        self.model_name = model_name

        self.query_proj = nn.Linear(input_dim, input_dim * head_num)
        self.reduction  = nn.Linear(head_num, 1, bias=False)

    def attention_function(self, query_embedding, hist_embeddings):
        """
        :param query_embedding: [batch, input_dim]
        :param hist_embeddings: [batch, n, input_dim]
        :return: scores: [batch, n, 1]
        """
        projected_query = torch.tanh(self.query_proj(query_embedding))
        projected_query = projected_query.view((-1, 1, self.head_num, self.input_dim))

        hist_embeddings = hist_embeddings.unsqueeze(dim=2)
        item_query_dot  = (projected_query * hist_embeddings).sum(dim=-1) # [b, n, hn]

        scores = self.reduction(item_query_dot)
        return scores

    def forward(self, hist_embeddings, mask_hist, query_embedding):
        """
        :param hist_embeddings: [batch, hist_len, input_dim]
        :param mask_hist: [batch, hist_len]
        :param query_embedding: [batch, input_dim]
        """
        mask_hist   = mask_hist.unsqueeze(dim=-1)
        attn_scores = self.attention_function(query_embedding, hist_embeddings)
        attn_scores += mask_hist

        if self.model_name == 'AEM':
            weight = F.softmax(attn_scores, dim=1)
        elif self.model_name == 'ZAM':
            batch_size  = hist_embeddings.size()[0]
            zero_embed  = torch.zeros((batch_size, 1, self.input_dim),
                                      dtype=torch.long, device=hist_embeddings.device)
            zero_score  = self.attention_function(query_embedding, zero_embed)
            weight      = attn_scores.exp() / (attn_scores.exp() +
                                               zero_score.exp()).sum(dim=1, keepdim=True)
        else:
            raise NotImplementedError

        user_embeddings = torch.sum(weight * hist_embeddings, dim=1)
        return user_embeddings


class AEM(nn.Module):
    def __init__(self, word_num, item_num, embedding_size, attention_hidden_dim):
        super().__init__()
        self.embed_size     = embedding_size
        self.word_embed     = nn.Embedding(word_num + 1, embedding_size, padding_idx=word_num)
        self.query_proj     = nn.Linear(embedding_size, embedding_size)
        self.item_embed     = nn.Embedding(item_num + 1, embedding_size, padding_idx=item_num)

        self.attn_layer     = AttentionLayer(embedding_size, attention_hidden_dim, self.__class__.__name__)

    def forward(self, items_hist, mask_hist,
                items, query_words,
                mode: str,
                review_words=None,
                neg_items=None, neg_review_words=None):
        """
        :param items_hist: [batch, hist_len]
        :param mask_hist: [batch, hist_len]
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param review_words: [batch,]
        :param neg_items: [batch, n]
        :param neg_review_words: [batch, n]
        """
        if mode == 'output_embedding':
            item_embeddings = self.item_embed(items)
            return item_embeddings

        hist_embeddings = self.item_embed(items_hist)

        # embed query
        query_embeddings = self.word_embed(query_words)
        valid_len = torch.count_nonzero(
            torch.count_nonzero(query_embeddings, dim=-1), dim=-1)  # padded words are always zero
        valid_len = valid_len.unsqueeze(dim=-1)
        query_embeddings = torch.sum(query_embeddings, dim=1) / (valid_len + 1e-6)
        query_embeddings = torch.tanh(self.query_proj(query_embeddings))

        user_embeddings     = self.attn_layer(hist_embeddings, mask_hist, query_embeddings)
        personalized_model  = (query_embeddings + user_embeddings) / 2

        if mode == 'test':
            return personalized_model

        elif mode == 'train':
            item_embeddings     = self.item_embed(items)
            neg_item_embeddings = self.item_embed(neg_items)

            word_embeddings     = self.word_embed(review_words)
            neg_word_embeddings = self.word_embed(neg_review_words)

            item_word_loss      = nce_loss(item_embeddings,
                                           word_embeddings, neg_word_embeddings)
            search_loss         = nce_loss(personalized_model,
                                           item_embeddings, neg_item_embeddings)

            return item_word_loss + search_loss
        else:
            raise NotImplementedError


class ZAM(AEM):
    def __init__(self, word_num, item_num, embedding_size, attention_hidden_dim):
        super().__init__(word_num, item_num, embedding_size, attention_hidden_dim)
