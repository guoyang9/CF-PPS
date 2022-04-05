import torch
from torch import nn
from utils.loss import nce_loss


class Model(nn.Module):
    def __init__(self, word_num, item_num, embedding_size):
        super(Model, self).__init__()
        self.word_embed = nn.Embedding(word_num + 1, embedding_size, padding_idx=word_num)
        self.word_proj  = nn.Linear(embedding_size, embedding_size)
        self.item_embed = nn.Embedding(item_num, embedding_size)

    def fse_func(self, words):
        """
        :param words: [batch, num, embed_size]
        """
        embeddings  = self.word_embed(words)
        valid_len   = torch.count_nonzero(
            torch.count_nonzero(embeddings, dim=-1), dim=-1) # padded words are always zero
        valid_len   = valid_len.unsqueeze(dim=-1)

        embeddings  = torch.sum(embeddings, dim=1) / (valid_len + 1e-6)
        embeddings  = torch.tanh(self.word_proj(embeddings))
        return embeddings

    def forward(self, items, query_words, mode: str, review_words=None, neg_items=None):
        """
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param review_words: [batch, ]
        :param neg_items: [batch, n]
        """
        if mode == 'output_embedding':
            item_embeddings = self.item_embed(items)
            return item_embeddings

        if mode == 'test':
            query_embeddings = self.fse_func(query_words)
            return query_embeddings

        if mode == 'train':
            item_embeddings     = self.item_embed(items)
            neg_item_embeddings = self.item_embed(neg_items)
            word_embeddings     = self.fse_func(review_words)

            item_word_loss = nce_loss(word_embeddings, item_embeddings, neg_item_embeddings)
            return item_word_loss
