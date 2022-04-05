import torch
from torch import nn
from utils.loss import nce_loss


class Model(nn.Module):
    def __init__(self, word_num, entity_num, embedding_size, factor):
        super().__init__()
        self.word_embed     = nn.Embedding(word_num + 1, embedding_size, padding_idx=word_num)
        self.query_proj     = nn.Linear(embedding_size, embedding_size)
        self.entity_embed   = nn.Embedding(entity_num, embedding_size)
        self.factor         = factor

    def forward(self, users, items, query_words,
                mode: str,
                review_words=None,
                neg_items=None, neg_words_user=None, neg_words_item=None):
        """
        :param users: [batch, ]
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param review_words: [batch, ]
        :param neg_items: [batch, n]
        :param neg_words_user: [batch, n]
        :param neg_words_item: [batch, n]
        """
        if mode == 'output_embedding':
            item_embeddings = self.entity_embed(items)
            return item_embeddings

        user_embeddings = self.entity_embed(users)

        # embed query
        query_embeddings = self.word_embed(query_words)
        valid_len = torch.count_nonzero(
            torch.count_nonzero(query_embeddings, dim=-1), dim=-1)  # padded words are always zero
        valid_len = valid_len.unsqueeze(dim=-1)
        query_embeddings = torch.sum(query_embeddings, dim=1) / (valid_len + 1e-6)
        query_embeddings = torch.tanh(self.query_proj(query_embeddings))

        personalized_model  = self.factor * query_embeddings + (1 - self.factor) * user_embeddings

        if mode == 'test':
            return personalized_model

        if mode == 'train':
            item_embeddings     = self.entity_embed(items)
            neg_item_embeddings = self.entity_embed(neg_items)

            word_embeddings     = self.word_embed(review_words)
            u_word_embeddings   = self.word_embed(neg_words_user)
            i_word_embeddings   = self.word_embed(neg_words_item)

            user_word_loss      = nce_loss(user_embeddings,
                                           word_embeddings, u_word_embeddings)
            item_word_loss      = nce_loss(item_embeddings,
                                           word_embeddings, i_word_embeddings)
            search_loss         = nce_loss(personalized_model,
                                           item_embeddings, neg_item_embeddings)

            return user_word_loss + item_word_loss + search_loss
