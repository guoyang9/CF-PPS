import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp
import graph_op as graph_op
from utils.loss import triplet_match_loss, ncf_bce_loss, nce_loss


class VanillaSearch(nn.Module):
    def __init__(self, word_num, item_num, embedding_size, head_num):
        super().__init__()
        self.item_embed     = nn.Embedding(item_num, embedding_size)
        self.word_embed     = nn.Embedding(word_num + 1,
                                           embedding_size, padding_idx=word_num)
        self.word_encode    = nn.MultiheadAttention(embedding_size,
                                                    head_num, batch_first=True)

    def sent_encode(self, sentence):
        sent_embed  = self.word_embed(sentence)
        sent_encode = self.word_encode(sent_embed, sent_embed, sent_embed)[0]
        sent_encode = sent_encode.mean(dim=1)
        return sent_encode

    def forward(self, users, items, query_words,
                review_words=None,
                mode='train', neg_items=None):
        """
        :param users: [batch, ]
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param review_words: [batch, num_sent_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param neg_items: [batch, n]
        """
        if mode == 'output_embedding':
            item_embed = self.item_embed(items)
            return item_embed

        query_encode = self.sent_encode(query_words)

        if mode == 'test':
            return query_encode

        if mode == 'train':
            # items
            item_embed      = self.item_embed(items)
            neg_item_embed  = self.item_embed(neg_items)

            search_loss = nce_loss(query_encode, item_embed, neg_item_embed)
            return search_loss


class MFSearch(nn.Module):
    def __init__(self, word_num, entity_num, embedding_size, head_num):
        super().__init__()
        self.entity_embed   = nn.Embedding(entity_num, embedding_size)
        self.word_embed     = nn.Embedding(word_num + 1,
                                           embedding_size, padding_idx=word_num)
        self.word_encode    = nn.MultiheadAttention(embedding_size,
                                                    head_num, batch_first=True)

    def sent_encode(self, sentence):
        sent_embed  = self.word_embed(sentence)
        sent_encode = self.word_encode(sent_embed, sent_embed, sent_embed)[0]
        sent_encode = sent_encode.mean(dim=1)
        return sent_encode

    def forward(self, users, items, query_words,
                review_words=None,
                mode='train', neg_items=None):
        """
        :param users: [batch, ]
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param review_words: [batch, num_sent_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param neg_items: [batch, n]
        """
        if mode == 'output_embedding':
            item_embed = self.entity_embed(items)
            return item_embed

        query_encode    = self.sent_encode(query_words)
        user_embed      = self.entity_embed(users)
        personalized    = query_encode + 0.1 * user_embed

        if mode == 'test':
            return personalized

        if mode == 'train':
            # items
            item_embed      = self.entity_embed(items)
            neg_item_embed  = self.entity_embed(neg_items)

            cf_loss     = triplet_match_loss(user_embed, item_embed, neg_item_embed)
            search_loss = nce_loss(personalized, item_embed, neg_item_embed)
            return search_loss + cf_loss


class NCFSearch(MFSearch):
    def __init__(self, word_num, entity_num, embedding_size, head_num,
                 num_layers=3, dropout=0.5):
        super().__init__(word_num, entity_num, embedding_size, head_num)
        self.MLP_embed  = nn.Embedding(entity_num,
                                       embedding_size * (2 ** (num_layers - 1)))

        MLP_modules     = []
        for i in range(num_layers):
            input_size = embedding_size * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # for ncf only
        self.prediction = nn.Linear(embedding_size * 2, 1)

    def forward(self, users, items, query_words,
                review_words=None,
                mode='train', neg_items=None):
        """
        :param users: [batch, ]
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param review_words: [batch, num_sent_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param neg_items: [batch, n]
        """
        if mode == 'output_embedding':
            item_embed_gmf  = self.entity_embed(items)
            return item_embed_gmf

        query_encode    = self.sent_encode(query_words)
        user_embed_gmf  = self.entity_embed(users)
        user_embed_mlp  = self.MLP_embed(users)
        personalized    = query_encode + 0.1 * user_embed_gmf

        if mode == 'test':
            return personalized

        if mode == 'train':
            # items
            item_embed_gmf      = self.entity_embed(items)
            item_embed_mlp      = self.MLP_embed(items)

            neg_item_embed_gmf  = self.entity_embed(neg_items)
            neg_item_embed_mlp  = self.MLP_embed(neg_items)

            # where ncf begins
            fusion_gmf      = user_embed_gmf * item_embed_gmf
            fusion_mlp      = self.MLP_layers(torch.cat([user_embed_mlp, item_embed_mlp], dim=-1))
            neg_fusion_gmf  = user_embed_gmf.unsqueeze(dim=1) * neg_item_embed_gmf
            neg_fusion_mlp  = self.MLP_layers((torch.cat(
                            [torch.stack([user_embed_mlp for _ in range(neg_items.size()[1])], dim=1),
                             neg_item_embed_mlp], dim=-1)))

            fusion          = self.prediction(torch.cat([fusion_gmf, fusion_mlp], dim=-1))
            neg_fusion      = self.prediction(torch.cat([neg_fusion_gmf, neg_fusion_mlp], dim=-1))

            cf_loss     = ncf_bce_loss(fusion, neg_fusion)
            search_loss = nce_loss(personalized, item_embed_gmf, neg_item_embed_gmf)
            return search_loss + cf_loss


class GraphSearch(MFSearch):
    def __init__(self, adj_mat,
                 word_num, entity_num,
                 embedding_size, head_num, conv_num):
        super().__init__(word_num, entity_num, embedding_size, head_num)
        self.conv_num   = conv_num
        self.sym_mat    = self.init_graph(adj_mat).cuda()

    def init_graph(self, adj_mat):
        adj_mat = sp.coo_matrix(adj_mat)
        deg_mat = graph_op.deg_est(adj_mat)
        sym_mat = deg_mat.dot(adj_mat).dot(deg_mat)
        return graph_op.tensor_from_coo(sp.coo_matrix(sym_mat))

    def graph_update(self):
        embed_layers = {'layer_0': self.entity_embed.weight}
        for layer in range(self.conv_num):
            embed_layers['layer_%d' % (layer + 1)] = torch.matmul(
                self.sym_mat, embed_layers['layer_%d' % layer])
        embed_mat   = torch.stack(list(embed_layers.values()), dim=1)
        self.graph  = torch.mean(embed_mat, dim=1)

    def forward(self, users, items, query_words,
                review_words=None,
                mode='train', neg_items=None):
        """
        :param users: [batch, ]
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param review_words: [batch, num_sent_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param neg_items: [batch, n]
        """
        if mode == 'train':
            self.graph_update()
        if mode == 'output_embedding':
            item_embed = self.graph[items]
            return item_embed

        query_encode    = self.sent_encode(query_words)
        user_embed      = self.graph[users]
        personalized    = query_encode + 0.1 * user_embed
        if mode == 'test':
            return personalized

        if mode == 'train':
            # items
            item_embed      = self.graph[items]
            neg_item_embed  = self.graph[neg_items]

            cf_loss     = triplet_match_loss(user_embed, item_embed, neg_item_embed)
            search_loss = nce_loss(personalized, item_embed, neg_item_embed)
            return search_loss + cf_loss
