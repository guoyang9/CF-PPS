import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vision_size, text_size,
                 embed_size, user_size,
                 mode, dropout, is_training):
        super().__init__()
        """
        Important Args:
        :param vision_size: for end_to_end is 4096, otherwise not
        :param text_size: for end_to_end is 512, otherwise not
        :param mode: choices ['end-to-end', 'vision', 'text', 'fine-tune']
        """
        self.mode = mode
        self.is_training = is_training

        # vision fully connected layers
        def vision_fc():
            return nn.Sequential(
                nn.Linear(vision_size, embed_size),
                nn.ELU(),
                nn.Dropout(p=dropout),
                nn.Linear(embed_size, embed_size),
                nn.ELU()
            )

        # text fully connected layers
        def text_fc():
            return nn.Sequential(
                nn.Linear(text_size, embed_size),
                nn.ELU(),
                nn.Dropout(p=dropout),
                nn.Linear(embed_size, embed_size),
                nn.ELU()
            )

        assert mode in ['end-to-end', 'vision', 'text', 'fine-tune']
        if not mode == 'text':
            self.vision_fc  = vision_fc()
        if not mode == 'vision':
            self.text_fc    = text_fc()

        # user and query embedding
        self.user_embed     = nn.Embedding(user_size, embed_size)
        self.query_embed    = nn.Sequential(
            nn.Linear(text_size, embed_size),
            nn.ELU()
        )

        # for embed user and item in the same space
        self.translate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ELU()
        )

        # item fully connected layers
        if self.mode in ['end-to-end', 'fine-tune']:
            self.item_fc = nn.Sequential(
                nn.Linear(2 * embed_size, embed_size),
                # nn.ELU(),
                # nn.Dropout(p=dropout),
                # nn.Linear(embed_size, embed_size),
                nn.ELU()
            )
        else:
            self.item_fc = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ELU(),
                nn.Linear(embed_size, embed_size),
                nn.ELU()
            )

    def forward(self, user, query,
                pos_text, neg_text,
                test_first=False,
                pos_vision=None, neg_vision=None):

        def encode_item(vision, text):
            if self.mode == 'vis':
                vision  = self.vision_fc(vision)
                fusion  = vision
            elif self.mode == 'text':
                text    = self.text_fc(text)
                fusion  = text
            else:
                vision  = self.vision_fc(vision)
                text    = self.text_fc(text)
                fusion  = torch.cat((vision, text), dim=-1)
            item = self.item_fc(fusion)
            item = self.translate(item)
            return item

        def encode_query(user, query):
            user    = F.elu(self.user_embed(user))
            user    = self.translate(user)
            query   = self.translate(self.query_embed(query))
            pred    = user + query
            return pred

        if self.is_training:
            pred_item   = encode_query(user, query)
            pos_item    = encode_item(pos_vision, pos_text)
            neg_items   = encode_item(neg_vision, neg_text)
            return pred_item, pos_item, neg_items
        else:
            if test_first:
                return encode_item(pos_vision, pos_text)
            else:
                return encode_query(user, query)
