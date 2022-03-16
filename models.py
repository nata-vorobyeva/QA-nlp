"""
Top-level model classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RUBertForQA(nn.Module):
    def __init__(self,
                 bert,
                 dropout):
        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.fc_start = nn.Linear(embedding_dim, 1)
        self.fc_end = nn.Linear(embedding_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        output = self.bert(text)
        # embedded = [batch size, seq len, emb dim]
        embedded = self.dropout(output['last_hidden_state'])
        # embedded = [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 0, 2)

        # logit_start, logit_end = [sent len, batch size, 1]
        logit_start = self.fc_start(self.dropout(embedded))
        logit_end = self.fc_end(self.dropout(embedded))
        # logit_start, logit_end = [batch size, sent len]
        logit_start = logit_start.permute(1, 0, 2).squeeze(-1)
        logit_end = logit_end.permute(1, 0, 2).squeeze(-1)

        # log_probs_start, log_probs_end = [batch size, sent len]
        log_probs_start = F.log_softmax(logit_start, -1)
        log_probs_end = F.log_softmax(logit_end, -1)

        return log_probs_start, log_probs_end
