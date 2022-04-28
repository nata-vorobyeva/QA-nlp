"""
Top-level model classes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    

class Retriever(HashingTfIdfVectorizer):
    """

    Class is inherited from HashingTfIdfVectorizer by DeepPavlov team
    (http://docs.deeppavlov.ai/)

    """
    def retrieve(self, query, n):   
        """
        Retrieve documents most relevant to a query.
        Documents are stored as tf-idf vectors in .npz format
        Cosine similarity is used as similarity measure between query and document vectors

        Args:
            query (list): Input strings. Lenght [number of queries]
            n (int): number of retrieved documents for each query

        Returns:
            retrieved_doc_id (list): retrieved documents id with lenght [number of queries * n].
            Query with index i gets doc_id with indexes from `i*n` to `(i+1)*n-1` inclusive.
            similarity (ndarray): cosine similarity matrix between query and document vectors
            with shape [total number of documents in .npz file, number of queries]

        """

        # get tf-idf vectors for queries
        query_matrix = self.__call__(query)
        # load tf-idf matrix with documents and document ids
        matrix, csr_data = self.load()
        doc_index = csr_data['doc_index']
        doc_id = np.fromiter(map(int, doc_index.keys()), dtype=np.int32)
        context_matrix = matrix.T
        # compute cosine similarity between each query and each document
        similarity = cosine_similarity(context_matrix, query_matrix, dense_output=True)
        # select n most relevant documents:
        # first find indexes of this documents for each query in similarity matrix
        retrieved_doc_index = np.argpartition(similarity, -n, axis=0)[-n:].T
        retrieved_doc_id = []
        # now transform indexes into document ids
        for index in retrieved_doc_index:
            retrieved_doc_id += list(doc_id[index])
        
        return retrieved_doc_id, similarity
