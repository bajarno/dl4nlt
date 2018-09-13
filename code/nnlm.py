import torch
import torch.nn as nn
import torch.nn.functional as F

class NNLM(nn.Module):
    """
    Implementation of NNLM (Bengio 2003)
    """
    def __init__(self, seq_len, vocab_size, embedding_dim, hidden_size, output_size, activation=torch.tanh):
        super(NNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.activation = activation

        # hidden layers
        # hidden_size is either integer (for one hidden layer) or list of integers (for multiple).
        emb_cat_size = embedding_dim * seq_len
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
 
        self.fc_hidden = []
        in_features = emb_cat_size
        for i, hs in enumerate(hidden_size):
            l = nn.Linear(in_features, hs)
            self.fc_hidden.append(l)
            self.add_module('fc{}'.format(i), l)
            in_features = hs

        # output layer
        self.fc_out = nn.Linear(hidden_size[-1], output_size)

    
    def forward(self, x):
        # embed [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)
        # cat embeddings [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len * embedding_dim]
        x = torch.reshape(x, (x.size(0), -1))
        # hidden layers [b, sl * emb] -> [b, hidden]
        for fc in self.fc_hidden:
            x = self.activation(fc(x))
        # [b, hidden] -> [b, out]
        out = self.fc_out(x) 
        return out