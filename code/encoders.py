import torch
import torch.nn as nn
import torch.nn.functional as F

class BOWEncoder(nn.Module):
    """
    Bag-of-Words encoder
    From "A Neural Attention Model for Abstractive Sentence Summarization" (Rush et al. 2015)

    TODO: convert EmbeddingBag to nn.Embedding and add padding functionality
    """
    def __init__(self, vocab_size, embedding_size, output_size):
        super(BOWEncoder, self).__init__()
        self.fc_out = nn.Linear(embedding_size, output_size)

    def forward(self, x, y_c, x_lengths, *_):
        """
        creates a vector representation for each sequence in x.

        If y_c is None, return single vector representation.
        If y is tensor of [batch_size, y_length, nnlm_order],
            tile vector representations to size [batch_size, y_length, output_size]
        """
        x_mean = x.sum(1) / x_lengths.unsqueeze(1).float()
        out = self.fc_out(x_mean)
        if y_c is not None:
            out = out.unsqueeze(1)
            out = out.expand(out.size(0), y_c.size(1), out.size(-1))
        return out


class ConvEncoder(nn.Module):
    """
    Convolutional encoder
    From "A Neural Attention Model for Abstractive Sentence Summarization" (Rush et al. 2015)

    Uses ReLU activations instead of the tanh activations used in paper.
    Hyperparameters from paper:
    Q = 2 -> kernel_size = 5
    """

    def __init__(self, vocab_size, embedding_dim, n_layers, hidden_size, output_size):
        super(ConvEncoder, self).__init__()
        self.layers = nn.Sequential()
        in_size = embedding_dim
        for i in range(n_layers):
            self.layers.add_module('conv{}'.format(i),
                                   nn.Conv1d(in_size, hidden_size, 
                                             kernel_size=5, padding=2))
            self.layers.add_module('maxpool{}'.format(i),
                                   nn.MaxPool1d(2))
            self.layers.add_module('ReLU{}'.format(i),
                                   nn.ReLU())
            in_size = hidden_size

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, y_c, *_):
        """
        creates a vector representation for each sequence in x.

        If y_c is None, return single vector representation.
        If y is tensor of [batch_size, y_length, nnlm_order],
            tile vector representations to size [batch_size, y_length, output_size]
        """ 
        h = self.layers(x.transpose(2, 1))
        # max over time
        h = F.adaptive_max_pool1d(h, 1).squeeze()
        out = self.fc_out(h)
        if y_c is not None:
            out = out.unsqueeze(1)
            out = out.expand(out.size(0), y_c.size(1), out.size(-1))
        return out
