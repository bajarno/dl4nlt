import torch
import torch.nn as nn
import torch.nn.functional as F

class BOWEncoder(nn.Module):
    """
    Bag-of-Words encoder
    From "A Neural Attention Model for Abstractive Sentence Summarization" (Rush et al. 2015)
    """
    def __init__(self, vocab_size, embedding_size, output_size):
        super(BOWEncoder, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, output_size)

    def forward(self, x, *args, **kwargs):
        return self.fc_out(self.embedding(x))


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
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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

    def forward(self, x, *args, **kwargs):
        h = self.embedding(x).transpose(2, 1)
        h = self.layers(h)
        # max over time
        h = F.adaptive_max_pool1d(h, 1).squeeze()
        out = self.fc_out(h)
        return out


if __name__=="__main__":
    batch_size = 4
    seqlength = 50
    vocab_size = 100
    hidden_size = 16
    x = torch.LongTensor(batch_size, seqlength).random_(vocab_size)
    print('input', x.size())
    bow = BOWEncoder(vocab_size, hidden_size, vocab_size)
    print('BOWEncoder out', bow.forward(x).size())
    conv = ConvEncoder(vocab_size, hidden_size, 3, hidden_size, vocab_size)
    print('ConvEncoder out', conv.forward(x).size())
