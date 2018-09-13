import torch
import torch.nn as nn
import torch.nn.functional as F

class BOWEncoder(nn.Module):
    """
    Bag-of-Words encoder
    From "A Neural Attention Model for Abstractive Sentence Summarization" (Rush et al. 2015)
    """
    def __init__(self, vocab_size, embedding_dim, output_size):
        super(BOWEncoder, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, output_size)

    def forward(self, x):
        return self.fc_out(self.embedding(x))


class ConvEncoder(nn.Module):
    """
    Convolutional encoder
    From "A Neural Attention Model for Abstractive Sentence Summarization" (Rush et al. 2015)
    """

    def __init__(self, vocab_size, embedding_dim, output_size):
        super(ConvEncoder, self).__init__()

    def forward(self, x):
        pass