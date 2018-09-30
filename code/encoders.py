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


class AttnEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, y_order):
        super(AttnEncoder, self).__init__()
        self.y_order = y_order
        self.attn = MaskedAttention(hidden_size)
        self.fc_y = nn.Linear(hidden_size*y_order, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y, xlen, ylen):
        y = torch.reshape(y, (y.size(0), y.size(1), -1))
        y = self.fc_y(y)
        scores = self.attn(x, y, xlen, ylen)
        return self.fc_out(torch.bmm(scores, x))


class MaskedAttention(nn.Module):
    """
    Implements Luong global attention with 'general' score mechanism.
    https://arxiv.org/pdf/1508.04025.pdf
    
    score(d_i, e_j) = dot(d_i, W_a @ e_j)
    Where d_i is i-th decoder output, e_j is j-th encoder output
    a_(d_i -> e_j) = softmax(score(d_i, e_j))
    
    Implementation differs slightly from paper, since W_a is nn.Linear (bias parameters)
    For 'dot' score mechanism: remove W
    """
    def __init__(self, hidden_size):
        super(MaskedAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, encoder_outputs, decoder_outputs, encoder_lengths=None, decoder_lengths=None):
        """
        encoder outputs: FloatTensor shape (batch_size, encoder_length, hidden_size)
        decoder_outputs: FloatTensor shape (batch_size, decoder_length, hidden_size)
        encoder_lengths: LongTensor shape (batch_size)
        decoder_lengths: LongTensor shape (batch_size)
        """
        
        # score[b, i, j] = score for sample b, decoder_i -> encoder_j 
        scores = torch.bmm(decoder_outputs, self.W(encoder_outputs).permute(0,2,1))
        
        if encoder_lengths is None:
            a = torch.softmax(scores, -1)
        else:
            mask = make_2d_mask(decoder_lengths, encoder_lengths)
            a = masked_softmax(scores, mask, -1)
        return a
        
        
def lengths_to_mask(lengths, max_length=None):
    """
    creates mask m for each length in length, 
    where m[i, :lengths[i]] = 1, rest is 0
    """
    if max_length is None:
        max_length = lengths.data.max()
    batch_size = lengths.size(0)

    rng = torch.arange(max_length).expand(batch_size, max_length).cpu()
    lengths = lengths.unsqueeze(1).expand(batch_size, max_length).cpu()
    return rng < lengths

def make_2d_mask(l1, l2):
    """
    Generates 2d binary mask by taking the carthesian product of mask(l1) x mask(l2)
    example:
    
    >>> make_2d_mask(torch.LongTensor([3,2]), torch.LongTensor([1,4]))
    tensor([[[1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],

            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [0., 0., 0., 0.]]])
    """
    m1 = lengths_to_mask(l1)
    m2 = lengths_to_mask(l2)
    return torch.bmm(m1.unsqueeze(2), m2.unsqueeze(1)).float()
    
def masked_softmax(logits, mask, dim, minval=-1e10):
    smask = torch.zeros_like(mask)
    smask[mask == 0] = minval
    return torch.softmax(logits + smask.to(logits.device), dim)

    
if __name__ == "__main__":
    h = 50
    order = 5
    batch = 10
    l = 10
    x = torch.FloatTensor(batch, l, h).random_(1)
    y = torch.FloatTensor(batch, l, order, h).random_(1)
    xlen = torch.LongTensor(batch).random_(l)
    xlen[0] = l
    ylen = torch.LongTensor(batch).random_(l)
    ylen[0] = l
    e = AttnEncoder(h, order)
    a = e(x, y, xlen, ylen)
    print(a.size())