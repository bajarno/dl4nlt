"""
Sequence-to-sequence model
- Bidirectional LSTM encoder.
- LSTM decoder.
- Masking for encoder rnn, decoder rnn and attention layer.
- S2SAttnDecoder output is not masked.
- Luong et al. [1] global general attention.

[1] Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.

Author: Eelco van der Wel
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class S2S(nn.Module):
    """
    Wrapper for the encoder-decoder S2S model
    """
    def __init__(self, encoder, decoder):
        super(S2S, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, xlen, ylen, teacher_forcing=True, output_length=None):
        if teacher_forcing:
            encoder_outputs, h = self.encoder(x, xlen)
            return self.decoder(y, ylen, h, encoder_outputs, xlen)[0]

        else:
            encoder_outputs, h = self.encoder(x, xlen)

            decoder_outputs = []
            for _ in range(output_length):
                decoder_output, h = self.decoder(y, ylen, h, encoder_outputs, xlen)
                decoder_outputs.append(decoder_output)
                y = torch.argmax(decoder_output.detach(), dim=-1)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            return decoder_outputs



class S2SEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.):
        super(S2SEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, 
                           dropout=dropout, batch_first=True, bidirectional=True)

        # To project bidirectional output back to hidden_size
        self.fc_out = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x, lengths=None):
        x = self.embedding(x)

        # RNN
        # Encoder and Decoder inputs both have to be sorted by length, but dont always have
        # same ordering after sorting. Solution: Inputs are expected to be sorted for decoder, 
        # and encoder sorts/unsorts internally.
        if lengths is not None:
            # Get sort/unsort orders, sort x.
            lengths_sorted, sort_idx = lengths.sort(0, descending=True)
            unsort_idx = sort_idx.sort(0)[1]
            x = x[sort_idx]
            x = pack_padded_sequence(x, lengths_sorted, batch_first=True)
        x, h = self.rnn(x)
        if lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
            # Unsort x and h
            x = x[unsort_idx]
            h = (h[0][:, unsort_idx, :], h[1][:, unsort_idx, :])
        out = self.fc_out(x)
        h_forward = self._get_forward_hidden(h)
        return out, h_forward

    def _get_forward_hidden(self, hidden):
        """ 
        Encoder is bidirectional, decoder is not. To pass the last encoder hidden state to the
        decoder, only the hidden + cell state of the forward direction is passed to the decoder.
        """
        h_bi, c_bi = hidden
        # Forward states are on even indices
        h_forward = h_bi[::2].contiguous()
        c_forward = c_bi[::2].contiguous()
        return (h_forward, c_forward)


class S2SAttnDecoder(nn.Module):
    """
    Implements LSTM Attention decoder with masking over encoder outputs.

    Attention mechanism is Luong attention with 'general' score function.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0., enc_bidirectional=True):
        super(S2SAttnDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, 
                           dropout=dropout, batch_first=True)

        self.attn = MaskedAttention(hidden_size)
        self.fc_context = nn.Linear(hidden_size*2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    
    def forward(self, decoder_inputs, decoder_lengths, 
                prev_hidden, encoder_outputs, encoder_lengths):

        # Inputs through decoder RNN
        emb = self.embedding(decoder_inputs)
        emb = pack_padded_sequence(emb, decoder_lengths, batch_first=True)
        decoder_outputs, decoder_h = self.rnn(emb, prev_hidden)
        decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)

        # Attention
        attn_weights = self.attn(encoder_outputs, decoder_outputs, encoder_lengths, decoder_lengths)
        # Calculate weighted average of encoder outputs with attention weights.
        # output = [batch_size, encoder_seq_length, hidden_dim]
        # attn_weights = [batch_size, decoder_seq_length, encoder_seq_length]
        context = torch.bmm(attn_weights, encoder_outputs)

        # Calculate output
        concat = torch.cat((decoder_outputs, context), -1)
        concat = torch.tanh(self.fc_context(concat))
        out = self.fc_out(concat)
        return out, decoder_h
        

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