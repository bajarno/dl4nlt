import torch
import torch.nn.functional as F

from encoders import BOWEncoder
from nnlm import NNLM

if __name__=="__main__":
    context_size = 4 # y context
    input_size = 100 # length of first paragraph
    vocab_size = 1000
    embedding_dim = 32
    hidden_size = 128
    batch_size = 16

    # x encoder
    encoder = BOWEncoder(vocab_size, embedding_dim, vocab_size)
    x = torch.LongTensor(batch_size, input_size).random_(0, vocab_size)
    x_enc = encoder(x)

    # context encoder
    y_c = torch.LongTensor(batch_size, context_size).random_(0, vocab_size)
    ctx_encoder = NNLM(context_size, vocab_size, embedding_dim, hidden_size, vocab_size)
    yc_enc = ctx_encoder.forward(y_c)

    # combine (as in 3.1, but V*h and W*enc already happen in encoder and NNLM)
    py_ycx = F.softmax(x_enc + yc_enc, dim=-1)
    print('input x', x.size())
    print('input y_c', y_c.size())
    print('output p(y | y_c, x)', py_ycx.size())