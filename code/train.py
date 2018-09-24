import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import BOWEncoder, ConvEncoder
from dataloader import get_dataloaders
from nnlm import NNLM, FBModel


def accuracy(preds, targets):
    return torch.eq(preds, targets).float().mean()

if __name__=="__main__":
    PAD_TOKEN = 0
    DEVICE = 'cuda'
    batch_size = 64
    order = 5
    train_loader, test_loader = get_dataloaders('../data/kaggle_parsed_preprocessed_10000_vocab.csv', markov_order=order, batch_size=batch_size)
    vocab_size = train_loader.dataset.vocab_size
    embedding_dim = 128
    hidden_size = 128
    num_epochs = 50
    teacher_force_ratio = 1
    teacher_force_decay = 0.95
    encoder_type = 'Conv'

    # Load single test batch for evaluation
    test_X, test_Y, test_xl, test_yl = next(iter(test_loader))

    # Define model
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_TOKEN)

    if encoder_type == 'BOW':
        encoder = BOWEncoder(vocab_size, embedding_dim, vocab_size)
    elif encoder_type == 'Conv':
        # 4 layers -> minimal X length = 2^4
        encoder = ConvEncoder(vocab_size, embedding_dim, 4, hidden_size, vocab_size)
    elif encoder_type == 'Attn':
        raise NotImplementedError

    nnlm = NNLM(order, vocab_size, embedding_dim, [hidden_size]*3, vocab_size)
    model = FBModel(embedding, encoder, nnlm).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    # EXPERIMENTAL: set UNK weight lower (maybe not needed with better vocab)
    loss_weights = torch.ones(vocab_size).to(DEVICE)
    loss_weights[train_loader.dataset.w2i['UNK']] = 0.3
    crit = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=0)

    for epoch in range(num_epochs):
        # TRAIN
        num_teacherforce = [0, 0]
        for batch_idx, (X, Y, xlen, ylen) in enumerate(train_loader):
            
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            # Make ngrams and targets
            y_c = torch.stack([Y[:, i:i+order] for i in range(0, Y.size(1)-order)], 1)
            y_t = Y[:, order:]

            # Train step
            model.train()
            opt.zero_grad()

            # No teacher forcing
            if np.random.random() > teacher_force_ratio:
                num_teacherforce[0] += 1
                y_c = y_c[:,0:1]
                out_length = y_t.size(1)
                out = model(X, y_c, xlen, ylen, output_length=out_length, teacher_forcing=False)
            else:
                num_teacherforce[1] += 1
                out = model(X, y_c, xlen, ylen, teacher_forcing=True)

            # Loss, optimization step
            loss = crit(out.transpose(2, 1), y_t)
            loss.backward()
            opt.step()

            if not batch_idx%20:
                acc = accuracy(torch.argmax(out, -1), y_t)
                print('{} loss {:.4f} acc {:.4f}'.format(epoch, loss.item(), acc.item()), end='\r')
        
        # EVAL
        model.eval()
        print(num_teacherforce)
        # Make ngrams and targets
        Y = test_Y.to(DEVICE)
        X = test_X.to(DEVICE)
        # Make ngrams and targets
        y_c = torch.stack([Y[:, i:i+order] for i in range(0, Y.size(1)-order)], 1)
        y_t = Y[:, order:]
        out = model(X, y_c, test_xl, test_yl)
        print(out.size())
        test_sentence = torch.argmax(out[-1], -1).cpu().numpy()
        test_sentence = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in test_sentence]
        correct = y_t.cpu()[-1].numpy()
        correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]
        print(test_sentence)
        print(correct)
        print()

        # Decay teacherforcing
        teacher_force_ratio *= teacher_force_decay