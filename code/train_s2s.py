import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import get_dataloaders
from attnseq2seq import S2S, S2SEncoder, S2SAttnDecoder

def accuracy(preds, targets):
    return torch.eq(preds, targets).float().mean()

def train(config):
    # Initialize the device which to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get torch loaders for training and test data
    train_loader, test_loader = get_dataloaders(config.dataset, markov_order=2, batch_size=config.batch_size)
    vocab_size = train_loader.dataset.vocab_size

    # Load single test batch for evaluation
    test_X, test_Y, test_xl, test_yl = next(iter(test_loader))

    teacher_force_ratio = config.teacher_force_ratio

    encoder = S2SEncoder(vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, dropout=config.dropout)
    decoder = S2SAttnDecoder(vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, dropout=config.dropout)
    model = S2S(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []
    for epoch in range(config.num_epochs):
        # TRAIN
        num_teacherforce = [0, 0]
        num_batches = len(train_loader)
        for batch_idx, (X, Y, xlen, ylen) in enumerate(train_loader):
            
            X = X.to(device)
            Y = Y.to(device)
            Y_in = Y[:,:-1]
            Y_t = Y[:, 1:]
            xlen = xlen.to(device)
            # ylen -= 1, outputs do not predict start token
            ylen = (ylen - 1).to(device)

            # Train step
            model.train()
            optimizer.zero_grad()

            # No teacher forcing
            if np.random.random() > teacher_force_ratio:
                num_teacherforce[0] += 1
                Y_in = Y_in[:,0:1]
                ylen = torch.ones_like(ylen).to(device)
                out_length =Y_t.size(1)
                out = model(X, Y_in, xlen, ylen, output_length=out_length, teacher_forcing=False)
            else:
                num_teacherforce[1] += 1
                out = model(X, Y_in, xlen, ylen, teacher_forcing=True)

            # Loss, optimization step
            loss = criterion(out.reshape(-1, vocab_size), Y_t.reshape(-1))
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if not batch_idx%20:
                pred = torch.argmax(out, -1)
                acc = accuracy(pred, Y_t)
                print('[Epoch {}/{}], step {:04d}/{:04d} loss {:.4f} acc {:.4f}'.format(epoch + 1, config.num_epochs, batch_idx, num_batches, loss.item(), acc.item()))
            
            if (epoch + 1 % 10 == 0 or epoch + 1 == config.num_epochs) and batch_idx == num_batches - 1: #save model every final step of each 10 epochs or last epoch
                torch.save(model, config.output_dir + '/test_model_epoch_'+str(epoch+1)+'.pt')

            if has_converged(losses):
                print('Model converged')
                return


        # EVAL
        # model.eval()
        # print(num_teacherforce)
        # # Load test batch
        # Y = test_Y.to(device)
        # X = test_X.to(device)
        # xlen = test_xl.to(device)
        # ylen = test_yl.to(device)
        # # Make ngrams and targets
        # y_c = torch.stack([Y[:, i:i+config.order] for i in range(0, Y.size(1)-config.order)], 1)
        # y_t = Y[:, config.order:]
        # out = model(X, y_c, xlen, ylen)
        # print(out.size())
        # if config.adasoft:
        #     test_sentence = criterion.predict(out.reshape(-1, output_size)).reshape(out.size(0), out.size(1))
        #     test_sentence = test_sentence.cpu().numpy()
        # else:
        #     test_sentence = torch.argmax(out[-1], -1).cpu().numpy()
        # test_sentence = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in test_sentence]
        # correct = y_t.cpu()[-1].numpy()
        # correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]
        # print(test_sentence)
        # print(correct)
        # print()

        # Decay teacherforcing
        teacher_force_ratio *= config.teacher_force_decay

def has_converged(losses):
    min_steps = 5
    if len(losses) < min_steps:
        return False

    for i in range(0, min_steps -1):
        diff = abs(losses[-(i+1)] - losses[-(i+2)]) 
        if diff > 1e-4:
            return False

    return True
        
if __name__ == "__main__":
    
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--pad_token', type=int, default=0, help='Token (int) used for padding.')
    
    # Model params
    parser.add_argument('--embedding_dim', type=int, default=128, help='Size of embedding.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Amount of hidden units.')
    parser.add_argument('--num_layers', type=int, default=1, help="Number of layers in encoder and decoder")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout value')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--teacher_force_ratio', type=int, default=1, help='TODO: add description.')
    parser.add_argument('--teacher_force_decay', type=float, default=0.95, help='TODO: add description.')

    parser.add_argument('--dataset', type=str, default='../data/kaggle_preprocessed_subword_5000.csv', help='The datafile used for training')
    parser.add_argument('--output_dir', type=str, default='./', help='The directory used for saving the model')
    
    # Misc params
    #parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    #parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
