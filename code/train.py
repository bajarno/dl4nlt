import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import BOWEncoder, ConvEncoder
from dataloader import get_dataloaders
from nnlm import NNLM, FBModel

def accuracy(preds, targets):
	return torch.eq(preds, targets).float().mean()

def train(config):
	
	# Initialize the device which to run the model on
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# get torch loaders for training and test data
	train_loader, test_loader = get_dataloaders(config.dataset, 
												markov_order=config.order, batch_size=config.batch_size)
	vocab_size = train_loader.dataset.vocab_size
	
	# Load single test batch for evaluation
	test_X, test_Y, test_xl, test_yl = next(iter(test_loader))

	teacher_force_ratio = config.teacher_force_ratio

	# Define model
	embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=config.pad_token)

	if config.adasoft:
		output_size = 1024
	else:
		output_size = vocab_size

	if config.encoder_type == 'BOW':
		encoder = BOWEncoder(vocab_size, config.embedding_dim, output_size)
	elif config.encoder_type == 'Conv':
		# 4 layers -> minimal X length = 2^4
		encoder = ConvEncoder(vocab_size, config.embedding_dim, 4, config.hidden_size, output_size)
	elif config.encoder_type == 'Attn':
		raise NotImplementedError

	nnlm = NNLM(config.order, vocab_size, config.embedding_dim, [config.hidden_size]*3, output_size)
	model = FBModel(embedding, encoder, nnlm).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


	if config.adasoft:
		criterion = nn.AdaptiveLogSoftmaxWithLoss(1024, vocab_size, [100, 1000, 5000, 10000]).to(device)
	else:
		# EXPERIMENTAL: set UNK weight lower (maybe not needed with better vocab)
		loss_weights = torch.ones(vocab_size).to(device)
		if 'UNK' in train_loader.dataset.w2i:
			loss_weights[train_loader.dataset.w2i['UNK']] = 0.3
		criterion = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=0)

	for epoch in range(config.num_epochs):
		# TRAIN
		num_teacherforce = [0, 0]
		num_batch = len(train_loader)
		for batch_idx, (X, Y, xlen, ylen) in enumerate(train_loader):
			
			X = X.to(device)
			Y = Y.to(device)
			xlen = xlen.to(device)
			ylen = ylen.to(device)

			# Make ngrams and targets
			y_c = torch.stack([Y[:, i:i+config.order] for i in range(0, Y.size(1)-config.order)], 1)
			y_t = Y[:, config.order:]

			# Train step
			model.train()
			optimizer.zero_grad()

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
			out = out.reshape(-1, output_size)
			y_t = y_t.reshape(-1)
			loss = criterion(out.reshape(-1, output_size), y_t.reshape(-1))
			if config.adasoft:
				loss = loss.loss
			loss.backward()
			optimizer.step()

			if not batch_idx%20:
				if config.adasoft:
					pred = criterion.predict(out)
				else:
					pred = torch.argmax(out, -1)
				acc = accuracy(pred, y_t)
				print('Epoch {}, step {:04d}/{:04d} loss {:.4f} acc {:.4f}'.format(epoch, batch_idx, num_batch, loss.item(), acc.item()))
			
			if epoch > 0 and epoch % 10 == 0:
				torch.save(model, 'test_model_epoch_'+str(epoch)+'.pt')
				
		
		# EVAL
		model.eval()
		print(num_teacherforce)
		# Load test batch
		Y = test_Y.to(device)
		X = test_X.to(device)
		xlen = test_xl.to(device)
		ylen = test_yl.to(device)
		# Make ngrams and targets
		y_c = torch.stack([Y[:, i:i+config.order] for i in range(0, Y.size(1)-config.order)], 1)
		y_t = Y[:, config.order:]
		out = model(X, y_c, xlen, ylen)
		print(out.size())
		if config.adasoft:
			test_sentence = criterion.predict(out.reshape(-1, output_size)).reshape(out.size(0), out.size(1))
			test_sentence = test_sentence.cpu().numpy()
		else:
			test_sentence = torch.argmax(out[-1], -1).cpu().numpy()
		test_sentence = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in test_sentence]
		correct = y_t.cpu()[-1].numpy()
		correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]
		print(test_sentence)
		print(correct)
		print()

		# Decay teacherforcing
		teacher_force_ratio *= config.teacher_force_decay
		
if __name__ == "__main__":
	
	# Parse training configuration
	parser = argparse.ArgumentParser()

	# Data params
	parser.add_argument('--pad_token', type=int, default=0, help='Token (int) used for padding.')
	parser.add_argument('--order', type=int, default=5, help='TODO: add description.')
	
	# Model params
	parser.add_argument('--embedding_dim', type=int, default=128, help='Size of embedding.')
	parser.add_argument('--hidden_size', type=int, default=128, help='Amount of hidden units.')
	parser.add_argument('--encoder_type', type=str, default='Conv', help='Type of Encoder: BOW, Conv, or Attn.')
	parser.add_argument('--adasoft', type=bool, default=False, help='Use adaptive softmax.')

	# Training params
	parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch.')
	parser.add_argument('--learning_rate', type=float, default=5e-3, help='Learning rate.')
	parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
	parser.add_argument('--teacher_force_ratio', type=int, default=1, help='TODO: add description.')
	parser.add_argument('--teacher_force_decay', type=float, default=0.95, help='TODO: add description.')

	parser.add_argument('--dataset', type=str, default='../data/kaggle_preprocessed_subword_5000.csv', help='The datafile used for training')
	
	# Misc params
	#parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
	#parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

	config = parser.parse_args()

	# Train the model
	train(config)
