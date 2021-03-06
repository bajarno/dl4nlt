import numpy as np
import argparse
import os.path
import sys
import glob
import shutil
import time

import torch
import torch.nn as nn

from encoders import BOWEncoder, ConvEncoder, AttnEncoder
from dataloader import get_dataloaders
from nnlm import NNLM, FBModel

def accuracy(preds, targets):
	return torch.eq(preds, targets).float().mean()

# Create dict with all params to save the model
def create_state(config, model, optimizer, criterion, epoch, loss, acc):
	state = {
		'config': config,
		'model': model.state_dict(), #TODO: check if .cpu() is needed
		'optimizer': optimizer.state_dict(),
		'epoch': epoch + 1,
		'loss': loss,
		'acc': acc
	}	
	return state
	
# Save most recent model and delete older model(s)
def save_model(state, is_best_model, model_dir, encoder_type, embedding_dim, hidden_size, loss):
	# Filename includes encoder type, embedding_dim, hidden_size, loss
	filename = '{}model-enc_{}-emb_{}-hid_{}-loss_{}'.format(
				model_dir, encoder_type.lower(), config.embedding_dim, hidden_size, round(loss, 3))
	filename_best = filename + '-best.pth.tar'
	filename += '.pth.tar'
	
	# Search for older models
	idx = filename.find('loss_')+5
	search_for_old = filename[:idx]+'*.pth.tar'
	results = sorted(glob.glob(search_for_old))
	
	print('Replacing latest model with current model into {}'.format(model_dir))
	
	# Delete older model
	for model in results:
		if '-best' not in model:
			os.remove(model)
	
	# Save current model
	torch.save(state, filename)
	
	if is_best_model:
		print('Replacing   best model with current model into {}'.format(model_dir))
		
		# Delete older best model
		for model in results:
			if '-best' in model:
				os.remove(model)
		# Copy the latest model as best
		shutil.copyfile(filename, filename_best)
		
		
# Check if current loss is lower than loss of saved model
def check_is_best(model_dir, encoder_type, embedding_dim, hidden_size, loss):
	search_for_best = '{}model-enc_{}-emb_{}-hid_{}-loss_*-best.pth.tar'.format(
						model_dir, encoder_type.lower(), config.embedding_dim, hidden_size)
	results = glob.glob(search_for_best)
	if len(results) > 0:
		best = sorted(results)[0] # Sort files alphabetically (lowest loss will be first)
		idx_start = best.find('loss_')+5
		idx_end = best.find('-best')
		lowest_loss = float(best[idx_start:idx_end])
		if loss < lowest_loss:
			return True # Current model is better
		else:
			return False # Saved model is better
	else:
		return True # No saved model yet
		
# Check if decrease of loss has stagnated
def has_converged(losses):
    min_steps = 5
    if len(losses) < min_steps:
        return False

    for i in range(0, min_steps -1):
        diff = abs(losses[-(i+1)] - losses[-(i+2)]) 
        if diff > 1e-4:
            return False

    return True

def beam_search(config, model, X, Y, xlen, ylen):
	k = config.beam_search_k
	sequence_len = len(np.trim_zeros(Y[:, config.order:].squeeze().cpu().numpy()))
	all_sequences = list()
	active_sequences = [[torch.stack([Y[:, :config.order]], 1), 0]]
	for beam_depth in range(sequence_len):
		all_candidates = list()
		for i in range(len(active_sequences)):
			seq, score = active_sequences[i]
			out = model(X, seq[:,:,-config.order:], xlen, ylen)
			out = torch.log_softmax(out, dim=-1)
			top_probs, top_idx = torch.topk(out, k+1, dim=-1, sorted = True)
			eos_idx = (top_idx[:,:,:k] == 3).squeeze().nonzero()
			num_candidates = k if eos_idx.shape[0] == 0 else k+1
			for j in range(num_candidates):
				candidate = [torch.cat((seq,top_idx[:,:,[j]]), dim=-1), score + top_probs[:,:,[j]]] # log probs!
				if (eos_idx.shape[0] == 0 or eos_idx != j):
					all_candidates.append(candidate)
				else:
					all_sequences.append(candidate)
		ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
		active_sequences = ordered[:k]

	all_sequences = all_sequences + active_sequences
	for i in range(len(all_sequences)):
		all_sequences[i][1] = all_sequences[i][1]/(all_sequences[i][0].shape[2]**config.length_bonus)

	all_sequences = sorted(all_sequences, key=lambda tup:tup[1], reverse=True)
	return all_sequences[:k]

def greedy_search(model, X, Y, xlen, ylen, test_loader):
	sequence_len = len(np.trim_zeros(Y[:, config.order:].squeeze().cpu().numpy()))
	y_c = torch.stack([Y[:, i:i+config.order] for i in range(0, Y.size(1)-config.order)], 1)
	y_c = y_c[:,0:1]
	out = model(X, y_c, xlen, ylen, output_length=sequence_len, teacher_forcing=False)
	greedy_sentence = torch.argmax(out[-1], -1).cpu().numpy()
	greedy_sentence = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in greedy_sentence]
	return greedy_sentence

def train(config):
	# Initialize the device which to run the model on
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("device:", device)
	# Get torch loaders for training and test data
	train_loader, test_loader = get_dataloaders(config.dataset, 
												markov_order=config.order+1, batch_size=config.batch_size)
	vocab_size = train_loader.dataset.vocab_size
	
	# Load single test batch for evaluation
	test_X, test_Y, test_xl, test_yl = next(iter(test_loader))
	
	# If we want the continue training and the given filename exists, load all params
	# Otherwise just start training from scratch
	if config.continue_training:
		file_path = config.model_dir+config.continue_training
		if os.path.isfile(file_path):
			print('Loading checkpoint \'{}\''.format(file_path))
			checkpoint = torch.load(file_path)
			config = checkpoint['config'] # Use saved config
			config.start_epoch = checkpoint['epoch']			
			print('Loaded  checkpoint \'{}\' (epoch {})'.format(file_path, checkpoint['epoch']))
			config.continue_training = file_path # To make sure it is no empty string
		else:
			print('No checkpoint found at \'{}\''.format(file_path))
			sys.exit('Please check the filename.')

	teacher_force_ratio = config.teacher_force_ratio

	# Define model
	embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=config.pad_token)

	if config.adasoft:
		output_size = 1024
	else:
		output_size = vocab_size

	# Define encoder
	if config.encoder_type == 'BOW': # Bag of Words
		encoder = BOWEncoder(vocab_size, config.embedding_dim, output_size)
	elif config.encoder_type == 'Conv': # Convolutions
		# 4 layers -> minimal X length = 2^4
		encoder = ConvEncoder(vocab_size, config.embedding_dim, 4, config.hidden_size, output_size)
	elif config.encoder_type == 'Attn': # Attention
		encoder = AttnEncoder(vocab_size, config.embedding_dim, config.order)

	# Define models and optimizer
	nnlm = NNLM(config.order, vocab_size, config.embedding_dim, [config.hidden_size]*3, output_size)
	model = FBModel(embedding, encoder, nnlm).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

	# If we want to continue training, load the existing model and optimizer
	if config.continue_training and checkpoint != None:
		print('Model and optimizer are copied from checkpoint.')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		
	# Define loss
	if config.adasoft:
		criterion = nn.AdaptiveLogSoftmaxWithLoss(1024, vocab_size, [100, 1000, 5000, 10000]).to(device)
	else:
		# EXPERIMENTAL: set UNK weight lower (maybe not needed with better vocab)
		loss_weights = torch.ones(vocab_size).to(device)
		if 'UNK' in train_loader.dataset.w2i:
			loss_weights[train_loader.dataset.w2i['UNK']] = 0.3
		criterion = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=0)
	
	if config.start_epoch >= config.num_epochs:
		sys.exit('Already trained for specified amount of epochs. Consider increasing num_epochs.')
	else:
		print('Start training.')
	losses = []
	for epoch in range(config.start_epoch, config.num_epochs):
		# TRAIN
		num_teacherforce = [0, 0]
		num_batches = len(train_loader)
		starttime = time.time()
		for batch_idx, (X, Y, xlen, ylen) in enumerate(train_loader):
			X = X.to(device)
			Y = Y.to(device)
			xlen = xlen.to(device)
			# Because we have history of size config.order, actual y_length is total y_length - order
			ylen = (ylen-config.order).to(device)
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
			losses.append(loss.item())
			loss.backward()
			optimizer.step()
			if not batch_idx%20:
				if config.adasoft:
					pred = criterion.predict(out)
				else:
					pred = torch.argmax(out, -1)
				acc = accuracy(pred, y_t)
				print('[Epoch {}/{}], step {:04d}/{:04d} loss {:.4f} acc {:.4f} time {:.4f}'.format(epoch +1, config.num_epochs, batch_idx, num_batches, loss.item(), acc.item(), time.time() - starttime ))
				starttime = time.time()
			# Save model every final step of each 10 epochs or last epoch
			#if (epoch + 1 % 10 == 0 or epoch + 1 == config.num_epochs) and batch_idx == num_batches - 1:
			#	torch.save(model, config.output_dir + '/test_model_epoch_'+str(epoch+1)+'.pt')
			if batch_idx % 500 == 0:
				state = create_state(config, model, optimizer, criterion, epoch, loss, accuracy)
				is_best_model = check_is_best(config.model_dir, config.encoder_type, config.embedding_dim, config.hidden_size, loss.item())
				save_model(state, is_best_model, config.model_dir, config.encoder_type, config.embedding_dim, config.hidden_size, loss.item())
								
			if has_converged(losses):
				print('Model has converged.')
				return
			break
		# Decay teacherforcing
		teacher_force_ratio *= config.teacher_force_decay

		# EVAL #TODO: Seperate script or move to test.py
		model.eval()

		# Choose random sample
		test_idx = np.random.randint(config.batch_size)

		# Load random sample from test batch
		xlen = test_xl[[test_idx]].to(device)
		Y = test_Y[[test_idx],:].to(device)
		X = test_X[[test_idx],:xlen].to(device)
		ylen = torch.Tensor([1]).to(device)

		# Greedy Search
		greedy_sequence = greedy_search(model, X, Y, xlen, ylen, test_loader)

		# Beam Search
		all_sequences = beam_search(config, model, X, Y, xlen, ylen)

		# Target sequence
		y_t = Y[:, config.order:]
		correct = y_t.cpu()[-1].numpy()
		correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]

		# print results
		print("greedy :", greedy_sequence)
		for counter, sequence in enumerate(all_sequences):
			print("number", counter+1, ":", [test_loader.dataset.i2w[i] for i in sequence[0].squeeze().cpu().numpy() if i > 1] )
		print('correct :', correct)
		print()

		
if __name__ == "__main__":
	
	# Parse training configuration
	parser = argparse.ArgumentParser()

	# Data params
	parser.add_argument('--pad_token', type=int, default=0, help='Token (int) used for padding.')
	parser.add_argument('--order', type=int, default=5, help='TODO: add description.')
	
	# Model params
	parser.add_argument('--embedding_dim', type=int, default=128, help='Size of embedding.')
	parser.add_argument('--hidden_size', type=int, default=128, help='Amount of hidden units.')
	parser.add_argument('--encoder_type', type=str, default='Attn', help='Type of Encoder: BOW, Conv, or Attn.')
	parser.add_argument('--adasoft', type=bool, default=False, help='Use adaptive softmax.')

	# Training params
	parser.add_argument('--continue_training', type=str, default='', help='Name of saved model that needs to train further.')
	parser.add_argument('--model_dir', type=str, default='../model_checkpoints/', help='Path to saved model that needs to train further.')
	parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch.')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
	parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
	parser.add_argument('--start_epoch', type=int, default=0, help='Start at this epoch.')
	parser.add_argument('--teacher_force_ratio', type=int, default=0.99, help='TODO: add description.')
	parser.add_argument('--teacher_force_decay', type=float, default=0.95, help='TODO: add description.')

	parser.add_argument('--dataset', type=str, default='../data/kaggle_preprocessed_subword_5000.csv', help='The datafile used for training')
	parser.add_argument('--output_dir', type=str, default='./', help='The directory used for saving the model')

	parser.add_argument('--beam_search_k', type=int, default=5, help='The number of sequences to store in the beam search algorithm')
	parser.add_argument('--length_bonus', type=float, default=1.4, help='Giving preference to longer sequences (if > 1) or shorter (if < 1)')

	# Misc params
	#parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
	#parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

	config = parser.parse_args()

	# Train the model
	train(config)
