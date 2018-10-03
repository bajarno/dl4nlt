
import numpy as np
import argparse
import os.path # or os
import sys
import glob
import shutil
import time
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import get_dataloaders
from attnseq2seq import S2S, S2SEncoder, S2SAttnDecoder
from train import accuracy

from collections import Counter
from collections import defaultdict
from tabulate import tabulate # pip install tabulate

from rouge import Rouge # pip install rouge

def current_rouge_scores(rouge_scores, num_examples, visualize=True):
	# Divide by total test examples to get avg
	rouge_scores = [[num / num_examples for num in rouge] for rouge in rouge_scores]
	if visualize:
		print(tabulate([['F1', round(rouge_scores[0][0], 3), round(rouge_scores[1][0], 3), round(rouge_scores[2][0], 3)],
						['P', round(rouge_scores[0][1], 3), round(rouge_scores[1][1], 3), round(rouge_scores[2][1], 3)],
						['R', round(rouge_scores[0][2], 3), round(rouge_scores[1][2], 3), round(rouge_scores[2][2], 3)]],
						headers=['', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']))
	return rouge_scores

def test(config):
	
	# Initialize the device which to run the model on
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# Load pre-trained model
	file_path = config.model_dir+config.model_file
	if os.path.isfile(file_path):
		print('Loading checkpoint \'{}\''.format(file_path))
		checkpoint = torch.load(file_path)
		config = checkpoint['config'] # Use saved config
		print('Loaded  checkpoint \'{}\' (epoch {})'.format(file_path, checkpoint['epoch']))
	else:
		print('No checkpoint found at \'{}\''.format(file_path))
		sys.exit('Please check the filename.')
	
	# Get torch loaders for training and test data
	train_loader, test_loader = get_dataloaders(config.dataset, 
		markov_order=2, batch_size=config.batch_size)
	vocab_size = train_loader.dataset.vocab_size
			
	# The following steps are to initialize the model, which will be overloaded with the trained model
	
	encoder = S2SEncoder(vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, dropout=config.dropout)
	decoder = S2SAttnDecoder(vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, dropout=config.dropout)
	model = S2S(encoder, decoder).to(device)
	
	# Load model from checkpoint and put in evalulation mode
	model.load_state_dict(checkpoint['model'])
	model.eval()
	print('Model loaded from checkpoint, start evaluation.')
	
						# f, p, r = (f1-score, precision, recall)
	rouge_scores = [[0, 0, 0], # rouge-1
					[0, 0, 0], # rouge-2
					[0, 0, 0]] # rouge-l
					
	num_examples = 0
	
	rouge_eval = Rouge()
	
	for batch_idx, (X, Y, xlen, ylen) in enumerate(test_loader):
		X = X.to(device)
		Y = Y.to(device)
		Y_in = Y[:,:-1]
		Y_t = Y[:, 1:]
		xlen = xlen.to(device)
		# ylen -= 1, outputs do not predict start token
		ylen = (ylen - 1).to(device)

		# No teacher forcing
		Y_in = Y_in[:,0:1]
		ylen = torch.ones_like(ylen).to(device)
		out_length =Y_t.size(1)
		out = model(X, Y_in, xlen, ylen, output_length=out_length, teacher_forcing=False)
	   
		# Calculate avg rouge scores over batch
		batch_correct = []
		batch_test_sentence = []
		for i in range(len(out)):
			test_sentence = torch.argmax(out[i], -1).cpu().numpy()
			test_sentence = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in test_sentence]
			correct = Y_t.cpu()[i].numpy()
			correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]
			batch_test_sentence.append(' '.join(word for word in test_sentence))
			batch_correct.append(' '.join(word for word in correct))			
		rouge = rouge_eval.get_scores(batch_correct, batch_test_sentence, True) # output format is dict	
		print(rouge)
		# Turn dict into lists and sum all corresponding elements with total
		for i in range(len(rouge_scores)):
			rouge_scores[i] = [round(sum(x), 2) for x in zip(rouge_scores[i], list(list(rouge.values())[i].values()))]
		
		num_examples += 1
		
		# Show every 10 batches
		if batch_idx % 10 == 0:
			print("batch_idx:", batch_idx)
			# Current average rouge scores
			temp_rouge_scores = current_rouge_scores(rouge_scores, num_examples)
	
	# Final average rouge scores
	final_rouge_scores = current_rouge_scores(rouge_scores, num_examples)		
		
if __name__ == "__main__":
	
	# Parse training configuration
	parser = argparse.ArgumentParser()

	# Data params
	parser.add_argument('--model_dir', type=str, default='../model_checkpoints/', help='Path to saved model that needs to be tested.')
	parser.add_argument('--model_file', type=str, required=True, help='Filename of saved model.')
	parser.add_argument('--dataset', type=str, default='../data/kaggle_preprocessed_subword_10000.csv', help='The datafile used for training')
	
	parser.add_argument('--pad_token', type=int, default=0, help='Token (int) used for padding.')
	
	# Model params
	parser.add_argument('--embedding_dim', type=int, default=128, help='Size of embedding.')
	parser.add_argument('--hidden_size', type=int, default=128, help='Amount of hidden units.')
	parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in encoder and decoder")
	parser.add_argument('--dropout', type=float, default=0.3, help='dropout value')
		
	# Testing params
	parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch.')

	config = parser.parse_args()

	# Test the model
	test(config)

		
		
		
		