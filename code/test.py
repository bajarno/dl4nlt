
import numpy as np
import argparse
import os.path # or os import sys
import glob
import shutil
import time
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import BOWEncoder, ConvEncoder, AttnEncoder
from dataloader import get_dataloaders
from nnlm import NNLM, FBModel
from train import accuracy, beam_search, greedy_search

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
	config_old = config
	
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
		markov_order=config.order + 1, batch_size=config.batch_size)
	vocab_size = train_loader.dataset.vocab_size
			
	# The following steps are to initialize the model, which will be overloaded with the trained model
	
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
	
	# Define models (only as init since they will be overwritten)
	nnlm = NNLM(config.order, vocab_size, config.embedding_dim, [config.hidden_size]*3, output_size)
	model = FBModel(embedding, encoder, nnlm).to(device)
		
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
		# Make ngrams and targets
		Y = X.to(device)
		X = Y.to(device)
		xlen = xlen.to(device)
		ylen = (ylen-config.order).to(device)
		
		y_t = Y[:, config.order:]
		# Only for greedy search:
		if not config_old.beam_search_strategy:
			y_c = torch.stack([Y[:, i:i+config.order] for i in range(0, Y.size(1)-config.order)], 1)
			y_c = y_c[:,0:1]
			out_length = y_t.size(1)
			out = model(X, y_c, xlen, ylen, output_length=out_length, teacher_forcing=False)
		
		batch_correct = []
		batch_predicted = []
		for i in range(config.batch_size):
			if config_old.beam_search_strategy: # BEAM SEARCH PREDICTION
				beam_xlen = xlen[[i]].to(device)
				beam_Y = Y[[i],:].to(device)
				beam_X = X[[i],:beam_xlen].to(device)
				beam_ylen = torch.Tensor([1]).to(device)
			
				all_sequences = beam_search(config, model, beam_X, beam_Y, beam_xlen, beam_ylen)
				best_sequence = all_sequences[0][0]
				predicted = [test_loader.dataset.i2w[i] for i in best_sequence.squeeze().cpu().numpy() if i > 1]
			
			else: # GREEDY PREDICTION
 				predicted = torch.argmax(out[i], -1).cpu().numpy()
 				predicted = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in predicted]
			
			# CORRECT PREDICTION
			correct = y_t.cpu()[i].numpy()
			correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]	
			
			# Transform to rouge subwords or normal words
			if config_old.rouge_subwords:
				correct = ''.join(word for word in correct).replace('▁', ' ')
				predicted = ''.join(word for word in predicted).replace('▁', ' ')
			else:
				predicted = ' '.join(word for word in predicted)
				correct = ' '.join(word for word in correct)
			
			print("PREDICTED:", predicted[0])
			print("CORRECT:", correct[0])
			batch_predicted.append(predicted)
			batch_correct.append(correct)
		
		'''
		if config_old.beam_search_strategy: # BEAM SEARCH	
			# Calculate avg rouge scores over batch
			batch_correct = []
			batch_test_sentence = []
			y_t = Y[:, config.order:]
			for i in range(config.batch_size):						
				beam_xlen = xlen[[i]].to(device)
				beam_Y = Y[[i],:].to(device)
				beam_X = X[[i],:beam_xlen].to(device)
				beam_ylen = torch.Tensor([1]).to(device)
				
				all_sequences = beam_search(config, model, beam_X, beam_Y, beam_xlen, beam_ylen)
				best_sequence = all_sequences[0][0]
				test_sentence = [test_loader.dataset.i2w[i] for i in best_sequence.squeeze().cpu().numpy() if i > 1]
				
				correct = y_t.cpu()[i].numpy()
				correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]
				
				if config_old.rouge_subwords:
					correct = ''.join(word for word in correct).replace('▁', ' ')
					test_sentence = ''.join(word for word in test_sentence).replace('▁', ' ')
				else:
					test_sentence = ' '.join(word for word in test_sentence)
					correct = ' '.join(word for word in correct)
					
				batch_test_sentence.append(test_sentence)
				batch_correct.append(correct)
						
		else: # GREEDY SEARCH
			y_c = torch.stack([Y[:, i:i+config.order] for i in range(0, Y.size(1)-config.order)], 1)
			y_t = Y[:, config.order:]
		
			# No teacher forcing
			y_c = y_c[:,0:1]
			out_length = y_t.size(1)
			out = model(X, y_c, xlen, ylen, output_length=out_length, teacher_forcing=False)
		
			# Calculate avg rouge scores over batch
			batch_correct = []
			batch_test_sentence = []
			for i in range(len(out)):
				test_sentence = torch.argmax(out[i], -1).cpu().numpy()
				test_sentence = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in test_sentence]
				correct = y_t.cpu()[i].numpy()
				correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]

				if config_old.rouge_subwords:
					correct = ''.join(word for word in correct).replace('▁', ' ')
					test_sentence = ''.join(word for word in test_sentence).replace('▁', ' ')
				else:
					test_sentence = ' '.join(word for word in test_sentence)
					correct = ' '.join(word for word in correct)

				batch_test_sentence.append(test_sentence)
				batch_correct.append(correct)			
			'''
		rouge = rouge_eval.get_scores(batch_predicted, batch_correct, True) # output format is dict	
		
		# Turn dict into lists and sum all corresponding elements with total
		rouge_scores[0][0] += rouge['rouge-1']['f']
		rouge_scores[0][1] += rouge['rouge-1']['p']
		rouge_scores[0][2] += rouge['rouge-1']['r']
		rouge_scores[1][0] += rouge['rouge-2']['f']
		rouge_scores[1][1] += rouge['rouge-2']['p']
		rouge_scores[1][2] += rouge['rouge-2']['r']
		rouge_scores[2][0] += rouge['rouge-l']['f']
		rouge_scores[2][1] += rouge['rouge-l']['p']
		rouge_scores[2][2] += rouge['rouge-l']['r']

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
	parser.add_argument('--dataset', type=str, default='../data/kaggle_preprocessed_subword_5000.csv', help='The datafile used for training')
	parser.add_argument('--order', type=int, default=5, help='TODO: add description.')
	
	# Model params
	parser.add_argument('--rouge_subwords', action="store_false", default=True)
		
	# Testing params
	parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch.')
	
	# Beam search
	parser.add_argument('--beam_search_strategy', type=bool, required=True)
	parser.add_argument('--beam_search_k', type=int, default=2, help='')

	config = parser.parse_args()

	# Test the model
	test(config)