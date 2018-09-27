from evaluate import rouge_score
#from evaluate import blue_score # TODO: fix import error

import numpy as np
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from collections import defaultdict
from tabulate import tabulate # pip install tabulate

from dataloader import get_dataloaders

def test(config):
	
	# Initialize the device which to run the model on
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# get torch loaders for training and test data
	train_loader, test_loader = get_dataloaders('../data/kaggle_parsed_preprocessed_10000_vocab.csv', 
												markov_order=config.order, batch_size=config.batch_size)
	
	# load pre-trained model
	model = torch.load(config.model_path)
	model.eval() # evaluation mode
	
	#TODO: check if faster with (merging) dicts instead of lists
	#TODO: use rouge's 'avg' to calculate multiple examples at once 
	'''
	rouge_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
					'rouge-2': {'f': 0, 'p': 0, 'r': 0},
					'rouge-l': {'f': 0, 'p': 0, 'r': 0}
					}
	'''
					# f, p, r = (f1-score, precision, recall)
	rouge_scores = [[0, 0, 0], # rouge-1
					[0, 0, 0], # rouge-2
					[0, 0, 0]] # rouge-l
					
	num_examples = 0
	for batch_idx, (X, Y, xlen, ylen) in enumerate(test_loader):
		# Make ngrams and targets
		Y = X.to(device)
		X = Y.to(device)
		
		# Make ngrams and targets
		y_c = torch.stack([Y[:, i:i+config.order] for i in range(0, Y.size(1)-config.order)], 1)
		y_t = Y[:, config.order:]
		out = model(X, y_c, xlen, ylen)
		#print("out", out.size())
		
		test_sentence = torch.argmax(out[-1], -1).cpu().numpy()
		test_sentence = [test_loader.dataset.i2w[i] if i > 0 else 'PAD' for i in test_sentence]
		correct = y_t.cpu()[-1].numpy()
		correct = [test_loader.dataset.i2w[i] for i in correct if i > 0]
		#print("test_sentence", test_sentence)
		#print("correct", correct)
		#print()
		
		test_sentence = ' '.join(word for word in test_sentence)
		correct = ' '.join(word for word in correct)
		rouge = rouge_score(correct, test_sentence) # output format is dict
		
		# turn dict into lists and sum all corresponding elements with total
		for i in range(len(rouge_scores)):
			rouge_scores[i] = [round(sum(x), 2) for x in zip(rouge_scores[i], list(list(rouge.values())[i].values()))]
		
		num_examples += 1

	# divide by total test examples to get avg
	rouge_scores = [[num / num_examples for num in rouge] for rouge in rouge_scores]
	
	# temporarily: just to show the results
	print(tabulate([['F1', rouge_scores[0][0], rouge_scores[1][0], rouge_scores[2][0]],
					['P', rouge_scores[0][1], rouge_scores[1][1], rouge_scores[2][1]],
					['R', rouge_scores[0][2], rouge_scores[1][2], rouge_scores[2][2]]],
					headers=['', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']))	
		
		
if __name__ == "__main__":
	
	# Parse training configuration
	parser = argparse.ArgumentParser()

	# Data params
	parser.add_argument('--model_path', type=str, default='test_model_epoch_50.pt', help='Path (including filename) for saved model.')
	parser.add_argument('--pad_token', type=int, default=0, help='Token (int) used for padding.')
	parser.add_argument('--order', type=int, default=5, help='TODO: add description.')
	
	# Model params
	parser.add_argument('--embedding_dim', type=int, default=128, help='Size of embedding.')
	parser.add_argument('--hidden_size', type=int, default=128, help='Amount of hidden units.')
	parser.add_argument('--encoder_type', type=str, default='Conv', help='Type of Encoder: BOW, Conv, or Attn.')
		
	# Testing params
	parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch.')
	#TODO: get this information from the saved model

	config = parser.parse_args()

	# Test the model
	test(config)

		
		
		
		