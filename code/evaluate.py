import numpy as np
from rouge import Rouge # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # sudo pip install -U nltk

def bleu_score(ground_truth, predicted, n_gram=0):
	'''
	Calculates BLEU score for given sentence pair.
	Inputs:
		- ground_truth: a string containing the correct title.
		- predicted: a string containing the predicted title.
		- n_gram: the n_gram for which you want the blue score
				(default 0, thus all n_grams have equal count).
	'''
	smoothing = SmoothingFunction()
	smooth = smoothing.method4
	#TODO: choose one from 8 available smoothing functions
	#TODO: fix warning "The hypothesis contains 0 counts of 4-gram overlaps..." when not smoothing
	
	ground_truth = [ground_truth.split(' ')]
	predicted = predicted.split(' ')
	
	if min(len(ground_truth[0]), len(predicted)) < 5:
		# if minimum length is smaller than 5 (maximum n_gram size is 4),
		# reweight so that only the lower n_grams count
		weights = np.ones(4, dtype=float)
		weights[len(predicted):] = 0
		weights[:len(predicted)] = weights[:len(predicted)] / len(predicted)
	elif n_gram == 0:
		# if 0 (default), all n_grams get equal weights
		weights = np.ones(4, dtype=float) / 4
	else:
		# if specified, only that n_gram counts (weights = 1, the others 0)
		weights = np.zeros(4, dtype=int)
		weights[n_gram-1] = 1

	return sentence_bleu(ground_truth, predicted, tuple(weights), smooth)
	
def rouge_score(ground_truth, predicted, avg=True):
	'''
	Calculates ROUGE score for given sentence pair.
	Inputs:
		- ground_truth: a string containing the correct title.
		- predicted: a string containing the predicted title.
		- avg: True: take avg of multiple pairs; False: return seperate scores.
	'''
	rouge = Rouge()
	return rouge.get_scores(predicted, ground_truth, avg)

# TESTING
ground_truth = 'the quick brown fox jumped over the lazy dog'
predicted = 'the fast brown fox jumped over the lazy dog or not'

print("Ground truth:  ", ground_truth)
print("Predicted:     ", predicted)
print("Bleu - equal   ", bleu_score(ground_truth, predicted, 0))
print("Bleu - n=1     ", bleu_score(ground_truth, predicted, 1))
print("Bleu - n=2     ", bleu_score(ground_truth, predicted, 2))
print("Bleu - n=3     ", bleu_score(ground_truth, predicted, 3))
print("Bleu - n=4     ", bleu_score(ground_truth, predicted, 4))
print("Rouge          ", rouge_score(ground_truth, predicted))

#TODO: wrapper for Pytorch such that this can be used during training / testing
