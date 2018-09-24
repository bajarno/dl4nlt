import json
import csv
import os
from tqdm import tqdm
from pprint import pprint

'''
This script transforms .json files, downloaded from https://webhose.io/datasets/ (News articles by category),
into one .csv file with title and text as the two columns.
This is the same format as the kaggle dataset, such that these can be merged.
'''
MAX_CHAR_LEN = 5000 # roughly 900 words
DIR = "all_articles/" # contains (sub)directories with json files
WRITE_DIR = "csv/"
WRITE_CSV = False

csv_data = [] # list to store title and text pairs

# loop through directory with topic dirs
for dirname in [dir_ for dir_ in os.listdir(DIR) if '.DS_Store' not in dir_]: # remove hidden mac files
	print("Going through", dirname, "directory...")
	# loop through directory with json files
	for filename in tqdm([dir_ for dir_ in os.listdir(DIR+dirname) if '.DS_Store' not in dir_]): # remove hidden mac files
		
		# open file, get relevant data in correct format
		with open(DIR+dirname+'/'+filename) as f:
			data = json.load(f)
			title = data['title']
			title = title.replace('\n', ' ').replace('\t', ' ').strip()
			text = data['text']
			if len(text) > MAX_CHAR_LEN:
				text = text[:MAX_CHAR_LEN]
			text = text.replace('\n', ' ').replace('\t', ' ').strip()
			csv_data.append([title, text])

if WRITE_CSV:
	# open csv file, write elements of csv_data to rows
	with open(WRITE_DIR+'articles_large.csv', 'w') as csvfile:
		filewriter = csv.writer(csvfile)
		for row in csv_data:
			filewriter.writerow(row)
