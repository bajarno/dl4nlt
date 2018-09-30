import torch
from collections import Counter
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import logging
import numpy as np
import torch.utils.data

logger = logging.getLogger(__name__)

class TextCSVDataset(torch.utils.data.Dataset):

    def __init__(self, fn, markov_order=2):
        if markov_order < 2:
            raise ValueError('Markov order < 2 is currently not supported')

        self.fn = fn
        self.order = markov_order
        self.max_padding_article = 120
        self.max_padding_title = 40 
        self.read_data()

        logger.info('Vocab size {}'.format(self.vocab_size))
        logger.info('Lines {}'.format(len(self.titles)))
        logger.info('Max title length (tokens) {}'.format(max(self.title_lengths)))
        logger.info('Max article length (tokens) {}'.format(max(self.article_lengths)))

    @property
    def vocab_size(self):
        return len(self.w2i) + 1 # to account for padding token

    def read_data(self):

        logger.info('Reading dataset %s', self.fn)
        self.w2i = dict()
        self.i2w = dict()

        self.titles = []
        self.title_lengths = []
        self.articles = []
        self.article_lengths = []

        counts = self.cap_text_and_count_frequencies()
        self.build_dictionary(counts)
        self.titles = self.convert_to_indices(self.titles)
        self.articles = self.convert_to_indices(self.articles)

    def build_dictionary(self, counts):
        words, _ = zip(*counts.most_common())
        for index, word in enumerate(words):
            self.w2i[word] = index + 1
            self.i2w[index + 1] = word


    def convert_to_indices(self, data):
        for i, datum in enumerate(data):
            for j, word in enumerate(datum):
                data[i][j] = self.w2i[word]

        return data

    def cap_text_and_count_frequencies(self):
        counter = Counter()

        with open(self.fn, 'r') as f_in:
            csvreader = csv.reader(f_in)
            for title, article in csvreader:
                title = self.cap_string(title, self.max_padding_title)
                article = self.cap_string(article, self.max_padding_article)
                
                self.titles.append(title)
                self.title_lengths.append(len(title))
                self.articles.append(article)
                self.article_lengths.append(len(article))

                for word in title + article:
                    counter[word] += 1

        return counter

    def cap_string(self, string, max_length):
        string = string.strip().split()
        extra_starttags = self.order - 2
        string = [string[0]] * extra_starttags + string
        string = string[:max_length]

        return string

    def add_word(self, word):
        if word not in self.w2i:
            index = len(self.i2w) + 1
            self.w2i[word] = index
            self.i2w[index] = word

    def __getitem__(self, x):
        title = self.titles[x]
        article = self.articles[x]
        return torch.LongTensor(article), torch.LongTensor(title), self.article_lengths[x], self.title_lengths[x]

    def make_square(self, inputs):
        n_data = len(inputs)
        max_length = max([len(item) for item in inputs])
        square = torch.zeros(n_data, max_length, dtype=torch.long)

        for i, item in enumerate(inputs):
            square[i, :len(item)] = item

        return square

    def make_batch(self, data):
        # longest Y first
        data.sort(key=lambda x: x[3], reverse=True)
        articles, titles, article_lengths, title_lengths = zip(*data)
        
        # dynamic padding
        articles = self.make_square(articles)
        titles = self.make_square(titles)
        return articles, titles, torch.LongTensor(article_lengths), torch.LongTensor(title_lengths)

    def __len__(self):
        return len(self.titles)

def get_dataloaders(fn, markov_order=2, batch_size=16, validation_split=.2, shuffle_dataset=True, random_seed=1):
    # Creating data indices for training and validation splits:
    dataset = TextCSVDataset(fn, markov_order)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.make_batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=dataset.make_batch)

    return train_loader, validation_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    train_loader, test_loader = get_dataloaders('test/preprocessed_testdata.csv')

    for batch_idx, (articles, titles, article_lengths, title_lengths) in enumerate(train_loader):
        logger.debug('Print first batch to check results')
        logger.debug('batch_idx %s, titles.shape %s, articles.shape %s, article_lengths.shape %s, title_lengths.shape %s', batch_idx, titles.shape, articles.shape, article_lengths.shape, title_lengths.shape)
        logger.debug('titles %s', titles)
        logger.debug('title_lengths %s', title_lengths)
