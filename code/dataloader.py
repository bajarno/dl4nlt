import torch
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import numpy as np
import torch.utils.data

class TextCSVDataset(torch.utils.data.Dataset):

    def __init__(self, fn):
        self.fn = fn
        self.max_tokens_content = 100
        self.max_tokens_title = 30 
        self.read_data()

    def read_data(self):
        self.w2i = dict()
        self.i2w = dict()
        self.titles = []
        self.content = []

        with open(self.fn, 'r') as f_in:
            csvreader = csv.reader(f_in)
            for title, content in csvreader:
                title = self.add_vocab_return_indices(title, is_title=True)
                content = self.add_vocab_return_indices(content, is_title=False)

                self.titles.append(title)
                self.content.append(content)

    def add_vocab_return_indices(self, string, is_title=False):
        max_tokens = self.max_tokens_title if is_title else self.max_tokens_content
        string = string.strip().split()[:max_tokens]
        indices = [0] * max_tokens
        for i, word in enumerate(string):
            self.add_word(word)
            indices[i] = self.w2i[word]

        return indices

    def add_word(self, word):
        if word not in self.w2i:
            index = len(self.i2w) + 1
            self.w2i[word] = index
            self.i2w[index] = word

    def __getitem__(self, x):
        title = self.titles[x]
        content = self.content[x]
        return torch.LongTensor(title), torch.LongTensor(content)

    def __len__(self):
        return len(self.titles)

def get_dataloaders(fn, batch_size=16, validation_split=.2):
    dataset = TextCSVDataset(fn)
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, validation_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders('test/preprocessed_testdata.csv')

    for batch_idx, (titles, contents) in enumerate(train_loader):
        print(batch_idx, titles.shape, contents.shape)
        print(titles)
        break
