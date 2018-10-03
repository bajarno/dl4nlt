from collections import Counter
import csv
from nltk import sent_tokenize, word_tokenize
import sentencepiece as spm

class DataPreprocessor:

    replace_top_n_common_words = 0
    max_vocab_size = 1000

    vocab = None
    subword_processor = None

    def __init__(self, subword_model = None):
        if subword_model == None:
            counter = self.count_words(input_file)
            self.vocab = self.build_vocab(counter)
        else:
            self.subword_processor = spm.SentencePieceProcessor()
            self.subword_processor.Load(subword_model)


    def count_words(self, input_file):
        counter = Counter()
        with open(input_file, 'r') as f:
            csvreader = csv.reader(f)
            for title, content in csvreader:
                words = word_tokenize(title) + word_tokenize(content)
                for word in words:
                    counter[word] += 1

        return counter

    def build_vocab(self, counter):
        i_a = self.replace_top_n_common_words
        i_b = self.max_vocab_size + self.replace_top_n_common_words
        words, counts = zip(*counter.most_common()[i_a:i_b])
        return words

    def replace_unk(self, sentence):
        new_sentence = []
        for word in word_tokenize(sentence):
            if word in self.vocab:
                new_sentence.append(word)
            else:
                new_sentence.append('UNK')
        return new_sentence

    def process_whole_word(self, content):
        sentences = sent_tokenize(content)
        sentences = [['<s>'] + self.replace_unk(s) + ['</s>'] for s in sentences]
        sentences = ' '.join(sentences)
        return ' '.join(sentences)

    def process_sub_word(self, content):
        sentences = sent_tokenize(content)
        sentences = [['<s>'] + self.subword_processor.encode_as_pieces(s) + ['</s>'] for s in sentences]
        sentences = [' '.join(s) for s in sentences]
        return ' '.join(sentences)

    def preprocess(self, input_file, output_file):
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            csvreader = csv.reader(f_in)
            writer =  csv.writer(f_out)
            for title, content in csvreader:
                new_title = None
                new_content = None
                if self.subword_processor != None:
                    new_title = self.process_sub_word(title)
                    new_content = self.process_sub_word(content)
                else:
                    new_title = self.process_whole_word(title)
                    new_content = self.process_whole_word(content)
                writer.writerow([new_title, new_content])

if __name__ == '__main__':
    input_file = './kaggle_parsed_1_40_100.csv'
    output_file = './preprocessed/kaggle_preprocessed_sub_16000.csv'

    preprocessor = DataPreprocessor('./subword_model/subword16000.model')
    # preprocessor = DataPreprocessor()
    # preprocessor.max_vocab_size = 10000
    preprocessor.preprocess(input_file, output_file)