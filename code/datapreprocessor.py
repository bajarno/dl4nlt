from collections import Counter
from nltk import sent_tokenize, word_tokenize
import csv

class DataPreprocessor:

    replace_top_n_common_words = 0
    max_vocab_size = 1000

    def count_words(self, input_file):
        counter = Counter()
        with open(input_file, 'r') as f:
            csvreader = csv.reader(f)
            for title, content in csvreader:
                line = title + ' ' + content
                words = self.tokenize(line)
                for word in words:
                    counter[word] += 1

        return counter

    def build_vocab(self, counter):
        i_a = self.replace_top_n_common_words
        i_b = self.max_vocab_size + self.replace_top_n_common_words
        words, counts = zip(*counter.most_common()[i_a:i_b])
        return words

    def tokenize(self, text):
        words = []
        for sentence in sent_tokenize(text):
            words.append(word_tokenize(sentence))

        flat_words = [item for sublist in words for item in sublist] 
        return flat_words

    def replace_unk(self, sentence, vocab):
        new_sentence = []
        for word in self.tokenize(sentence):
            if word in vocab:
                new_sentence.append(word)
            else:
                new_sentence.append('UNK')
        return ' '.join(new_sentence)
    
    def save_file(self, input_file, vocab, output_file):
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            csvreader = csv.reader(f_in)
            writer =  csv.writer(f_out)
            for title, content in csvreader:
                new_title = self.replace_unk(title, vocab)
                new_content = self.replace_unk(content, vocab)
                writer.writerow(['<s> ' + new_title + ' </s>', '<s> ' + new_content + ' </s>'])
                

    def preprocess(self, input_file, output_file):
        counter = self.count_words(input_file)
        vocab = self.build_vocab(counter)
        self.save_file(input_file, vocab, output_file)


if __name__ == "__main__":
    input_file = 'test/testdata.csv'
    output_file = 'test/preprocessed_testdata.csv'

    preprocessor = DataPreprocessor()
    preprocessor.preprocess(input_file, output_file)

