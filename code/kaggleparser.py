import csv
import re
import sys
from nltk import sent_tokenize, word_tokenize
import sentencepiece as spm

csv.field_size_limit(sys.maxsize)

class KaggleParser:
    min_content_length = 10 # Minimum amount of words in content
    max_content_length = 100 # Maximum amount of words in content

    tag_re = re.compile(r'<[^>]+>') # Regular expression for removing HTML tags

    subword_processor = None

    def __init__(self, subword_model = None):
        if subword_model != None:
            self.subword_processor = spm.SentencePieceProcessor()
            self.subword_processor.Load(subword_model)

    def processTitle(self, title):
        title = self.tag_re.sub('', title) # Remove HTML tags

        # Remove paper name from title
        title = title.replace(' - The New York Times', '')
        title = title.replace(' - Breitbart', '')

        # Ignore Atlantic Daily articles due to noise
        if title.startswith('The Atlantic Daily:'):
            return None

        title = title.lower()

        tokens = ['<s>']
        if self.subword_processor != None:
            tokens += self.subword_processor.encode_as_pieces(title)
        else:
            tokens += word_tokenize(title)
        tokens += ['</s>']

        return tokens

    def processContent(self, content):
        content = self.tag_re.sub('', content) # Remove HTML tags
        content = content.lower()
        sentences = sent_tokenize(content)

        tokens = []
        for s in sentences:
            s_tokens = ['<s>']
            if self.subword_processor != None:
                s_tokens += self.subword_processor.encode_as_pieces(s)
            else:
                s_tokens += word_tokenize(s)
            s_tokens += ['</s>']

            # Add sentence if total amount of tokens is lower than max
            if len(tokens) + len(s_tokens) < self.max_content_length:
                tokens += s_tokens
            else:
                break

        if len(tokens) < self.min_content_length:
            return None

        return tokens

    def process_source(self, source, target):
        reader = csv.reader(source)
        writer = csv.writer(target)

        next(reader, None) # skip header

        for row in reader:
            title = self.processTitle(row[2])
            content = self.processContent(row[9])

            if title != None and content != None and len(title) < len(content):
                writer.writerow([' '.join(title), ' '.join(content)])

    def process_sources(self, source_paths, target_path):
        target = open(target_path, 'w')

        for source_path in source_paths:
            source = open(source_path, 'r')

            self.process_source(source, target)

if __name__ == '__main__':
    source_paths = ['../data/kaggle/articles1.csv', '../data/kaggle/articles2.csv', '../data/kaggle/articles3.csv']
    target_path = '../data/kaggle_parsed_subword_20000.csv'

    parser = KaggleParser('../data/subword20000.model')
    parser.process_sources(source_paths, target_path)



