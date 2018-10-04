import csv
import re
import sys
from nltk import sent_tokenize

csv.field_size_limit(sys.maxsize)

class KaggleParser:
    min_title_length = 1

    min_content_length = 40 # Minimum amount of words in content
    max_content_length = 100 # Maximum amount of words in content

    tag_re = re.compile(r'<[^>]+>') # Regular expression for removing HTML tags

    def processTitle(self, title):
        title = self.tag_re.sub('', title) # Remove HTML tags
        title = title.lower()

        # Remove paper name from title
        title = title.replace(' - The New York Times', '')
        title = title.replace(' - Breitbart', '')

        # Title length needs to be bigger than specified minimum.
        if len(title.split()) < self.min_title_length:
            return None

        # Ignore Atlantic Daily articles due to noise
        if title.startswith('The Atlantic Daily:'):
            return None

        return title

    def processContent(self, content):
        content = self.tag_re.sub('', content) # Remove HTML tags
        content = content.lower()
        sentences = sent_tokenize(content)

        content_sentences = []
        total_word_count = 0
        for s in sentences:
            s_word_count = len(s.split())

            # Add sentence if total amount of tokens is lower than max
            if s_word_count + total_word_count < self.max_content_length:
                content_sentences += [s]
                total_word_count += s_word_count
            else:
                break

        if total_word_count < self.min_content_length:
            return None

        return ' '.join(content_sentences)

    def process_source(self, source, target):
        reader = csv.reader(source)
        writer = csv.writer(target)

        next(reader, None) # skip header

        for row in reader:
            title = self.processTitle(row[2])
            content = self.processContent(row[9])

            if title != None and content != None and len(title.split()) < len(content.split()):
                writer.writerow([title, content])

    def process_sources(self, source_paths, target_path):
        target = open(target_path, 'w')

        for source_path in source_paths:
            source = open(source_path, 'r')

            self.process_source(source, target)

if __name__ == '__main__':
    source_paths = ['../data/kaggle/articles1.csv', '../data/kaggle/articles2.csv', '../data/kaggle/articles3.csv']
    target_path = './kaggle_parsed_1_40_100.csv'

    parser = KaggleParser()
    parser.process_sources(source_paths, target_path)