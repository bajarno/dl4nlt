import csv
import re

class KaggleParser:
    min_content_length = 10 # Minimum amount of words in content
    max_content_length = 50 # Maximum amount of words in content

    def processTitle(self, title):

        # Remove paper name from title
        title = title.replace(' - The New York Times', '')

        # Ignore Atlantic Daily articles due to noise
        if title.startswith('The Atlantic Daily:'):
            return None

        return title


    def processContent(self, content):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)

        maxindex = 0

        for s in sentences:
            length = len(s.split())

            if maxindex + length < self.max_content_length:
                maxindex += length
            else:
                break

        content = content.split()[:maxindex]

        if len(content) < self.min_content_length:
            return None

        content = ' '.join(content)
        return content

    def process_source(self, source_path, target_path):
        source = open(source_path, 'r')
        target = open(target_path, 'w')

        reader = csv.reader(source)
        writer = csv.writer(target)

        i_r = 0
        for row in reader:
            if i_r > 100:
                break

            if i_r > 0:
                title = self.processTitle(row[2])
                content = self.processContent(row[9])

                if title != None and content != None:
                    writer.writerow([title,content])

            i_r += 1

if __name__ == '__main__':
    source_path = '../data/kaggle/articles2.csv'
    target_path = '../data/kaggle_parsed.csv'

    parser = KaggleParser()
    parser.process_source(source_path, target_path)



