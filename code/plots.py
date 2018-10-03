#!/usr/bin/env python

from collections import Counter
import argparse
import matplotlib.pyplot as plt
import csv

def parse_file_line_by_line(filename):
    with open(filename, 'r', encoding="utf8") as f:
        for sentence in f.readlines():
            yield sentence.strip().split()

def parse_csv(fn):
    with open(fn, 'r', encoding="utf8") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            yield ' '.join(row).strip().split()

def get_args():
    parser = argparse.ArgumentParser(description='Make word a word frequency plot')

    parser.add_argument('files', metavar='file, label', type=str, nargs="+")
    parser.add_argument('--csv', default=parse_file_line_by_line, const=parse_csv, action="store_const")
    parser.add_argument('--save', type=str, help='Filename to save figure')
    parser.add_argument('--show', type=bool, default=True)

    args = parser.parse_args()

    assert len(args.files) % 2 == 0 , 'number of files and labels should be equal'

    args.files = [(args.files[i], args.files[i+1]) for i in range(0, len(args.files), 2)]

    return args


def collect_count(file, fn):
    count_per_word = Counter()
    word_per_sentence = Counter()

    for sentence in fn(file):
        word_per_sentence[len(sentence)] += 1
        for word in sentence:
            count_per_word[word] += 1


    vocab = set([word for word, _ in count_per_word.items()])
    words, counts = zip(*count_per_word.most_common())
    sen_length, sen_counts = zip(*word_per_sentence.most_common())

    return {'count_per_word': count_per_word,
            'word_per_sentence': word_per_sentence,
            'words': words,
            'counts': counts,
            'sen_length': sen_length,
            'sen_counts': sen_counts,
            'vocab': vocab,
    }

def collect_counts(files, fn):
    counts = []

    for file, label in files:
        counts.append((file, label, collect_count(file, fn)))

    return counts

def print_stats(stats, file):
    print('{}'.format(file))
    print('{} tokens, vocab {}'.format(sum(stats['count_per_word'].values()), len(stats['vocab'])))
    print(stats['count_per_word'].most_common(10))
    print(stats['word_per_sentence'].most_common(10))

def print_stats_all(counts):
    for file, label, count in counts:
        print_stats(count, label)

def plot_word_freqs(counts, fn=None, show=False):
    for file, label, count in counts:
        plt.loglog(range(len(count['counts'])), count['counts'], label=label)
    plt.legend()

    plt.ylabel('Word frequency')
    plt.xlabel('Word rank')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')

    if show:
        plt.show()

if __name__ == "__main__":
    args = get_args()
    counts = collect_counts(args.files, args.csv)
    print_stats_all(counts)
    plot_word_freqs(counts, fn=args.save, show=args.show)

