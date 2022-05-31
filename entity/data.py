
import os
import helper
import string

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad = 'pad'
        self.idx2word.append(self.pad)
        self.word2idx[self.pad] = len(self.idx2word) - 1
        self.start = '<s>'
        self.idx2word.append(self.start)
        self.word2idx[self.start] = len(self.idx2word) - 1
        self.stop = '</s>'
        self.idx2word.append(self.stop)
        self.word2idx[self.stop] = len(self.idx2word) - 1
        self.unk_token = '<unk>'
        self.idx2word.append(self.unk_token)
        self.word2idx[self.unk_token] = len(self.idx2word) - 1

    def build_dict(self, sample):
        word_count = Counter()
        for query in sample.data:
            word_count.update(query.query_terms)

        print("total word number: ",len(word_count))
        most_common = word_count.most_common(len(word_count))

        for (index, w) in enumerate(most_common):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)

class Query(object):
    def __init__(self):
        self.query_terms = []

    def add_text(self, text, tokenize, max_length, w2v):
        content_terms = helper.query2vec(text,w2v)
        if len(content_terms) > max_length:
            self.query_terms = content_terms[:max_length]
        else:
            self.query_terms =  content_terms


class Corpus(object):
    def __init__(self, _tokenize, max_q_len):
        self.tokenize = _tokenize
        self.data = []
        self.max_q_len = max_q_len

    def parse(self, unique_queries, w2v, max_example=None):
        """Parses the content of a unique queries"""
        for query in unique_queries:
            current_query = Query()
            current_query.add_text(query, self.tokenize, self.max_q_len, w2v)
            self.data.append((current_query))


class char_Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad = 'pad'
        self.idx2word.append(self.pad)
        self.word2idx[self.pad] = len(self.idx2word) - 1
        self.start = '<s>'
        self.idx2word.append(self.start)
        self.word2idx[self.start] = len(self.idx2word) - 1
        self.stop = '</s>'
        self.idx2word.append(self.stop)
        self.word2idx[self.stop] = len(self.idx2word) - 1
        self.unk_token = '<unk>'
        self.idx2word.append(self.unk_token)
        self.word2idx[self.unk_token] = len(self.idx2word) - 1

    def build_dict(self):
        all_letters = string.ascii_letters+string.digits+string.punctuation

        for (index, w) in enumerate(all_letters):
            self.idx2word.append(w[0])
            self.word2idx[w[0]] = len(self.idx2word) - 1


    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)
