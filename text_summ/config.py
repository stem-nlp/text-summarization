# encoding: utf8

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


WORKERS = 8

WV_WORD_DIM = 128
WV_WORD_VECTOR_FILEPTH = os.path.join(BASE_DIR, 'datasets/wordvectors.k')
NEWS_CORPUS_FILEPATH = os.path.join(BASE_DIR, 'datasets/news_corpus.txt')
