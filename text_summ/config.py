# encoding: utf8

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


WORKERS = 8

WV_WORD_DIM = 128
NEWS_CORPUS_ORIGIN_FILEPATH = os.path.join(BASE_DIR, 'datasets/sqlResult_1558435.csv')
NEWS_CORPUS_DATASET_FILEPATH = os.path.join(BASE_DIR, 'datasets/news_corpus.csv')
NEWS_CORPUS_FILEPATH = os.path.join(BASE_DIR, 'datasets/news_corpus.txt')
WV_WORD_VECTOR_FILEPTH = os.path.join(BASE_DIR, 'datasets/wordvectors.kv')
WV_WORD_VECTOR_TEXT_FILEPTH = os.path.join(BASE_DIR, 'datasets/wordvectors.kv.txt')
