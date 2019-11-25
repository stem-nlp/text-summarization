# encoding: utf8


"""
Generate corpus data.
"""

from env import *  # NOQA
from text_summ.model.preprocess import process_corpus_data


def main():
    process_corpus_data()


if __name__ == '__main__':
    main()
