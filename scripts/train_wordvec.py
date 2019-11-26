# encoding:utf8

"""
Train word vector.
"""

from env import *  # NOQA

import logging
from text_summ import config
from text_summ.model.word_embedding import train_wordvec

_logger = logging.getLogger()


def main():
    train_wordvec(config.NEWS_CORPUS_FILEPATH, binary=True,
                  save_wv=True, wv_filepath=config.WV_WORD_VECTOR_FILEPTH)


if __name__ == '__main__':
    main()
