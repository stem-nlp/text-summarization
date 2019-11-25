# encoding:utf8

"""
Train word vector.
"""

from env import *  # NOQA

import logging
from text_summ import config
from text_summ.model.word2vector import train_wordvec

_logger = logging.getLogger()


def main():
    train_wordvec(config.NEWS_CORPUS_FILEPATH, save_wv=True)


if __name__ == '__main__':
    main()
    _logger.info('debug')
