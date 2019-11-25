# encoding: utf8

import logging
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence

from text_summ import config

_logger = logging.getLogger()


class TrainCallback(CallbackAny2Vec):
    """
    Train Callback
    """

    def __init__(self,):
        self.epochs = 0

    def on_epoch_end(self, model):
        self.epochs += 1
        _logger.info('Train Epoch {} end'.format(self.epochs))


def train_wordvec(corpus_filepath, binary=True, save_wv=False):
    """
    Train word vectors.
    """
    sentences = LineSentence(corpus_filepath)
    model = Word2Vec(sentences, size=config.WV_WORD_DIM, workers=config.WORKERS,
                     callbacks=(TrainCallback(),))

    if save_wv:
        model.wv.save_word2vec_format(config.WV_WORD_VECTOR_FILEPTH, binary=binary)

    return model


def load_wordvec(wordvec_filepath, binary=True):
    return KeyedVectors.load_word2vec_format(wordvec_filepath, binary=binary)


def analogy(model, x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]
