# encoding: utf8

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence


from text_summ import config


def train_wordvec(corpus_filepath):
    """
    Train word vectors.
    """
    sentences = LineSentence(corpus_filepath)
    model = Word2Vec(sentences, config.WV_WORD_DIM, workers=config.WORKERS)
    model.kv.save(config.WV_WORD_VECTOR_FILEPTH)

    return model
