# encoding:utf8

"""
Text summarization model.
"""

import re
import jieba
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from text_summ import config
from .word_embedding import load_wordvec
from .sif_embedding import sif_embedding
from .utils import load_stopwords

_logger = logging.getLogger('app')


def cut(s):
    return list(jieba.cut(s))


def most_similarity(Vs, Vt, top_n=10, n_neighbors=3):
    """
    Text summarize model.

    :param Vs, sentence vectors.
    :param Vt, title vector.
    :param top_n, top n most similarity.
    """
    # Get topn most similary sentences
    X = np.vstack((Vt, Vs))
    similarities = cosine_similarity(X)[0][1:]

    # KNN smoothing
    new_sim = np.zeros(len(similarities))
    for i, sim in enumerate(similarities):
        start = max([0, i - (n_neighbors+1)])
        end = i + n_neighbors + 1
        new_sim[i] = np.mean(similarities[start:end])

    sim = new_sim

    return sorted(np.argsort(sim)[::-1][:top_n])


def summarize(content, title, top_n=3):
    """
    Text summarize
    """
    # split content to sentences
    _logger.debug('split sentences')

    sentences = re.split(r'\n|\\n|\.|。|,|，', content)
    sentences = list(filter(lambda sent: True if sent.strip() else False, sentences))  # filter null sentence
    _logger.info('sentences shape {}'.format(np.array(sentences).shape))

    sentence_corpus = []
    # title sentence
    sentence_corpus.append(' '.join(cut(title)))
    # content sentences
    for sent in sentences:
        sentence_corpus.append(' '.join(cut(sent)))
    sentence_corpus = np.array(sentence_corpus)

    # load word vector
    _logger.debug('load trained word vector')
    wv = load_wordvec(config.WV_WORD_VECTOR_FILEPTH)

    # load stop words
    stop_words = load_stopwords(config.STOPWORDS_FILEPATH)

    # sentence vector
    _logger.info('sentence vector')
    _logger.info('sentence corpus shape {}'.format(sentence_corpus.shape))
    sent_vector = sif_embedding(sentence_corpus, wv, stop_words=stop_words)
    _logger.debug('sentence vector shape {}'.format(sent_vector.shape))

    Vt = sent_vector[0]
    Vs = sent_vector[1:]

    # get topn similary sentences
    sent_ind = most_similarity(Vs, Vt, top_n=top_n, n_neighbors=1)
    _logger.debug(sent_ind)

    most_similar_sents = sentence_corpus[1:][sent_ind]
    most_similar_sents = list(map(lambda sent: sent.replace(' ', ''), most_similar_sents))
    summary = '，'.join(most_similar_sents) + '。'

    return summary
