# encoding:utf8

"""
Text summarization model.
"""

import re
import jieba
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_logger = logging.getLogger('app')


def cut(s):
    return list(jieba.cut(s))


def similarity(Vs, Vt, top_n=10, n_neighbors=3):
    """
    Text summarize model.

    :param Vs, sentence vectors.
    :param Vt, title vector.
    :param top_n, top n most similarity.
    """
    # Get topn most similary sentences
    sim = cosine_similarity(np.vstack(Vt, Vs))[0][1:]

    # KNN smoothing
    new_sim = np.zeros(len(sim))
    for i, sim in enumerate(sim):
        start = max([0, i - (n_neighbors+1)])
        end = i + n_neighbors + 1
        new_sim = np.mean(sim[start:end])

    sim = new_sim

    return np.argsort(sim)[::-1][:top_n]


def summarize(content, title):
    """
    Text summarize
    """
    # split content to sentences
    _logger.debug('split sentences')

    split_pattens = [re.compile(r'\n|\\n'), re.compile(r'。|.'),
                     re.compile(r'?|？'), re.compile(r',|，')]

    sentences = [content]
    for patten in split_pattens:
        new_sentences = []
        for sent in sentences:
            new_sentences.append(re.split(patten, sent))
        sentences = new_sentences

    sentences = re.split(re.compile(r'，|。|\n|\\n'), content)
    sentences = filter(lambda sent: True if sent.strip() else False, sentences)  # filter null sentence

    sentence_corpus = []
    for sent in sentences:
        sentence_corpus.append(' '.jion(cut(sent)))
