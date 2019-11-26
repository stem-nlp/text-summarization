# encoding:utf8

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


def sif_embedding(sentences: list, wv, wv_dim, alpha=1e-3, nc=1,
                  stop_words=(), **kwargs):
    """
    SIF(smooth inverse frequency) embedding.

    :param sentences, sentence list, eg: ['小米 很 不错', '小米 手机 还 行']
    :param wv, word vectors
    :param wv_dim, word vector dimension
    :param alpha, SIF weight parameter
    :param nc, n component
    :param stop_words, stop words

    return  sentence vector
    """
    # calculate probs and weight coefficient
    counter = CountVectorizer(stop_words=stop_words)
    X = counter.fit_transform(sentences)
    X = X.toarray()     # convert to numpy narray
    freqs = X.sum(axis=0)
    probs = freqs / np.sum(freqs)
    weight_coeff = alpha / (alpha + probs)

    # vocabulary vector matrix
    features = counter.get_feature_names()
    vocabulary_vector = np.zeros((len(features), wv_dim))
    for i in range(len(features)):
        vocabulary_vector[i] = wv[features[i]]

    # sentence vector
    sentence_len = X.sum(axis=1)
    sentence_vector = np.divide(np.multiply(X, weight_coeff).dot(vocabulary_vector).T, sentence_len).T

    # sentence vector minus primary component
    svd = TruncatedSVD(n_components=nc)
    svd.fit(sentence_vector)
    pc = svd.components_
    sentence_vector -= sentence_vector.dot(pc.T).dot(pc)

    return sentence_vector
