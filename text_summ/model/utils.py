# encoding: utf8


def load_stopwords(filepath):
    """
    Load stop words.

    :param filepath
    :return stopwords list.
    """
    with open(filepath) as f:
        stopwords = f.read()
        return stopwords.split('\n')
