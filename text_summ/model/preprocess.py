# encodint: utf8

"""
数据预处理

数据源：
1. 维基百科中文语料库
2. 汉语新闻语料库
"""

import re
import jieba
import logging
import pandas as pd

from text_summ import config

_logger = logging.getLogger('app')


STRIP_SYMBOLS = list(set(['，', '。', '《', '》', ' ', '!', '！' '、', '.', '；', '~', '、',
                          '”', '“', ';', '%', '\n', '\\n', '（', '）', '/', '——', '：',
                          '…', '@', '！', '？', ' ,', ' :', '】', '’', ')', '(', '／', '·',
                          '-', '?']))


def cleanup_data(s):
    s = str(s)
    news_lines = []
    for line in re.split(r'\\n|\n', s):
        line = re.sub(re.compile(r'\\u\d{4}'), '', '%r' % line)[1:-1]
        if line:
            news_lines.append(line.strip())

    return '\n'.join(news_lines)


def cut(s):
    return list(jieba.cut(s))


def _strip_symbol_filter(x):
    return False if x in STRIP_SYMBOLS else True


def gen_train_corpus(df: pd.DataFrame, filepath):
    with open(filepath, 'w') as f:
        for cols in df.values:
            for col in cols:
                tokens = cut(str(col))
                tokens = list(filter(_strip_symbol_filter, tokens))
                f.write(' '.join(tokens) + '\n')

    return None


def drop_na_rows(dataset, columns=('title', 'content')):
    """
    drop na rows.

    :param dataset, pandas DataFrame
    :return drop na dataset, pandas DataFrame
    """

    for col in columns:
        dataset = dataset[dataset[col].notnull()]

    return dataset


def process_news_corpus_data():
    """
    process news corpus data, and generate gensim training corpus.
    """
    news_dataset = pd.read_csv(config.NEWS_CORPUS_ORIGIN_FILEPATH)
    columns = ['title', 'content']

    # extract column data.
    df = news_dataset.loc[:, columns]

    # drop na rows
    df = drop_na_rows(df, columns=columns)

    # cleanup data
    new_array = []
    for cols in df.values:
        new_array.append([cleanup_data(col) for col in cols])

    # output to file
    news_corpus_dataset = pd.DataFrame(new_array, columns=columns)
    news_corpus_dataset.to_csv(config.NEWS_CORPUS_DATASET_FILEPATH, index=False)

    # generate gensim training corpus
    gen_train_corpus(news_corpus_dataset, config.NEWS_CORPUS_FILEPATH)


def process_corpus_data():
    process_news_corpus_data()
