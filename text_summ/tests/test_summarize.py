# encoding: utf8

import logging
import pandas as pd
from text_summ import config
from text_summ.model.model import summarize

_logger = logging.getLogger()


def test_summarize():
    dataset = pd.read_csv(config.NEWS_CORPUS_DATASET_FILEPATH)
    record = dict(dataset.iloc[11, :])
    title, content = record['title'], record['content']

    summary = summarize(content, title)
    print('\n'.join(['title: '+title, 'content: ' + content, 'summary: '+summary]))
