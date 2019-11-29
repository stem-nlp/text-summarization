# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import pandas as pd
# from nltk.tokenize import StanfordTokenizer
import networkx as nx
import jieba
import re

path='/home/student/nlp_group_stem/nlp_p1/web/model/'
wv_model=KeyedVectors.load_word2vec_format(path + 'model_skipgram.model',binary=False)
word_dict=np.load(path+'words_dict.npy',allow_pickle=True)
embedding_size = 200


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    def length(self):
        return len(self.word_list)


def word_frequency(word_text, wordfreq_dict):
    if word_text in wordfreq_dict:
        return wordfreq_dict[word_text]
    else:
        return 1.0

def sentence2vec(sentence_list, embedding_size, wordfreq_dict, a=1e-3):
    sentence_processed = []
    for sentence in sentence_list:
        vecsum = np.zeros(embedding_size)
        len_sentence = sentence.length()
        for word in sentence.word_list:
            a_coef = a / (a + word_frequency(word.text, wordfreq_dict))
            vecsum = np.add(vecsum, np.multiply(a_coef, word.vector))
        vecsum = np.divide(vecsum, len_sentence)
        sentence_processed.append(vecsum)
    pram_components=min(len(sentence_list), 20)
    pca = PCA(n_components=pram_components)
    pca.fit(np.array(sentence_processed))
    u = pca.components_[0]
    u = np.multiply(u, np.transpose(u))
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)
    sentence_cut = []
    for vecsum in sentence_processed:
        sentence_cut.append(vecsum - np.multiply(u, vecsum))
    return sentence_cut

def get_title_vector(title):
    sentence_vec = []
    sentences = []
    sentence = jieba.cut(title)
    for word in sentence:
        try:
            vec = wv_model[word]
        except KeyError:
            vec = np.zeros(embedding_size)
        sentence_vec.append(Word(word, vec))
    sentence_unit = Sentence(sentence_vec)
    sentences.append(sentence_unit)
    sentence_vectors = sentence2vec(sentences, embedding_size, word_dict)
    return sentence_vectors

def knn_smooth_nearby(scores):
    knn_scores=[]
    n=len(scores)
    if((n-2)>0):
        knn_scores.append((scores[0]+scores[1])/2)
        for i in range(1,n-1):
            knn_scores.append((scores[i-1]+scores[i]+scores[i+1])/3)
        knn_scores.append((scores[n-1]+scores[n-2])/2)
    else:
        knn_scores=scores
    return knn_scores

def model(X, y):
    return [(Xi, yi) for Xi, yi in zip(X, y)]

def predict(x, s_set, k=3):
    most_similars = sorted(model(s_set, s_set), key=lambda xi: abs(xi[0]-x))[:k]
    return np.mean([i[1] for i in most_similars])

def knn_Smooth(scores,k=3):
    knn_scores=[]
    for s in scores:
        t=predict(s,scores,k)
        print("t",t)
        knn_scores.append(t)
    return knn_scores

def get_summary(input_title,input_body,rate=0.3):
    summary=""
    title = input_title
    article = input_body
    rate=float(rate)
    if pd.isnull(article):
        if pd.notnull(title):
            summary = title
        else:
            summary = 'no information'
    else:
        article = re.split('！|。|？', article.replace('\r\r\n', ''))
        sentences = []
        raw_sentences = []
        for sentence in article:
            if sentence != '':
                sentence_vec = []
                raw_sentence = sentence.replace(' ', '') + '。'
                if raw_sentence not in raw_sentences:
                    raw_sentences.append(raw_sentence)
                    sentence = jieba.cut(sentence)
                    for word in sentence:
                        try:
                            vec = wv_model[word]
                        except KeyError:
                            vec = np.zeros(embedding_size)
                        sentence_vec.append(Word(word, vec))
                    sentence_unit = Sentence(sentence_vec)
                    if(sentence_unit.length()>0):
                        sentences.append(sentence_unit)
        # 利用TIF求出句子向量集合-sentence_vectors，全文向量-paragragh_vector，标题向量-title_vectors
        sentence_vectors = sentence2vec(sentences, embedding_size, word_dict)
        paragragh_vector=np.mean(sentence_vectors,axis=0)
        title_vectors=get_title_vector(title)

        #求出每个句子向量与标题向量、全文向量的相似度，相加得到每个句子的相似度得分。
        scores=[]
        for sv in sentence_vectors:
            score=cosine_similarity(sv.reshape(1, embedding_size),paragragh_vector.reshape(1, embedding_size))
            +cosine_similarity(sv.reshape(1, embedding_size),title_vectors[0].reshape(1, embedding_size))
            scores.append(score)
        scores[0] += 0.2
        scores[len(scores)-1] += 0.2
        # 利用KNN对句子相似度得分进行平滑
        # knn_scores=knn_Smooth(scores,3)
        # 相邻三个元素求平均值
        knn_scores = knn_smooth_nearby(scores)
        # 整理求出topn个句子
        ranked_sentences = sorted(((knn_scores[i], s) for i, s in enumerate(raw_sentences)), reverse=True)
        top_sentense = {}
        p_num = max(int(len(ranked_sentences) * rate), 2)
        for top_num in range(min(len(ranked_sentences), p_num)):
            for sentence in article:
                if (sentence == (ranked_sentences[top_num][1]).replace('。', '')):
                    top_sentense[article.index(sentence)]=ranked_sentences[top_num][1]
        sort_sentense=dict(sorted(top_sentense.items(), key=lambda item:item[0]))
        for s in sort_sentense.values():
            summary+=s
        print("result:",summary)
    return summary


