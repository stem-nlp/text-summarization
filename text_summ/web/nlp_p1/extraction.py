# # -*- coding: utf-8 -*-
#
# from gensim.models import KeyedVectors
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.stats import pearsonr
# import pandas as pd
# # from nltk.tokenize import StanfordTokenizer
# import networkx as nx
# import jieba
# import re
#
# path='D:/nlpProject/nlp_p1/model/'
# wv_model=KeyedVectors.load_word2vec_format(path + 'model_skipgram_2_10.model',binary=False)
# word_dict=np.load(path+'words_dict.npy',allow_pickle=True)
# embedding_size = 200
#
#
# class Word:
#     def __init__(self, text, vector):
#         self.text = text
#         self.vector = vector
#
#
# class Sentence:
#     def __init__(self, word_list):
#         self.word_list = word_list
#
#     def length(self):
#         return len(self.word_list)
#
#
# def word_frequency(word_text, wordfreq_dict):
#     if word_text in wordfreq_dict:
#         return wordfreq_dict[word_text]
#     else:
#         return 1.0
#
#
# def sentence2vec(sentence_list, embedding_size, wordfreq_dict, a=1e-3):
#     sentence_processed = []
#     for sentence in sentence_list:
#         vecsum = np.zeros(embedding_size)
#         len_sentence = sentence.length()
#         for word in sentence.word_list:
#             a_coef = a / (a + word_frequency(word.text, wordfreq_dict))
#             vecsum = np.add(vecsum, np.multiply(a_coef, word.vector))
#         vecsum = np.divide(vecsum, len_sentence)
#         sentence_processed.append(vecsum)
#     pram_components=min(len(sentence_list), 15)
#     pca = PCA(n_components=pram_components)
#     pca.fit(np.array(sentence_processed))
#     u = pca.components_[0]
#     u = np.multiply(u, np.transpose(u))
#     if len(u) < embedding_size:
#         for i in range(embedding_size - len(u)):
#             u = np.append(u, 0)
#     sentence_cut = []
#     for vecsum in sentence_processed:
#         sentence_cut.append(vecsum - np.multiply(u, vecsum))
#     return sentence_cut
#
#
# df_articles = pd.read_csv(path + 'testdata/news_title.csv', encoding='utf-8')
#
# for i in range(len(df_articles.id)):
#     title = df_articles.loc[i, 'title']
#     article = df_articles.loc[i, 'paragraph']
#     if pd.isnull(article):
#         if pd.notnull(title):
#             df_articles.loc[i, 'summary'] = title
#         else:
#             df_articles.loc[i, 'summary'] = 'no information'
#     else:
#         article =re.split('！|。|？',article.replace('\r\r\n', ''))
#         sentences = []
#         raw_sentences = []
#         for sentence in article:
#             if sentence != ' ':
#                 sentence_vec = []
#                 raw_sentence = sentence.replace(' ', '') + '。'
#                 if raw_sentence not in raw_sentences:
#                     raw_sentences.append(raw_sentence)
#                     # sentence = sentence.split() + ['。']
#                     sentence = jieba.cut(sentence)
#                     for word in sentence:
#                         try:
#                             vec = wv_model[word]
#                         except KeyError:
#                             vec = np.zeros(embedding_size)
#                         sentence_vec.append(Word(word, vec))
#                         sentence_unit = Sentence(sentence_vec)
#                         sentences.append(sentence_unit)
#         sentence_vectors = sentence2vec(sentences, embedding_size, word_dict)
#         len_sentences = len(sentence_vectors)
#         similarity_matrix = np.zeros([len_sentences, len_sentences])
#         for i in range(len(sentences)):
#             for j in range(len(sentences)):
#                 if i != j:
#                     similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, embedding_size),
#                                                                 sentence_vectors[j].reshape(1, embedding_size))[0][0]
#         nx_graph = nx.from_numpy_array(similarity_matrix)
#         scores = nx.pagerank(nx_graph)
#         ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(raw_sentences)), reverse=True)
#         df_articles.loc[i, 'summary'] = ''
#         top_sentense = {}
#         result=""
#         for top_num in range(min(len(ranked_sentences), 2)):
#             df_articles.loc[i, 'summary'] = df_articles.loc[i, 'summary'] + ranked_sentences[top_num][1]
#             for sentence in article:
#                 if(sentence.replace('。', '').replace('？', '').replace('！', '')==
#                    (ranked_sentences[top_num][1]).replace('。', '').replace('？', '').replace('！', '')):
#                     top_sentense[article.index(sentence)]=ranked_sentences[top_num][1]
#         sort_sentense=dict(sorted(top_sentense.items(), key=lambda item:item[0]))
#         for s in sort_sentense.values():
#             result+=s
#         print("result:",result)
# df_articles.to_csv(path + 'testdata/processed_new3.csv', encoding='utf_8_sig')
#
# class testcalss:
#     def __init__(self,t):
#         self.t=t
#
#     def test(self):
#         print("test,sdm")
#
#
#
# # 选取的句数应该为相对值不是绝对值，目前统一为2
# # 可计算title与各句间的余弦相似度，对抽取的句子进行筛选，问题是部分新闻题目与内容不符
# # 对于微博体等语义分散的文章应该选用生成式算法，或者在判断cos_similarity小于一定threshold后直接用标题
# # 针对新闻有专属stop_words词库，如编译/来源/免责声明/文/摄影/访问等词是常见词，应该去除
# # 可根据句子所处位置或者主题词出现频率提取
# # 句子可设定threashold,有些仅包含人名的短句可能出现在摘要里。
#
#
