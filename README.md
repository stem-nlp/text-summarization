# Text Summarization

文本自动摘要

## 数据预处理

数据源：

1. [维基百科中文语料库](https://dumps.wikimedia.org/zhwiki/20190720/zhwiki-20190720-pages-articles.xml.bz2)
2. [汉语新闻语料库](https://github.com/Computing-Intelligence/datasource/blob/master/export_sql_1558435.zip)

其中，维基百科中文语料库核汉语新闻语料库用于词向量的训练，汉语新闻语料库用于自动摘要。
维基百科的信息结构比较复杂，可以[工具](https://github.com/attardi/wikiextractor)提取.

## 模型

### 词向量

#### 训练词向量

使用`gensim`训练词向量 TODO

#### 词向量测试

词向量的语义相似性 TODO

词向量的语义线性关系 TODO

词向量的可视化 TODO
使用`t-sne`进行高维向量的可视化

### 句向量

SIF方法进行句子的向量化 TODO

根据句子向量为每个句子赋予权重

KNN平滑

end-to-end模型

## 可视化

使用flask、bootstrap可视化 TODO
