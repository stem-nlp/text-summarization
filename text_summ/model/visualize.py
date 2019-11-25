# encoding: utf8

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def tsne_plot(wv, size=1000):
    labels = []
    tokens = []

    counter = 0
    for word in wv.vocab:
        tokens.append(wv[word])
        labels.append(word)

        counter += 1
        if counter > size:
            break

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
