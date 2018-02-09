"the best r we got for LSI is 2" 

import a
import b
import pylab as pl
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer

data = a.retrieve_data()
labels = a.get_class(data)

# decomposition is the according dimension reduction function
def plot(decomposition, tfidf, r):
    truncated = decomposition(tfidf, r) 
    km = a.k_means_cluster(truncated)
    colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
    return truncated, colors

# set question_4_a to True to get the answer of 4(a)
if __name__ == "__main__":
    question_4_a = True
    question_4_b = True 

    r_lsi = 2
    r_nmf = 2
    dataset = a.retrieve_data()
    tfidf = a.get_TFIDF(dataset)

    if question_4_a:
        # using truncated svd
        first = pl.subplot(231)
        first.set_title('truncated svd')
        truncated, colors = plot(b.get_truncated_svd, tfidf, r_lsi)
        km = a.k_means_cluster(truncated)
        result = a.get_result(km, labels)
        a.print_result(result)
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using nmf 
        second = pl.subplot(232)
        second.set_title('nmf')
        truncated, colors = plot(b.get_nmf, tfidf, r_lsi)
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    # what function should we use to reduce dimension
    if question_4_b:
        # normalizing features
        truncated = b.get_nmf(tfidf, r_lsi) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        km = a.k_means_cluster(truncated)
        result = a.get_result(km, labels)
        a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(233)
        first.set_title('normalize festures')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)
    
        # using non-linear transformation
        non_linear = b.get_nmf(tfidf, r_nmf) 
        non_linear = FunctionTransformer(np.log1p).transform(non_linear)
        km = a.k_means_cluster(non_linear)
        result = a.get_result(km, labels)
        a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(234)
        first.set_title('non-linear')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using normalize first and then non-linear
        truncated = b.get_nmf(tfidf, r_lsi) 
        truncated = preprocessing.scale(truncated, with_mean = False)
        b_3_first = FunctionTransformer(np.log1p).transform(truncated)
        km = a.k_means_cluster(b_3_first)
        result = a.get_result(km, labels)
        a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(235)
        first.set_title('normalize first and then non-linear')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

        # using non_linear fist and then normalize
        b_3_second = preprocessing.scale(non_linear)
        km = a.k_means_cluster(b_3_second)
        km = a.k_means_cluster(non_linear)
        result = a.get_result(km, labels)
        a.print_result(result)
        colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
        first = pl.subplot(236)
        first.set_title('non-linear first and then normalize')
        pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    pl.show()





