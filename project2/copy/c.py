"the best r we got for LSI is 2" 

import a
import b
import pylab as pl

# decomposition is the according dimension reduction function
def plot(decomposition, tfidf, r):
    truncated = decomposition(tfidf, r) 
    km = a.k_means_cluster(truncated)
    colors = map(lambda(x): 'r' if x == 0 else 'b', km.labels_)
    return truncated, colors

if __name__ == "__main__":
    r_lsi = 2
    r_nmf = 2
    dataset = a.retrieve_data()
    tfidf = a.get_TFIDF(dataset)

    # using truncated svd
    first  = pl.subplot(211)
    first.set_title('truncated svd')
    truncated, colors = plot(b.get_truncated_svd, tfidf, r_lsi)
    pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    # using nmf 
    second = pl.subplot(212)
    second.set_title('nmf')
    truncated, colors = plot(b.get_nmf, tfidf, r_lsi)
    pl.scatter(truncated[:, 0:1], truncated[:, 1:2], c = colors)

    pl.show()
    




