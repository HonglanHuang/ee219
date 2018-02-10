import a
import b
import c
from sklearn.datasets import fetch_20newsgroups
import pylab as pl

r = [1, 2, 3, 5, 10, 20, 50, 100, 300]

def get_result(reduce_function, tfidf, labels):
    homo_list = []
    complete_list = []
    vscore_list = []
    rand_list = []
    mutual_list = []
    # using truncated svd
    for rank in r:
        truncated_svd = reduce_function(tfidf, rank)
        km = a.k_means_cluster(truncated_svd, k)
        result = a.get_result(km, labels)
        homo_list.append(result[0])
        complete_list.append(result[1])
        vscore_list.append(result[2])
        rand_list.append(result[3])
        mutual_list.append(result[4])
    pl.plot(r, homo_list, label = "homo")
    pl.plot(r, complete_list, label = "complete")
    pl.plot(r, vscore_list, label = "vscore")
    pl.plot(r, rand_list, label = "rand")
    pl.plot(r, mutual_list, label = "mutual")
    pl.legend(loc = "upper right")
    pl.show()

if __name__ == "__main__":
    k = 20
    dataset = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 42) 
    tfidf = a.get_TFIDF(dataset)
    labels = dataset.target
#    get_result(b.get_truncated_svd, tfidf)
#    get_result(b.get_nmf, tfidf, labels)
    
