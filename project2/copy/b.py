import a
import numpy as np
from scipy.sparse.linalg import svds
import pylab as pl

r = [1, 2, 3, 5, 10, 20, 50, 100, 300]
data = a.retrieve_data()

# plot the variance v.s. r
def get_svd(tfidf):
    number = 1000
    U, s, V = svds(tfidf, number)
#    print tfidf.shape
#    print U.shape 
#    print s.shape 
#    print V.shape 
    s = s[::-1]
    s = map(lambda(x): x * x, s)
    for i in range(1, number):
        s[i] += s[i - 1]
    s = map(lambda(x): x / s[number - 1], s)
    x = range(1, number + 1)
    pl.plot(x, s)
    pl.show()
    print type(s)
    return U, s, V

# use LSI to do the reduction
def use_LSI(U, s, V):
    r = [1000]
    U = np.array(U) 
    V = np.array(V) 
    s = np.array(s) 
    labels = a.get_class(data)
    homo_list = []
    complete_list = []
    vscore_list = []
    rand_list = []
    mutual_list = []
    for rank in r:
        # get the reducted U, s, V
        r_U = U[:, 0:rank] 
        r_s = s[0:rank]
        x = r_U.dot(np.diag(r_s))
        km = a.k_means_cluster(x) 
        result = a.get_result(km, labels)
        homo_list.append(result[0])
        complete_list.append(result[1])
        vscore_list.append(result[2])
        rand_list.append(result[3])
        mutual_list.append(result[4])
    # plot
    pl.plot(r, homo_list, label = "homo")
    pl.plot(r, complete_list, label = "complete")
    pl.plot(r, vscore_list, label = "vscore")
    pl.plot(r, rand_list, label = "rand")
    pl.plot(r, mutual_list, label = "mutual")
    pl.legend(loc = "upper left")
    pl.show()

if __name__ == "__main__":
    data = a.retrieve_data()
    tfidf = a.get_TFIDF(data)
    U, s, V = get_svd(tfidf)
    use_LSI(U, s, V)


