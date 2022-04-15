import numpy as np
import glob
from sklearn.cluster import KMeans
from data_loader import data_preprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from kneed import KneeLocator

file_list = glob.glob("*.com")


data = data_preprocess(file_list)
K = np.arange(1, len(file_list)+1, step=5)


def cluster(i, inertia):
    kmeans = KMeans(K[i], init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(data)
    inertia[i] = kmeans.inertia_

def cluster_multiple_k():
    inertia = Manager().list()
    inertia.extend([0]*len(K))
    p = Pool(28)
    for i in range(len(K)):
        p.apply_async(func=cluster, args=(i, inertia))
    p.close()
    p.join()

    kn = KneeLocator(K, inertia, curve='convex', direction='decreasing')
    print(kn.knee)
    plt.plot(K, inertia, marker='.')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig('fig.png')








