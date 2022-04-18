#!/usr/bin/env python
import numpy as np
import glob
from sklearn.cluster import KMeans
from multiprocessing import Pool, Manager
from data_loader import data_preprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def find_elbow_point_index(x, y):
    # Normalize x and y to the unit space.
    x = np.array(x)
    min_x = min(x)
    max_x = max(x)
    scale_x = (x - min_x) / (max_x - min_x)
    y = np.array(y)
    min_y = min(y)
    max_y = max(y)
    scale_y = (y - min_y) / (max_y - min_y)
    # Get the line formed by the first and the last points: ax+by+1=0
    a = (scale_y[-1]-scale_y[0])/(scale_x[-1]*scale_y[0]-scale_x[0]*scale_y[-1])
    b = (scale_x[-1]-scale_x[0])/(scale_x[0]*scale_y[-1]-scale_x[-1]*scale_y[0])
    # Calculate perpendicular distances to the diagonal line.
    distances = abs(a*scale_x+b*scale_y+1)
    distances /= np.sqrt(a**2+b**2)
    return np.argmax(distances)

class KmeansBestK:

    def __init__(self, file_list, plot=True):
        self.file_list = file_list
        self.data, self.hbonds = data_preprocess(file_list)
        self.klist = np.arange(1, len(self.file_list)//5)
        self.inertia = Manager().list()
        self.inertia.extend([0]*len(self.klist))
        self.nprocessor = 20
        self.plot = plot

    def cluster(self, i):
        kmeans = KMeans(self.klist[i])
        kmeans.fit(self.data)
        self.inertia[i] = kmeans.inertia_
        return kmeans

    def cluster_over_klist(self):
        p = Pool(self.nprocessor)
        for i in range(len(self.klist)):
            p.apply_async(func=self.cluster, args=(i,))
        p.close()
        p.join()

    def find_bestkindex(self):
        self.cluster_over_klist()
        kindex = find_elbow_point_index(self.klist, self.inertia)
        if self.plot:
            plt.plot(self.klist, self.inertia, marker='.')
            plt.vlines(self.klist[kindex], plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
            plt.xlabel("Number of clusters")
            plt.ylabel("Sum of distances to cluster centers")
            plt.savefig('elbow_max%d_best%d.png' % (self.klist[-1], self.klist[kindex]))
        return kindex


def cluster_geometry(file_list):
    bestk_finder = KmeansBestK(file_list)
    bestk_index = bestk_finder.find_bestkindex()
    kmeans = bestk_finder.cluster(bestk_index)
    labels = kmeans.labels_
    return labels, bestk_finder.hbonds

if __name__=='__main__':
    file_list = glob.glob("*gjf")
    file_list.sort()
    labels, _ = cluster_geometry(file_list)
    for file, label in zip(file_list, labels):
        print(file, label)






