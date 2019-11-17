import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist


def euclidean(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'euclidean')[0]


def manhattan(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'cityblock')[0]


def chebyshev(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'chebyshev')[0]


def cosine(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'cosine')[0]


def pearson(image1, image2):
    X = np.vstack([image1, image2])
    return np.corrcoef(X)[0][1]


def hamming(image1, image2):
    return np.shape(np.nonzero(image1 - image2)[0])[0]


def jaccard(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'jaccard')


def braycurtis(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'braycurtis')[0]


def mahalanobis(image1, image2):
    X = np.vstack([image1, image2])
    XT = X.T
    return pdist(XT, 'mahalanobis')


image1 = Image.open(
    'CT/Sample_1_20190818_131304/Sample_1_20190818_131304_3DFilter(Soft)_0002.png')
image2 = Image.open(
    'CT/Sample_1_20190818_131304/Sample_1_20190818_131304_3DFilter(Soft)_0003.png')

image2 = image2.resize(image1.size)
image1 = np.asarray(image1).flatten()
image2 = np.asarray(image2).flatten()


print("余弦相似度：%f" % (cosine(image1, image2)))
print("曼哈顿相似度：%f" % (manhattan(image1, image2)))
