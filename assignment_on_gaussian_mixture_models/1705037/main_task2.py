import imageio
import numpy as np
from sklearn.decomposition import PCA
from plot import *
import warnings
warnings.filterwarnings("ignore")
def min_max(X):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    return min, max

def normalization(X, min, max):
    X_normal = (X - min) / (max - min)
    return X_normal


#load data
data = []
with open('Assignment 3 Materials/data2D.txt') as f:
    for line in f.readlines():
        data.append([float(x) for x in line.split()])
# print(data)
X = np.array(data)
# min, max = min_max(X)
# X = normalization(X, min, max)

k_star=3

if X.shape[1] > 2:
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    V = pca.components_
    M = pca.mean_

    images = create_cluster_animation_for_nD(X2,k_star,V,M,X)
    imageio.mimsave('./gmm.gif', images, fps=1)
    exit(0)

images = create_cluster_animation(X, k_star)
imageio.mimsave('./gmm.gif', images, fps=1)