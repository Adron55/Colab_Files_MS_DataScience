from scipy.cluster.hierarchy import dendrogram
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

def load_dat(filename):
    mydata = pd.read_csv(filename, sep=' ', names=['x', 'y'], header=None)
    return mydata


def plot_points_classe_2d(X, Y, titre=' '):
    # setup marker generator and color map
    markers = ('o', '^', 's', 'x', 'v', 'h')
    colors = ('red', 'blue', 'green', 'gray', 'cyan', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(Y))])
    plt.figure()
    for idx, cl in enumerate(np.unique(Y)):
        plt.scatter(X[Y == cl, 0], X[Y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label='classe {}'.format(cl))
    # plt.show()
    plt.title(titre)
    plt.legend(loc='best')


def plot_dendogram(model, label_points=None):
    plt.style.use('ggplot')
    dendro = []
    for a, b in model.children_:
        dendro.append([a, b, float(len(dendro)+1), len(dendro)+1])
        # le dernier coefficient devrait contenir le nombre de feuilles
        # dependant de ce noeud  et non le dernier indice
        # de même, le niveau (3eme colonne) ne devrait pas etre
        # le nombre de noeud  mais la distance de Ward
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(dendro, color_threshold=1, labels=label_points,
               show_leaf_counts=True, ax=ax, orientation="right")

def generate_toy_datasets():
    n_points = 20 #nombre de points par classe
    X_toy, y_toy = make_blobs(n_samples=n_points,centers=3)

    #Génération des jeux de données

    # 3 clusters
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    #Anisotropic
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    X_aniso=np.vstack([X_aniso[y==0], X_aniso[y==1]])
    y_aniso=np.hstack([y[y==0], y[y==1]])
    # Unequal Variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    # Different sizes
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_filtered = np.hstack([y[y == 0][:500],  y[y == 1][:100], y[y==2][:10]])

    return X_toy, y_toy, X, y, X_aniso, y_aniso, X_varied, y_varied, X_filtered, y_filtered
data = generate_toy_datasets()
