# plot_2d_classifier_functions.py

# Funktion zum Visualisieren der Entscheidungsgrenzen von 2D-Classifiern
#
# Eingabeparameter: 
#  - clf: Instanz des Klassifikators -> bspw.: clf = nc (für nc = NearestCentroid())
#  - X: Merkmalsmatrix
#  - y: Klassenvektor



# Darstellen der Entscheidungsgrenzen eines 2D-Classifiers (Code von hier: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py):
def plot_2d_seperator(clf, X, y):
    from matplotlib.colors import ListedColormap
    import numpy as np
    import matplotlib.pyplot as plt
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    h = .02  # step size in the mesh
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Y_pred = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Y_pred = Y_pred.reshape(xx1.shape)
#    plt.figure()
    plt.pcolormesh(xx1, xx2, Y_pred, cmap=cmap_light, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)    
    plt.xlabel('Merkmal A')
    plt.ylabel('Merkmal B')
