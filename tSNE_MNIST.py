#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:46:11 2021

@author: rodrigo
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets, manifold

# Read data
mnist = datasets.fetch_openml('mnist_784', version=1,
                              as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

X = mnist["data"]
t = mnist["target"]

# Instatiate tSNE
tsne = manifold.TSNE(n_components=3, random_state=42)

size_ = 2000
# fit a subset to reduce computing time
X_reduced_tsne = tsne.fit_transform(X[:size_])

# Prepare axis
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')
cmap = cm.get_cmap('jet', 10)

# Do scatter plot
scat = ax.scatter(*X_reduced_tsne.T, c=t[:size_], s=10, cmap=cmap,
                  edgecolors='None', alpha=0.8)

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
fig.colorbar(scat)

plt.show()
