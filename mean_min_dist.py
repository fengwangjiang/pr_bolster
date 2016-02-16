#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi


def _mean_min_dists_feature_wise(X, y, k=3, percentile=0.5):
    """Calculate mean of min distances for each feature, used for new bolstering
    methods.

    :param X: nxD dataset
    :param y: label
    :param k: number of neighbors
    :param percentile:
    :output : 2xD, the value at ith row, and jth column is the mean of the
    minimum distances of each sample in jth feature for the ith class.
    """
    X1, X2 = X[y == 0], X[y == 1]
    p = X.shape[1]
    n1, n2 = X1.shape[0], X2.shape[0]
    dist1np = np.zeros((n1+n2, p))
    dist2np = np.zeros((n1+n2, p))
    # calculate distances for each feature dimension
    for i in range(p):
        nbrs1 = NearestNeighbors(
            n_neighbors=k+1, algorithm='ball_tree', n_jobs=-2)
        nbrs1.fit(X1[:, i].reshape(n1, -1))
        distances1, indices1 = nbrs1.kneighbors(X[:, i].reshape(n1+n2, -1))

        nbrs2 = NearestNeighbors(
            n_neighbors=k+1, algorithm='ball_tree', n_jobs=-2)
        nbrs2.fit(X2[:, i].reshape(n2, -1))
        distances2, indices2 = nbrs2.kneighbors(X[:, i].reshape(n1+n2, -1))

        tmp1 = np.mean(distances1[:, 1:], axis=1)
        tmp2 = np.mean(distances2[:, 1:], axis=1)
        dist1np[:, i] = tmp1
        dist2np[:, i] = tmp2
    d1 = np.mean(dist1np, axis=0)
    d2 = np.mean(dist2np, axis=0)
    cp = chi.ppf(percentile, 1)
    sig1 = d1/cp
    sig2 = d2/cp
    return np.vstack((sig1, sig2))


def _mean_min_dists(X, y, k=1, percentile=0.5):
    """Calculate mean min distances used for original bolstering sig.

    :param X:
    :param y:
    :param k:
    :param percentile:
    """
    X1, X2 = X[y == 0], X[y == 1]
    p = X.shape[1]

    nbrs1 = NearestNeighbors(
        n_neighbors=k+1, algorithm='ball_tree', n_jobs=-2).fit(X1)
    distances1, indices1 = nbrs1.kneighbors(X)
    # distances1, indices1 = nbrs1.kneighbors(X1)
    nbrs2 = NearestNeighbors(
        n_neighbors=k+1, algorithm='ball_tree', n_jobs=-2).fit(X2)
    distances2, indices2 = nbrs2.kneighbors(X)
    # distances2, indices2 = nbrs2.kneighbors(X2)
    d1 = np.mean(distances1[:, 1:])
    d2 = np.mean(distances2[:, 1:])

    cp = chi.ppf(percentile, p)
    sig1 = d1/cp * np.ones(p)
    sig2 = d2/cp * np.ones(p)
    return np.vstack((sig1, sig2))
