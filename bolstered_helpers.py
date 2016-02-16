#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.utils import check_random_state
from mean_min_dist import _mean_min_dists_feature_wise
from mean_min_dist import _mean_min_dists
import logging


def _bolstered_blobs(X, y, cov, n_montecarlo=10, random_state=None):
    """Return n_montecarlo multivariate normal points for each point in X

    :param X:
    :param y:
    :param cov:
    :param n_montecarlo:
    :param random_state:
    """
    generator = check_random_state(random_state)
    n, p = X.shape
    shape = np.shape(cov)
    if len(shape) == 1:
        cov = np.diag(cov)
    assert np.shape(cov) == (p, p)
    X_out = np.zeros((n * n_montecarlo, p))
    y_out = np.array(list(y) * n_montecarlo, dtype=np.int)
    stop = 0
    for i, mean in enumerate(X):
        start, stop = stop, stop + n_montecarlo
        X_out[start:stop, :] =\
            generator.multivariate_normal(mean, cov, n_montecarlo)
    assert y_out.sum() == n_montecarlo * y.sum()
    return (X_out, y_out)


def bolstered_blobs(X, y, n_montecarlo=10,
                    new_bolster=True, random_state=None):
    """For each sample in X, generate n_montecarlo bolstered samples.

    :param X:
    :param y:
    :param n_montecarlo:
    :param new_bolster:
    :param random_state:
    """
    assert (np.unique(y).astype(np.int) ==
            np.array([0, 1], dtype=np.int)).all()
    X0 = X[y == 0, ]
    y0 = y[y == 0]
    X1 = X[y == 1, ]
    y1 = y[y == 1]
    if new_bolster:
        sig0, sig1 = _mean_min_dists_feature_wise(X, y)
    else:
        sig0, sig1 = _mean_min_dists(X, y)
    logger = logging.getLogger(__name__)
    msg = "new_bolster: {}\n sig0: {}\n sig1: {}".format(
        new_bolster, sig0, sig1)
    logger.debug(msg)

    X0_out, y0_out = _bolstered_blobs(X0, y0, sig0, n_montecarlo,
                                      random_state=random_state)
    X1_out, y1_out = _bolstered_blobs(X1, y1, sig1, n_montecarlo,
                                      random_state=random_state)
    return (np.vstack((X0_out, X1_out)), np.concatenate((y0_out, y1_out)))


def bolstered_blobs_partial(X, y, Xp, yp, n_montecarlo=10,
                            new_bolster=True, random_state=None):
    """
    For each sample in Xp, generate n_montecarlo bolstered samples using
    sigmas generated from X, y.
    """
    assert (np.unique(y).astype(np.int) ==
            np.array([0, 1], dtype=np.int)).all()
    if new_bolster:
        sig0, sig1 = _mean_min_dists_feature_wise(X, y)
    else:
        sig0, sig1 = _mean_min_dists(X, y)
    Xp0 = Xp[yp == 0, ]
    yp0 = yp[yp == 0]
    Xp1 = Xp[yp == 1, ]
    yp1 = yp[yp == 1]

    Xp0_out, yp0_out = _bolstered_blobs(Xp0, yp0, sig0, n_montecarlo,
                                        random_state=random_state)
    Xp1_out, yp1_out = _bolstered_blobs(Xp1, yp1, sig1, n_montecarlo,
                                        random_state=random_state)
    return (np.vstack((Xp0_out, Xp1_out)), np.concatenate((yp0_out, yp1_out)))


def test_bolstered_blobs():
    from make_datasets import make_dataset
    n1, n2 = 5, 5
    n_features = 5
    n_informative = 2
    class_sep = 2
    X, y = make_dataset(n1, n2, n_features, n_informative,
                        class_sep, shuffle=True, random_state=None)
    n_montecarlo = 10
    X_, y_ = bolstered_blobs(X, y, n_montecarlo)
    # print(np.bincount(y))
    # print(np.bincount(y_))
    # print(X)
    # print(y)
