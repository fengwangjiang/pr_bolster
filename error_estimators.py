#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import logging
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import (f_classif, SelectKBest)
from bolstered_helpers import (bolstered_blobs_partial, bolstered_blobs)
from sklearn.utils import check_random_state


def cross_validation_score(clf, X, y, n_feat_selected,
                           n_folds=10, random_state=0):
    """Return cross validation error.

    For classifier clf, do feature selection for each fold, and estimate
    the classfication error.

    :param clf:
    :param X:
    :param y:
    :param n_feat_selected:
    :param n_folds:
    :param random_state:
    """
    logger = logging.getLogger(__name__)
    n_samples = len(y)
    #  kf = cross_validation.KFold(n_samples, n_folds=n_folds, indices=True)
    kf = cross_validation.KFold(n_samples, n_folds=n_folds)
    errs0 = []
    errs90 = []
    for train_index, test_index in kf:
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        msg = "cross validation {} folds, train size {}, test size\
        {}".format(n_folds, len(y_train), len(y_test))
        logger.debug(msg)

        univariate_filter = SelectKBest(f_classif, k=n_feat_selected)\
            .fit(X_train, y_train)
        X_train_r = univariate_filter.transform(X_train)
        X_test_r = univariate_filter.transform(X_test)
        clf.fit(X_train_r, y_train)
        err0 = 1 - clf.score(X_test_r, y_test)
        err_cv_resub = 1 - clf.score(X_train_r, y_train)
        err90 = 1.0/n_folds * err_cv_resub + (1-1.0/n_folds) * err0

        errs0.append(err0)
        errs90.append(err90)
    return np.array(errs0).mean(), np.array(errs90).mean()


def bolster_cross_validation(clf, X, y, n_feat_selected,
                             bolster_before_feat_selection=True,
                             new_bolster=True, n_folds=10, random_state=0):
    """bolstered cross validation, for each fold left over, we generate a few
    more samples around them.

    :param clf:
    :param X:
    :param y:
    :param n_feat_selected:
    :param bolster_before_feat_selection:
    :param new_bolster:
    :param n_folds:
    :param random_state:
    """
    n_samples = len(y)
    #  kf = cross_validation.KFold(n_samples, n_folds=n_folds, indices=True)
    kf = cross_validation.KFold(n_samples, n_folds=n_folds)
    errs = []
    for train_index, test_index in kf:
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        univariate_filter = SelectKBest(f_classif, k=n_feat_selected)\
            .fit(X_train, y_train)
        X_train_r = univariate_filter.transform(X_train)
        clf.fit(X_train_r, y_train)

        if bolster_before_feat_selection:
            X_, y_ = bolstered_blobs_partial(X, y, X_test, y_test,
                                             new_bolster=new_bolster,
                                             random_state=random_state)
            X_r = univariate_filter.transform(X_)
            err = 1 - clf.score(X_r, y_)
        else:
            Xr = univariate_filter.transform(X)
            X_test_r = univariate_filter.transform(X_test)
            X_, y_ = bolstered_blobs_partial(Xr, y, X_test_r, y_test,
                                             new_bolster=new_bolster,
                                             random_state=random_state)
            err = 1 - clf.score(X_, y_)
        errs.append(err)
    err_bolster_cv10 = np.array(errs).mean()
    return err_bolster_cv10


def bootstrap(n, n_iter, train_size, test_size, random_state):
    """bootstrap

    :param n:
    :param n_iter:
    :param train_size:
    :param test_size:
    :param random_state:
    """
    rng = check_random_state(random_state)
    for i in range(n_iter):
        # random partition
        permutation = rng.permutation(n)
        ind_train = permutation[:train_size]
        ind_test = permutation[train_size:train_size + test_size]
        # bootstrap in each split individually
        train = rng.randint(0, train_size, size=(train_size,))
        test = rng.randint(0, test_size, size=(test_size,))
        yield ind_train[train], ind_test[test]


def bootstrap_score(clf, X, y, n_feat_selected, n_iter=100, random_state=0):
    """
    bootstrap error 0 and 632 for classifier clf, for each iteration we need
    to do feature selection.
    ------------------------
    """
    n_samples = len(y)
    train_size = np.int(np.ceil(n_samples * 0.5))
    test_size = n_samples - train_size
    assert (train_size > 0)
    assert (test_size > 0)
    bs = bootstrap(n_samples, n_iter, train_size, test_size, random_state)

    errs0 = []
    errs632 = []
    for train_index, test_index in bs:
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        if len(np.unique(y_train)) < 2:
            continue
        univariate_filter = SelectKBest(f_classif, k=n_feat_selected)\
            .fit(X_train, y_train)
        X_train_r = univariate_filter.transform(X_train)
        X_test_r = univariate_filter.transform(X_test)
        clf.fit(X_train_r, y_train)
        err0 = 1 - clf.score(X_test_r, y_test)
        err_bs_resub = 1 - clf.score(X_train_r, y_train)
        err632 = (1-0.632) * err_bs_resub + 0.632 * err0
        errs0.append(err0)
        errs632.append(err632)
    return np.array(errs0).mean(), np.array(errs632).mean()


def _error_calculation(clf, X, y, bolster=False, random_state=None):
    if bolster:
        X_, y_ = bolstered_blobs(X, y, new_bolster=True,
                                 random_state=random_state)
        err_new = 1 - clf.score(X_, y_)
        X_, y_ = bolstered_blobs(X, y, new_bolster=False,
                                 random_state=random_state)
        err_old = 1 - clf.score(X_, y_)
        return err_old, err_new
    else:
        return 1 - clf.score(X, y)


def bolster_resub(clf, X, y, feat_selector,
                  bolster_before_feat_selection=True, new_bolster=True,
                  random_state=None):
    """docstring for bolster_resub

    :param clf:
    :param X: nxD
    :param y: (n*n_montecarlo)xD
    :param feat_selector:
    :param bolster_before_feat_selection:
    :param new_bolster:
    :param random_state:
    X_: (n*n_montecarlo)xD
    X_r: (n*n_montecarlo)xn_feat_selected
    Xr: nxn_feat_selected
    Xr_: (n*n_montecarlo)xn_feat_selected
    """
    if bolster_before_feat_selection:
        X_, y_ = bolstered_blobs(X, y, new_bolster=new_bolster,
                                 random_state=random_state)
        X_r = feat_selector.transform(X_)
        err = 1 - clf.score(X_r, y_)
    else:
        Xr = feat_selector.transform(X)
        Xr_, y_ = bolstered_blobs(Xr, y, new_bolster=new_bolster,
                                  random_state=random_state)
        err = 1 - clf.score(Xr_, y_)
    return err


def error_estimators(clf, X, y, n_feat_selected, test_size=0.9,
                     random_state=None):
    """Error estimators"""
    logger = logging.getLogger(__name__)
    logger.debug("Start error_estimator")
    print("Start error_estimator")
    sss = StratifiedShuffleSplit(y, 1, test_size=test_size,
                                 random_state=random_state)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    univariate_filter = SelectKBest(f_classif,
                                    k=n_feat_selected).fit(X_train, y_train)
    X_train_r = univariate_filter.transform(X_train)
    X_test_r = univariate_filter.transform(X_test)

    clf.fit(X_train_r, y_train)

    err_resub = _error_calculation(
        clf, X_train_r, y_train, random_state=random_state)
    err_true = _error_calculation(
        clf, X_test_r, y_test, random_state=random_state)
    # err_bresub_old, err_bresub_new =\
    # _error_calculation(clf, X_train_r, y_train, bolster=True,
    # random_state=random_state)
    # err_cv10 = 1 - cross_val_score(clf, X_train_r, y_train, cv=10).mean()
    err_bresub_old_D = bolster_resub(clf, X_train, y_train,
                                     feat_selector=univariate_filter,
                                     bolster_before_feat_selection=True,
                                     new_bolster=False,
                                     random_state=random_state)
    err_bresub_old_d = 0
    #  err_bresub_old_d = bolster_resub(clf, X_train, y_train,
                                     #  feat_selector=univariate_filter,
                                     #  bolster_before_feat_selection=False,
                                     #  new_bolster=False,
                                     #  random_state=random_state)
    err_bresub_new_D = bolster_resub(clf, X_train, y_train,
                                     feat_selector=univariate_filter,
                                     bolster_before_feat_selection=True,
                                     new_bolster=True,
                                     random_state=random_state)
    err_bresub_new_d = 0
    #  err_bresub_new_d = bolster_resub(clf, X_train, y_train,
                                     #  feat_selector=univariate_filter,
                                     #  bolster_before_feat_selection=False,
                                     #  new_bolster=True,
                                     #  random_state=random_state)
    err_bcv10_old_d = 0
    #  err_bcv10_old_d =\
        #  bolster_cross_validation(clf, X_train, y_train,
                                 #  n_feat_selected=n_feat_selected,
                                 #  bolster_before_feat_selection=False,
                                 #  new_bolster=False, n_folds=10,
                                 #  random_state=random_state)
    err_bcv10_new_d = 0
    #  err_bcv10_new_d =\
        #  bolster_cross_validation(clf, X_train, y_train,
                                 #  n_feat_selected=n_feat_selected,
                                 #  bolster_before_feat_selection=False,
                                 #  new_bolster=True, n_folds=10,
                                 #  random_state=random_state)
    err_bcv10_old_D = 0
    #  err_bcv10_old_D =\
        #  bolster_cross_validation(clf, X_train, y_train,
                                 #  n_feat_selected=n_feat_selected,
                                 #  bolster_before_feat_selection=True,
                                 #  new_bolster=False, n_folds=10,
                                 #  random_state=random_state)
    err_bcv10_new_D = 0
    #  err_bcv10_new_D =\
        #  bolster_cross_validation(clf, X_train, y_train,
                                 #  n_feat_selected=n_feat_selected,
                                 #  bolster_before_feat_selection=True,
                                 #  new_bolster=True, n_folds=10,
                                 #  random_state=random_state)
    err_cv10, err_cv10_90 =\
        cross_validation_score(clf, X_train, y_train,
                               n_feat_selected=n_feat_selected,
                               n_folds=10, random_state=random_state)
    err_bs0, err_bs632 = bootstrap_score(clf, X_train, y_train,
                                         n_feat_selected=n_feat_selected,
                                         n_iter=100,
                                         random_state=random_state)

    print("Finish error_estimator")
    logger.debug("Finish error_estimator")
    return (err_true, err_resub, err_bresub_old_d, err_bresub_new_d,
            err_bresub_old_D, err_bresub_new_D, err_cv10, err_bs0, err_bs632,
            err_bcv10_old_d, err_bcv10_new_d, err_bcv10_old_D,
            err_bcv10_new_D, err_cv10_90)


def test_error_estimators():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.datasets import make_classification
    clf = LinearDiscriminantAnalysis()
    n_samples = 150
    n_features = 300
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    X, y = make_classification(n_samples=n_samples,
                               n_features=300,
                               n_informative=10,
                               n_clusters_per_class=1)
    errors = error_estimators(clf, X, y)
    # err_true, err_resub, err_bresub, err_cv10
    print(errors)
