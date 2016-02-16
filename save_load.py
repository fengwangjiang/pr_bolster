#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_selection import (f_classif, SelectKBest)


def save_dataset(X, f_name, header):
    """save_dataset

    :param X:
    :param f_name:
    :param header:
    """
    with open(f_name, 'wb') as f:
        f.write(header.encode(encoding='UTF-8'))
        np.savetxt(f, X, fmt='%.4f', delimiter="\t")


def load_dataset(f_name):
    """load_dataset from file

    :param f_name:
    """
    with open(f_name, 'rb') as f:
        h = f.readline().split()
        data = np.genfromtxt(f, delimiter="\t")
    return h, data


def test_save_dataset():
    """docstring for test_save_dataset"""
    np.random.seed(0)
    X = np.random.randn(5, 3)
    f_name = "/tmp/data_example.tsv"
    labels = ['A', 'B', 'C']
    header = '\t'.join(labels) + '\n'
    save_dataset(X, f_name, header)


def test_save_load_dataset():
    """test_save_load_dataset"""
    np.random.seed(0)
    X = np.random.randn(5, 3)
    labels = ['A1', 'B1', 'C1']
    f_name = "/tmp/data_example.tsv"
    header = '\t'.join(labels) + '\n'
    save_dataset(X, f_name, header)
    h, data = load_dataset(f_name)
    print("header is:\n{}".format(h))
    print("data is:\n{}".format(data))


def load_bc_dataset():
    """docstring for load_bc_dataset
    load the breast cancer real data set, which has 180 poor prognosis(y=0)
    and 115 good prognosis(y=1) samples, about 61% and 39% of the whole data
    set 295 samples.
    """
    #  import ipdb
    #  ipdb.set_trace()
    f_train = "data/Training_Data.txt"
    f_test = "data/Testing_Data.txt"
    h_train, data_train = load_dataset(f_train)
    h_test, data_test = load_dataset(f_test)
    assert h_train == h_test, "training data file header: {}\
        is not equal to testing file header: {}".format(h_train, h_test)
    n_col = len(h_train)
    assert data_train.shape[1] == n_col & data_test.shape[1] == n_col,\
        "training data feature num: {} should equal testing data feature num:\
        {}".format(data_train.shape[1], data_test.shape[1])
    #  index_train = data_train[:, 0]
    #  index_test = data_test[:, 0]
    X_train = data_train[:, 1:-1]
    X_test = data_test[:, 1:-1]
    y_train = data_train[:, -1]
    y_test = data_test[:, -1]

    #  index = np.concatenate((index_train, index_test))
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test)).astype(np.int)
    assert y.sum() == 115
    return X, y


def test_load_bc_dataset():
    """docstring for test_load_bc_dataset"""
    X, y = load_bc_dataset()
    assert X.shape == (295, 70)
    assert y.sum() == 115


def load_lc_dataset():
    """docstring for load_lc_dataset
    load the lung cancer real data set, which has 139 adenocarcinomas(y=0) and
    47 non-adenocarcinomas(y=1).
    http://www.ncbi.nlm.nih.gov/pmc/articles/PMC61120/
    select k=70 from 12600 genes for easier further processing.
    """
    #  import ipdb
    #  ipdb.set_trace()
    xls_file = "data/pnas_191502998_DatasetA_12600gene.xls"
    parse_cols = [0] + list(range(2, 140+1)) + list(range(158, 204+1))
    y = [0] * 139 + [1] * 47
    data = pd.read_excel(xls_file, sheetname=0, header=0, index_col=0,
                         parse_cols=parse_cols)
    data.columns = y
    data.columns.name = 'label'
    X = np.array(data.T)
    y = np.array(y, dtype=np.int)
    univariate_filter = SelectKBest(f_classif, k=70).fit(X, y)
    X_r = univariate_filter.transform(X)
    return X_r, y
