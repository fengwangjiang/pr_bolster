#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import table
from error_estimators import error_estimators
from datetime import datetime
import os
import subprocess
import logging
from pprint import pprint
from make_datasets import make_dataset_2
from save_load import save_dataset
from save_load import (load_bc_dataset, load_lc_dataset)
from scipy.stats import beta
from scipy.stats._continuous_distns import FitSolverError
from scipy.stats._continuous_distns import FitDataError
from config import (DIR_DATA, DIR_FIGURE, BETA_BOX, MEAN_STD, BVR_PLOT)

#  RESULTS = os.path.join(os.getcwd(), "results")
#  DIR_DATA = os.path.join(RESULTS, "data")
#  DIR_FIGURE = os.path.join(RESULTS, "figures")
#  BETA_BOX = os.path.join(DIR_FIGURE, "beta_box")
#  MEAN_STD = os.path.join(DIR_FIGURE, "mean_std")
#  BVR_PLOT = os.path.join(DIR_FIGURE, "bias_var_rms")


def strip_filename_suffix(f_name):
    """docstring for strip_filename_suffix"""
    f_name = os.path.basename(f_name)
    base_name = os.path.splitext(f_name)[0]
    name_list = base_name.split('_')
    suffix_len = len(name_list[-1]) + 1
    f_name_prefix = base_name[:-suffix_len]
    return f_name_prefix


def test_strip_filename_suffix():
    """docstring for test_strip_filename_suffix"""
    f_name = "./abc/clf_LDA_n_100_D_100_d0_15_d_10_error.tsv"
    f_name_prefix = strip_filename_suffix(f_name)
    assert f_name_prefix == "clf_LDA_n_100_D_100_d0_15_d_10"
    pprint(f_name_prefix)


def file_name_generator(clf_name, n_samples, n_features, n_informative,
                        n_feat_selected, dataset="synthetic_data"):
    """docstring for file_name_generator"""
    f_name_dict = dict(clf=clf_name, n=np.int(n_samples),
                       D=n_features, d0=n_informative, d=n_feat_selected)
    if "synthetic" in dataset:
        base_name = "clf_{clf}_n_{n}_D_{D}_d0_{d0}_d_{d}".format(**f_name_dict)
    else:
        base_name = "clf_{clf}_n_{n}_D_{D}_d_{d}".format(**f_name_dict)
    return base_name


def file_name_generator_real_data(clf_name, n_samples, n_features,
                                  n_feat_selected):
    """docstring for file_name_generator_real_data"""
    f_name_dict = dict(clf=clf_name, n=np.int(n_samples),
                       D=n_features, d=n_feat_selected)
    base_name = "clf_{clf}_n_{n}_D_{D}_d_{d}".format(**f_name_dict)
    return base_name


def file_name_parser(f_name):
    """docstring for file_name_parser"""
    base_name = os.path.basename(f_name)
    name_list = base_name.split('_')
    # clf_LDA_n_100_D_200_d0_15_d_10_error.tsv
    # ['clf', 'n', 'D', 'd0', 'd']
    # ['LDA', '100', '200', '15', '10']
    k = name_list[0:-1:2]
    v = name_list[1::2]
    v = [x if i == 0 else np.int(x) for i, x in enumerate(v)]
    # ['LDA', 100, 200, 15, 10]
    name_dict = dict(zip(k, v))
    return name_dict


def dict_compare(d1, d2):
    """Compare dictionaries, return added, removed, same keys, modified
    content.

    :param d1:
    :param d2:
    """
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def test_file_name_generator_parser():
    """docstring for test_file_name_generator_parser"""
    clf_name = "LDA"
    n_samples = 100
    n_features = 100
    n_informative = 15
    n_feat_selected = 10
    file_name = file_name_generator(clf_name, n_samples, n_features,
                                    n_informative, n_feat_selected)
    f_name_dict = file_name_parser(file_name)
    true_dict = dict(clf=clf_name, n=n_samples, D=n_features,
                     d0=n_informative, d=n_feat_selected)
    added, removed, modified, same = dict_compare(f_name_dict, true_dict)
    assert added == set()
    assert removed == set()
    assert modified == {}
    assert same == set(("clf", "n", "D", "d0", "d"))


def load_errors(f_name):
    """Preprocess data in file f_name.

    This function compute the statistics of data in file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading error from {}".format(f_name))
    df_errors = pd.read_csv(f_name, sep="\t")
    header = df_errors.columns
    # error_true_estimate_from_cv = df_errors.cv10.mean()
    # df_deviation = (df_errors - error_true_estimate_from_cv).\
    # loc[:, header[1:]]
    error_true_estimate_from_true = df_errors.true.mean()
    df_deviation = (df_errors - error_true_estimate_from_true).\
        loc[:, header[1:]]
    # print(list(header))
    # print(df_deviation)
    header_out = [header[i] for i in (1, 4, 5, 6, 8)]
    df_deviation_resub_bresubOld_bresubNew_cv10_bs632 =\
        df_deviation.loc[:, header_out]
    return df_deviation_resub_bresubOld_bresubNew_cv10_bs632


def deviation_boxplot(df, fig_name, fig_title="boxplot"):
    """boxplot of error deviations."""
    logger = logging.getLogger(__name__)
    logger.info("Saving boxplot to {}".format(fig_name))
    df.plot(kind="box")
    # plt.hlines(y=0, xmin=0, xmax=len(df.columns)+1)
    plt.axhline(0, color='k')
    # plt.axhline(df.cv10.mean(), color='k')
    plt.ylim(-0.5, 0.5)
    plt.title(fig_title)
    plt.savefig(fig_name)
    plt.close()


def test_deviation_boxplot():
    """docstring for test_deviation_boxplot"""
    path = os.path.join(DIR_DATA, "synthetic_data")
    #  path = "./results/data/synthetic_data"
    f_name = "clf_LDA_n_100_D_100_d0_15_d_10_error.tsv"
    f_name = os.path.join(path, f_name)
    df = load_errors(f_name)
    fig_name = "./tmp/test_deviation_boxplot.pdf"
    deviation_boxplot(df, fig_name)
    # plt.show()


def beta_fit_plot(df, fig_name, fig_title="betafit"):
    """Beta fitting plot. This function plot beta fittings.
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving beta fit plot to {}".format(fig_name))
    l, r = -0.5, 0.5
    dd = r - l
    xAxis = np.arange(l, r, 0.001)
    yscale = 0.01

    def betafitHelper(x):
        # print("beta fit x is: {}".format(x[:5]))
        try:
            para = beta.fit(x, floc=l, fscale=dd)
        except FitSolverError:
            return np.zeros_like(xAxis)
        except FitDataError:
            return np.zeros_like(xAxis)
        # print(para)
        beta_pdf = yscale * beta.pdf(xAxis, para[0], para[1], loc=l, scale=dd)
        return beta_pdf

    header = df.columns
    beta_pdfs = np.zeros((len(xAxis), len(header)))
    beta_pdfs_df = pd.DataFrame(beta_pdfs, columns=header)
    for col in header:
        beta_pdfs_df[col] = betafitHelper(df[col])
    beta_pdfs_df.plot(x=xAxis)
    plt.axvline(x=0, color='k')
    plt.title(fig_title)
    plt.savefig(fig_name)
    plt.close()


def test_beta_plot():
    """Test beta fit plot"""
    path = os.path.join(DIR_DATA, "synthetic_data")
    #  path = "./results/data/synthetic_data"
    f_name = "clf_LDA_n_100_D_100_d0_15_d_10_error.tsv"
    f_name = os.path.join(path, f_name)
    df = load_errors(f_name)
    fig_name = "./tmp/test_betafit.pdf"
    beta_fit_plot(df, fig_name)
    # plt.show()


def box_beta_plot(f_name, dataset="synthetic_data", verbose=True):
    """docstring for box_beta_plot"""

    f_name = os.path.basename(f_name)
    data_dir_name = os.path.join(DIR_DATA, dataset)
    #  data_dir_name = "./results/data/" + dataset
    data_file_name = os.path.join(data_dir_name, f_name)
    # figure names
    # f_name = base_name + "_error.tsv"
    f_name_prefix = strip_filename_suffix(f_name)
    fig_dir_name = os.path.join(BETA_BOX, dataset)
    #  fig_dir_name = "./results/figures/" + dataset
    fig_box_name = f_name_prefix + "_boxplot.pdf"
    fig_box_name = os.path.join(fig_dir_name, fig_box_name)
    fig_betafit_name = f_name_prefix + "_betafit.pdf"
    fig_betafit_name = os.path.join(fig_dir_name, fig_betafit_name)
    if not os.path.exists(fig_dir_name):
        os.makedirs(fig_dir_name)

    df = load_errors(data_file_name)
    deviation_boxplot(df, fig_box_name)
    beta_fit_plot(df, fig_betafit_name)


def run_one_suite(clf, X, y, n_iter, n_feat_selected, test_size=0.9,
                  random_state=None):
    """docstring for run_one_suite"""
    # Calculate errors
    n_err_estimators = 14
    errors = np.zeros((n_iter, n_err_estimators))
    logger = logging.getLogger(__name__)
    logger.debug("Start run_one_suite")
    print("Start run_one_suite")
    for row in range(n_iter):
        error = error_estimators(clf, X, y, n_feat_selected=n_feat_selected,
                                 test_size=test_size,
                                 random_state=random_state)
        errors[row, :] = error
    print("Finish run_one_suite")
    logger.debug("Finish run_one_suite")
    return errors


def run_experiment(clf, clf_name, n_iter, n_samples, n_features,
                   n_informative, n_feat_selected, train_size=0.1,
                   dataset="synthetic_data_M1", verbose=True,
                   random_state=None):
    """run_experiment

    :param clf:
    :param clf_name:
    :param n_iter:
    :param n_samples:
    :param n_features:
    :param n_informative:
    :param n_feat_selected:
    :param verbose:
    :param random_state:
    """

    logger = logging.getLogger(__name__)
    tstart = datetime.now()

    # file header and names
    # (err_true, err_resub, err_bresub_old_d, err_bresub_new_d,
    # err_bresub_old_D, err_bresub_new_D, err_cv10, err_bs0, err_bs632,
    # err_bcv10_old_d, err_bcv10_new_d, err_bcv10_old_D,
    # err_bcv10_new_D, err_cv10_90)
    labels = ["true", "resub", "bresub_old_d", "bresub_new_d", "bresub_old_D",
              "bresub_new_D", "cv10", "bs0", "bs632", "bcv10_old_d",
              "bcv10_new_d", "bcv10_old_D", "bcv10_new_D", "cv10_90"]
    # labels = ["true", "resub", "bresub_old", "bresub_new",
    # "cv10", "bs0", "bs632"]
    header = "\t".join(labels) + "\n"
    n_train = np.int(n_samples * train_size)
    base_name = file_name_generator(clf_name, n_train,
                                    n_features, n_informative,
                                    n_feat_selected)
    f_name = base_name + "_error.tsv"
    dir_name = os.path.join(DIR_DATA, dataset)
    #  dir_name = "./results/data/" + dataset
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_name = os.path.join(dir_name, f_name)

    # Generate dataset
    type = np.int(dataset[-1])
    X, y = make_dataset_2(
        n_samples_1=n_samples//2, n_samples_2=n_samples//2,
        n_features=n_features, n_informative=n_informative,
        class_sep=0.4, scale=2.25, type=type, G=20, rho=0.25,
        shuffle=True, random_state=random_state)
    print("In run_experiment")
    errors = run_one_suite(clf, X, y, n_iter, n_feat_selected,
                           test_size=1-train_size,
                           random_state=random_state)
    # write errors to file
    logger.info("Start writing errors to file: {}".format(f_name))
    save_dataset(errors, f_name=f_name, header=header)
    tend = datetime.now()
    logger.info("Done...")
    logger.debug("Elaped time for one run: {}".format(tend - tstart))
    box_beta_plot(f_name, dataset=dataset, verbose=verbose)


def run_real_data(clf, clf_name, n_iter, n_samples_train=50, n_feat_selected=3,
                  verbose=True, random_state=None, dataset="breast_cancer"):
    """run_real_data

    :param clf:
    :param clf_name:
    :param n_iter:
    :param n_samples_train:
    :param n_feat_selected:
    :param verbose:
    :param random_state:
    :param dataset:
    """
    logger = logging.getLogger(__name__)
    tstart = datetime.now()
    # load breast cancer data set
    if dataset == "breast_cancer":
        X, y = load_bc_dataset()
    elif dataset == "lung_cancer":
        X, y = load_lc_dataset()
    n_samples, n_features = X.shape
    # file header and names
    labels = ["true", "resub", "bresub_old_d", "bresub_new_d", "bresub_old_D",
              "bresub_new_D", "cv10", "bs0", "bs632", "bcv10_old_d",
              "bcv10_new_d", "bcv10_old_D", "bcv10_new_D", "cv10_90"]
    header = "\t".join(labels) + "\n"

    base_name = file_name_generator(
        clf_name, n_samples_train,
        n_features, 0, n_feat_selected, dataset=dataset)
    f_name = base_name + "_error.tsv"
    dir_name = os.path.join(DIR_DATA, dataset)
    #  dir_name = "./results/data/" + dataset
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_name = os.path.join(dir_name, f_name)

    errors = run_one_suite(clf, X, y, n_iter, n_feat_selected,
                           test_size=n_samples - n_samples_train,
                           random_state=random_state)
    # write errors to file
    logger.info("Start writing errors to file: {}".format(f_name))
    save_dataset(errors, f_name=f_name, header=header)
    tend = datetime.now()
    logger.info("Done...")
    logger.debug("Elaped time for one run: {}".format(tend - tstart))
    box_beta_plot(f_name, dataset=dataset, verbose=verbose)


def load_errors_mean_std(f_name, dataset="synthetic_data"):
    """
    create bar plot, with error as bar high, and with std as error bar.
    """
    df = pd.read_csv(f_name, sep="\t")
    # header = df_errors.columns
    f_name = os.path.basename(f_name)
    name_list = f_name.split('_')
    # clf_LDA_n_100_D_200_d0_15_d_10_error.tsv
    # ['clf', 'n', 'D', 'd0', 'd']
    # ['LDA', '100', '200', '15', '10']
    k = name_list[0:-1:2]
    v = name_list[1::2]
    v = [x if i == 0 else np.int(x) for i, x in enumerate(v)]
    # ['LDA', 100, 200, 15, 10]
    name_dict = dict(zip(k, v))
    for k_, v_ in name_dict.items():
        df[k_] = v_
    grouped = df.groupby(by=k)
    # grouped = df.groupby(by=['clf', 'n', 'D', 'd0', 'd'])

    means = grouped.mean()
    errors = grouped.std()
    fig, ax = plt.subplots()
    means.plot(yerr=errors, ax=ax, kind='bar')
    ax.set_xticklabels('')
    # ax.set_xticklabels(",".join(str(v)))
    # fig.tight_layout()
    dir_name = os.path.join(MEAN_STD, dataset)
    #  dir_name = "./tests/figures/mean_std/" + dataset
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # fig_name = os.path.splitext(base_name)[0] + "_mean_std.pdf"
    f_name_prefix = strip_filename_suffix(f_name)
    fig_name = f_name_prefix + "_mean_std.pdf"
    fig_name = os.path.join(dir_name, fig_name)
    plt.savefig(fig_name)
    plt.close()


def deviation_regressionplot(df, fig_name):
    """regression plot of error deviations."""
    # x = np.arange(len(df))
    # y = df.iloc[:, 0]
    # # plt.scatter(x, y)
    # fit = np.polyfit(x, y, 1)
    # fit_fn = np.poly1d(fit)
    # # fit_fn is now a function which takes in x and returns an estimate for y
    # plt.plot(x, y, 'yo', x, fit_fn(x), '--k',
    # label="slope: {:.4f}".format(fit[0]))
    # plt.legend()
    # # scatter_matrix(df, alpha=0.8, diagonal='kde')
    # plt.savefig(fig_name)
    fig, ax = plt.subplots(1, 1)
    table(ax, np.round(df.describe(), 2),
          loc="upper right", colWidths=[0.1] * 7)
    # df.plot(ax=ax)
    plt.savefig(fig_name)
    plt.close()


def test_deviation_regressionplot():
    path = os.path.join(DIR_DATA, "synthetic_data")
    #  path = "./results/data/synthetic_data"
    f_name = "clf_LDA_n_100_D_100_d0_15_d_10_error.tsv"
    f_name = os.path.join(path, f_name)
    df = load_errors(f_name)
    fig_name = "./tmp/test_deviation_regressionplot.pdf"
    deviation_regressionplot(df, fig_name)
    #  plt.show()


def bvr_plot(df, fig_name):
    """
    Bias, var, rms plots.
    """
    fig, axes = plt.subplots(nrows=3, ncols=1)
    df.mean().plot(ax=axes[0], kind="barh")
    df.std().plot(ax=axes[1], kind="barh")
    df_rms = (df.mean() ** 2 + df.var()) ** (0.5)
    df_rms.plot(ax=axes[2], kind="barh")
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def calculate_bvr(f_name):
    df = load_errors(f_name)
    base_name = os.path.basename(f_name)
    name_list = base_name.split('_')
    # clf_LDA_n_100_D_200_d0_15_d_10_error.tsv
    # ['clf', 'n', 'D', 'd0', 'd']
    # ['LDA', '100', '200', '15', '10']
    k = name_list[0:-1:2]
    v = name_list[1::2]
    v = [x if i == 0 else np.int(x) for i, x in enumerate(v)]
    # ['LDA', 100, 200, 15, 10]
    name_dict = dict(zip(k, v))
    for k, v in name_dict.items():
        df[k] = v
    grouped = df.groupby(by=['clf', 'n', 'D', 'd0', 'd'])

    def bias(x):
        return np.mean(x)

    def variance(x):
        return np.var(x, ddof=1)

    def rms(x):
        return np.sqrt(np.mean(x) ** 2 + np.var(x, ddof=1))
    bvr_df = grouped.agg([bias, variance, rms])
    return bvr_df


def calculate_bvr_real_data(f_name):
    df = load_errors(f_name)
    base_name = os.path.basename(f_name)
    name_list = base_name.split('_')
    # clf_LDA_n_100_D_70_d_10_error.tsv
    # ['clf', 'n', 'D', 'd']
    # ['LDA', '100', '200', '10']

    k = name_list[0:-1:2]
    v = name_list[1::2]
    v = [x if i == 0 else np.int(x) for i, x in enumerate(v)]
    # ['LDA', 100, 200, 10]
    name_dict = dict(zip(k, v))
    for k, v in name_dict.items():
        df[k] = v
    grouped = df.groupby(by=['clf', 'n', 'D', 'd'])

    def bias(x):
        return np.mean(x)

    def variance(x):
        return np.var(x, ddof=1)

    def rms(x):
        return np.sqrt(np.mean(x) ** 2 + np.var(x, ddof=1))
    bvr_df = grouped.agg([bias, variance, rms])
    return bvr_df


def test_calculate_bvr():
    """docstring for test_calculate_bvr"""
    path = os.path.join(DIR_DATA, "synthetic_data")
    #  path = "./results/data/synthetic_data"
    f_name = "clf_LDA_n_100_D_100_d0_15_d_10_error.tsv"
    f_name = os.path.join(path, f_name)
    #  import ipdb
    #  ipdb.set_trace()
    bvr_df = calculate_bvr(f_name)
    print(bvr_df)


def test_calculate_bvr_real_data():
    """docstring for test_calculate_bvr_real_data"""
    f_name = os.path.join(
        DIR_DATA, "breast_cancer",
        "clf_LDA_n_100_D_70_d_10_error.tsv")
    #  f_name = "results/data/breast_cancer/clf_LDA_n_100_D_70_d_10_error.tsv"
    bvr_df = calculate_bvr_real_data(f_name)
    print(bvr_df)


def init_params(dataset="synthetic_data"):
    """docstring for init_params"""
    clf_name_list = ["LDA", "LSVM", "RBFSVM", "3NN"]
    n_samples_list = np.arange(20, 101, 10)
    if "synthetic" in dataset:
        n_features_list = np.array([100])
    else:
        n_features_list = np.array([70])
    n_feat_selected_list = np.array([2, 3, 5, 8, 10, 12, 15])

    return (clf_name_list, n_samples_list, n_features_list,
            n_feat_selected_list)


#  def init_params(dataset="synthetic_data"):
    #  """docstring for init_params"""
    #  clf_name_list = ["LDA"]  # "LSVM", "RBFSVM", "3NN"]
    #  n_samples_list = np.arange(20, 21, 10)
    #  if "synthetic" in dataset:
    #  n_features_list = np.array([100])
    #  else:
    #  n_features_list = np.array([70])
    #  n_feat_selected_list = np.array([2])  # 3, 5, 8, 10, 12, 15])

    #  return (clf_name_list, n_samples_list, n_features_list,
    #  n_feat_selected_list)


def test_init_params():
    """docstring for test_init_params"""
    print(init_params())


def bvr_dataframe(dataset="synthetic_data"):
    """docstring for bvr_dataframe"""
    n_informative = 15

    clf_name_list, n_samples_list, n_features_list, n_feat_selected_list =\
        init_params(dataset=dataset)

    bvr_df_list = []
    for clf_name in clf_name_list:
        for n_samples in n_samples_list:
            for n_features in n_features_list:
                for n_feat_selected in n_feat_selected_list:
                    base_name = file_name_generator(clf_name, n_samples,
                                                    n_features, n_informative,
                                                    n_feat_selected)
                    f_name = base_name + "_error.tsv"
                    dir_name = os.path.join(DIR_DATA, dataset)
                    #  dir_name = os.path.join("./results/data", dataset)
                    f_name = os.path.join(dir_name, f_name)
                    bvr_df = calculate_bvr(f_name)
                    bvr_df_list.append(bvr_df)
    bvr_df_list = pd.concat(bvr_df_list)
    bvr_df_list_sort = bvr_df_list.sortlevel().sortlevel()\
        .sortlevel().sortlevel().sortlevel()

    return bvr_df_list_sort


def bvr_dataframe_real_data(dataset="breast_cancer"):
    """docstring for bvr_dataframe_real_data"""

    clf_name_list, n_samples_list, n_features_list, n_feat_selected_list =\
        init_params(dataset=dataset)

    bvr_df_list = []
    for clf_name in clf_name_list:
        for n_samples in n_samples_list:
            for n_features in n_features_list:
                for n_feat_selected in n_feat_selected_list:
                    base_name = file_name_generator_real_data(clf_name,
                                                              n_samples,
                                                              n_features,
                                                              n_feat_selected)
                    f_name = base_name + "_error.tsv"
                    dir_name = os.path.join(DIR_DATA, dataset)
                    #  dir_name = "./results/data/" + dataset
                    f_name = os.path.join(dir_name, f_name)
                    bvr_df = calculate_bvr_real_data(f_name)
                    bvr_df_list.append(bvr_df)
    bvr_df_list = pd.concat(bvr_df_list)
    bvr_df_list_sort = bvr_df_list.sortlevel().sortlevel()\
        .sortlevel().sortlevel()

    return bvr_df_list_sort


def bvr_plots(df, fig_name, clf_name_list=['LDA'],
              n_samples_list=range(20, 101, 10),
              n_features_list=[100],
              n_informative=[15], n_feat_selected_list=[5], bvr='rms'):
    """docstring for bvr_plots"""
    # print("bvr_df index levels: {}".format(df.index.levels))
    row_ = (clf_name_list, n_samples_list, n_features_list,
            n_informative, n_feat_selected_list)
    row_ = tuple(map(list, row_))
    row_len = np.array(map(len, row_))
    col_ = (slice(None), bvr)
    # print("row: {}, col: {}".format(row_, col_))
    df_plot = df.loc[row_, col_]

    assert np.sum(row_len != 1) == 1,\
        "only ONE of {} is of length > 1".format(row_)
    ind_gt_one = np.where(row_len > 1)[0]
    x_tick_labels = row_[ind_gt_one]

    x_labels = ['classifier', 'n_samples', 'n_features', 'n_informative',
                'n_feature_selected']
    x_label = x_labels[ind_gt_one]

    ind_eq_one = np.where(row_len == 1)[0]

    title_a = ['clf', 'n', 'D', 'd0', 'd']
    title_b = row_
    title_a_plot = [title_a[i] for i in ind_eq_one]
    title_b_plot = [title_b[i][0] for i in ind_eq_one]
    title_ = ''
    for t_a, t_b in zip(title_a_plot, title_b_plot):
        tmp = t_a + '_' + str(t_b) + '_'
        title_ += tmp
    title = title_[:-1]

    ax = df_plot.plot()
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(bvr)
    ax.set_title(title)
    # ax.set_xlabel(b_plot.index.names[4])
    # ax.set_xticklabels(b_plot.index.levels[4])
    ax.legend(df_plot.columns.droplevel(level=1), loc="best", frameon=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig(fig_name)
    # plt.show()
    plt.close()


def bvr_plots_real_data(df, fig_name, clf_name_list=['LDA'],
                        n_samples_list=range(20, 101, 10),
                        n_features_list=[70],
                        n_feat_selected_list=[5], bvr='rms'):
    """docstring for bvr_plots_real_data"""
    # print("bvr_df index levels: {}".format(df.index.levels))
    row_ = (clf_name_list, n_samples_list, n_features_list,
            n_feat_selected_list)
    row_ = tuple(map(list, row_))
    row_len = np.array(map(len, row_))
    col_ = (slice(None), bvr)
    # print("row: {}, col: {}".format(row_, col_))
    df_plot = df.loc[row_, col_]

    assert np.sum(row_len != 1) == 1,\
        "only ONE of {} is of length > 1".format(row_)
    ind_gt_one = np.where(row_len > 1)[0]
    x_tick_labels = row_[ind_gt_one]

    x_labels = ['classifier', 'n_samples', 'n_features',
                'n_feature_selected']
    x_label = x_labels[ind_gt_one]

    ind_eq_one = np.where(row_len == 1)[0]

    title_a = ['clf', 'n', 'D', 'd']
    title_b = row_
    title_a_plot = [title_a[i] for i in ind_eq_one]
    title_b_plot = [title_b[i][0] for i in ind_eq_one]
    title_ = ''
    for t_a, t_b in zip(title_a_plot, title_b_plot):
        tmp = t_a + '_' + str(t_b) + '_'
        title_ += tmp
    title = title_[:-1]

    ax = df_plot.plot()
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(bvr)
    ax.set_title(title)
    ax.legend(df_plot.columns.droplevel(level=1), loc="best", frameon=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig(fig_name)
    plt.close()


def test_bvr_plots(dataset="synthetic_data"):
    """docstring for test_bvr_plots"""
    df = bvr_dataframe(dataset=dataset)
    # bvr_vs_n
    # bvr_vs_d
    dir_name = os.path.join(BVR_PLOT, dataset)
    #  dir_name = os.path.join("./tests/figures", dataset)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_name_list_bvr_vs_n = ['b_vs_n', 'v_vs_n', 'r_vs_n']
    fig_name_list_bvr_vs_d = ['b_vs_d', 'v_vs_d', 'r_vs_d']
    bvr_list = ['bias', 'variance', 'rms']
    clf_name_list = ["LDA", "LSVM", "RBFSVM", "3NN"]
    n_feat_selected_list = [2, 3, 5, 8, 10, 12, 15]

    for clf in clf_name_list:
        for i, bvr in enumerate(bvr_list):
            fig_name = clf + '_' + fig_name_list_bvr_vs_n[i] + '.pdf'
            fig_name = os.path.join(dir_name, fig_name)
            if os.path.exists(fig_name):
                print(fig_name + "exists...")
                continue
            bvr_plots(df, fig_name, clf_name_list=[clf],
                      n_feat_selected_list=[5], bvr=bvr)
            # subprocess.call(["xdg-open", fig_name])

        for i, bvr in enumerate(bvr_list):
            fig_name = clf + '_' + fig_name_list_bvr_vs_d[i] + '.pdf'
            fig_name = os.path.join(dir_name, fig_name)
            if os.path.exists(fig_name):
                print(fig_name + "exists...")
                continue
            bvr_plots(df, fig_name, clf_name_list=[clf], n_samples_list=[100],
                      n_feat_selected_list=n_feat_selected_list,
                      bvr=bvr)
            # subprocess.call(["xdg-open", fig_name])


def test_bvr_plots_real_data(dataset="breast_cancer"):
    """docstring for test_bvr_plots_real_data"""
    df = bvr_dataframe_real_data(dataset=dataset)
    # bvr_vs_n
    # bvr_vs_d
    # dir_name = "./tests/figures/breast_cancer"
    dir_name = os.path.join(BVR_PLOT, dataset)
    #  dir_name = "./tests/figures/" + dataset
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_name_list_bvr_vs_n = ['b_vs_n', 'v_vs_n', 'r_vs_n']
    fig_name_list_bvr_vs_d = ['b_vs_d', 'v_vs_d', 'r_vs_d']
    bvr_list = ['bias', 'variance', 'rms']
    clf_name_list = ["LDA", "LSVM", "RBFSVM", "3NN"]
    n_feat_selected_list = [2, 3, 5, 8, 10, 12, 15]

    for clf in clf_name_list:
        for i, bvr in enumerate(bvr_list):
            fig_name = clf + '_' + fig_name_list_bvr_vs_n[i] + '.pdf'
            fig_name = os.path.join(dir_name, fig_name)
            if os.path.exists(fig_name):
                print(fig_name + "exists...")
                continue
            bvr_plots_real_data(df, fig_name, clf_name_list=[clf],
                                n_feat_selected_list=[5], bvr=bvr)
            # subprocess.call(["xdg-open", fig_name])

        for i, bvr in enumerate(bvr_list):
            fig_name = clf + '_' + fig_name_list_bvr_vs_d[i] + '.pdf'
            fig_name = os.path.join(dir_name, fig_name)
            if os.path.exists(fig_name):
                print(fig_name + "exists...")
                continue
            bvr_plots_real_data(df, fig_name, clf_name_list=[clf],
                                n_samples_list=[100],
                                n_feat_selected_list=n_feat_selected_list,
                                bvr=bvr)
            # subprocess.call(["xdg-open", fig_name])


def test_bvr_plots_2():
    """docstring for test_bvr_plots"""
    df = bvr_dataframe()
    fig_name = "bvr_plots_2.pdf"
    dir_name = "./tmp"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig_name = os.path.join(dir_name, fig_name)
    bvr_plots(df, fig_name, n_feat_selected_list=[10])
    subprocess.call(["xdg-open", fig_name])


def test_bvr_dataframe():
    """docstring for test_bvr_dataframe"""
    bvr_df = bvr_dataframe()
    print(bvr_df.head(2))


def test_bvr_dataframe_real_data():
    """docstring for test_bvr_dataframe"""
    bvr_df = bvr_dataframe_real_data()
    print(bvr_df.head(2))


def test_run_experiment(verbose=True):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    n_iter = 10
    n_samples = 300
    n_features = 10
    n_informative = 5
    n_feat_selected = 3
    run_experiment(clf, n_iter, n_samples, n_features, n_informative,
                   n_feat_selected, verbose=verbose)


def test_load_errors():
    """docstring for test_load_errors"""
    f_name = os.path.join(DIR_DATA, "clf_LDA_D_10_d0_5_d_3_error.tsv")
    #  f_name = "./results/data/clf_LDA_D_10_d0_5_d_3_error.tsv"
    df = load_errors(f_name)
    print(df)


def test_load_errors_mean_std(dataset="synthetic_data"):
    n_informative = 15
    clf_name_list, n_samples_list, n_features_list, n_feat_selected_list =\
        init_params(dataset=dataset)

    dir_name = os.path.join(DIR_DATA, dataset)
    #  dir_name = "./results/data/" + dataset
    # f_name = dir_name + "clf_LDA_n_100_D_100_d0_15_d_2_error.tsv"
    # f_name = dir_name + "clf_LDA_n_20_D_100_d0_15_d_2_error.tsv"
    for clf_name in clf_name_list:
        for n_samples in n_samples_list:
            for n_features in n_features_list:
                for n_feat_selected in n_feat_selected_list:
                    if "synthetic" in dataset:
                        base_name =\
                            file_name_generator(clf_name, n_samples,
                                                n_features,
                                                n_informative,
                                                n_feat_selected)
                    else:
                        base_name =\
                            file_name_generator_real_data(clf_name,
                                                          n_samples,
                                                          n_features,
                                                          n_feat_selected)
                    f_name = base_name + "_error.tsv"
                    f_name = os.path.join(dir_name, f_name)
                    load_errors_mean_std(f_name, dataset)


def test_load_bc_dataset():
    """docstring for test_load_bc_dataset"""
    index, header, X, y = load_bc_dataset()
    print("headers are: {}".format(header))
    print("X shape: {}".format(X.shape))
    print("index length: {}".format(len(index)))
    zero_one_count = np.bincount(y)
    print("y has {} labels, {} are 0s, {} are 1s.".format(len(y),
                                                          zero_one_count[0],
                                                          zero_one_count[1]))


def test_load_lc_dataset():
    """docstring for test_load_lc_dataset"""
    X, y = load_lc_dataset()
    print("X shape: {}".format(X.shape))
    zero_one_count = np.bincount(y)
    print("y has {} labels, {} are 0s, {} are 1s.".format(len(y),
                                                          zero_one_count[0],
                                                          zero_one_count[1]))


def test_beta_fit_plot():
    """docstring for test_deviation_boxplot"""
    f_name = os.path.join(DIR_DATA, "clf_SVC_D_50_d0_15_d_5_error.tsv")
    #  f_name = "./results/data/clf_SVC_D_50_d0_15_d_5_error.tsv"
    df = load_errors(f_name)
    fig_name = os.path.join(DIR_FIGURE, "test_beta_fit_plot.pdf")
    #  fig_name = "./results/figures/test_beta_fit_plot.pdf"
    beta_fit_plot(df, fig_name)
    # plt.show()


def test_box_beta_plot(dataset="synthetic_data"):
    """docstring for test_box_beta_plot"""
    n_informative = 15
    clf_name_list, n_samples_list, n_features_list, n_feat_selected_list =\
        init_params(dataset=dataset)

    for clf_name in clf_name_list:
        for n_samples in n_samples_list:
            for n_features in n_features_list:
                for n_feat_selected in n_feat_selected_list:
                    base_name = file_name_generator(clf_name, n_samples,
                                                    n_features, n_informative,
                                                    n_feat_selected,
                                                    dataset=dataset)
                    f_name = base_name + "_error.tsv"
                    box_beta_plot(f_name, dataset=dataset, verbose=False)
