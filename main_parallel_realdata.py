#!/usr/bin/env python
"""main function for error estimation."""
#  import numpy as np
import os
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import utils
#  from config import DIR_FIGURE
#  import multiprocessing
#  import logging
#  import functools
import argparse


def write_list(thelist, header, filename="tmp.txt"):
    """write each item in thelist to a line in filename"""
    with open(filename, 'w') as fh:
        fh.write("{}\n".format(header))
        for item in thelist:
            fh.write("{}\n".format(item))


def preproc(dataset):
    """docstring for preproc"""
    clf_list = [LinearDiscriminantAnalysis(), SVC(kernel='linear'),
                SVC(kernel='rbf'), KNeighborsClassifier(n_neighbors=3)]

    clf_name_list, n_samples_list, n_features_list, n_feat_selected_list =\
        utils.init_params(dataset=dataset)
    n_samples_list = n_samples_list * 10

    conf_dir = os.path.join("./config", dataset)
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)
    clf_file = os.path.join(conf_dir, "clf_file")
    clf_name_file = os.path.join(conf_dir, "clf_name_file")
    n_samples_file = os.path.join(conf_dir, "n_samples_file")
    n_features_file = os.path.join(conf_dir, "n_features_file")
    n_feat_selected_file = os.path.join(conf_dir, "n_feat_selected_file")

    write_list(clf_list, "clf", filename=clf_file)
    write_list(clf_name_list, "clf_name", filename=clf_name_file)
    write_list(n_samples_list, "n_samples", filename=n_samples_file)
    write_list(n_features_list, "n_features", filename=n_features_file)
    write_list(n_feat_selected_list, "n_feat_selected",
               filename=n_feat_selected_file)
    #  need to make clf in clf_file on ONE line, then paste clf_file and
    #  clf_name_file together.

    #  import subprocess
    #  cmd = ["paste", clf_file, clf_name_file]
    #  clf_clf_name_file = os.path.join(conf_dir, "clf_clf_name_file")
    #  with open(clf_clf_name_file, 'w') as f:
    #  subprocess.popen(cmd, stdout=f)


def test_preproc():
    """docstring for test_preproc"""
    preproc(dataset="synthetic_data_M1")
    #  preproc(dataset="breast_cancer")
    #  preproc(dataset="lung_cancer")
    #  assert os.path.exists('clf_name_file')


def runner(dataset="synthetic_data_M1", verbose=False, random_state=None):
    """runner

    :param dataset:
    :param random_state:
    """
    """docstring for runner"""
    parser = argparse.ArgumentParser(
        description="Run experiment on dataset {}".format(dataset))
    parser.add_argument("--clf", action="store", help="The classifier object")
    parser.add_argument("--clf_name", action="store",
                        help="The classifier name")
    parser.add_argument("--n_iter", action="store", type=int, default=100,
                        help="The number of iterations")
    parser.add_argument("--n_samples", action="store", type=int,
                        help="The number of samples")
    parser.add_argument("--n_features", action="store", type=int,
                        help="The number of features")
    parser.add_argument("--n_informative", action="store", type=int,
                        default=15, help="The number of informative features")
    parser.add_argument("--n_feat_selected", action="store", type=int,
                        help="The number of selected features")
    parser.add_argument("--train_size", action="store", type=float,
                        default=0.1, help="The number of training samples")
    args = parser.parse_args()
    #  print(args)
    utils.run_experiment(
        clf=eval(args.clf),
        clf_name=args.clf_name,
        n_iter=args.n_iter,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_feat_selected=args.n_feat_selected,
        train_size=args.train_size,
        dataset=dataset,
        verbose=verbose,
        random_state=random_state)
    #  utils.test_load_errors_mean_std(dataset=dataset)


def preproc_realdata(dataset):
    """docstring for preproc"""
    clf_list = [LinearDiscriminantAnalysis(), SVC(kernel='linear'),
                SVC(kernel='rbf'), KNeighborsClassifier(n_neighbors=3)]

    clf_name_list, n_samples_train_list, _, n_feat_selected_list =\
        utils.init_params(dataset=dataset)
    dataset_list = ["breast_cancer", "lung_cancer"]

    conf_dir = os.path.join("./config", dataset)
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)
    clf_file = os.path.join(conf_dir, "clf_file")
    clf_name_file = os.path.join(conf_dir, "clf_name_file")
    n_samples_train_file = os.path.join(conf_dir, "n_samples_train_file")
    n_feat_selected_file = os.path.join(conf_dir, "n_feat_selected_file")
    dataset_file = os.path.join(conf_dir, "dataset_file")

    write_list(clf_list, "clf", filename=clf_file)
    write_list(clf_name_list, "clf_name", filename=clf_name_file)
    write_list(n_samples_train_list, "n_samples_train",
               filename=n_samples_train_file)
    write_list(n_feat_selected_list, "n_feat_selected",
               filename=n_feat_selected_file)
    write_list(dataset_list, "dataset", filename=dataset_file)
    #  need to make clf in clf_file on ONE line, then paste clf_file and
    #  clf_name_file together.


def test_preproc_realdata():
    """docstring for test_preproc"""
    preproc_realdata(dataset="breast_cancer")
    #  preproc(dataset="lung_cancer")


def runner_realdata(verbose=False, random_state=None):
    """runner

    :param dataset:
    :param random_state:
    """
    """docstring for runner_realdata"""
    parser = argparse.ArgumentParser(
        description="Run experiment on real dataset")
    parser.add_argument("--dataset", action="store",
                        help="The dataset to run")
    parser.add_argument("--clf", action="store", help="The classifier object")
    parser.add_argument("--clf_name", action="store",
                        help="The classifier name")
    parser.add_argument("--n_iter", action="store", type=int, default=100,
                        help="The number of iterations")
    parser.add_argument("--n_samples_train", action="store", type=int,
                        help="The number of training samples")
    parser.add_argument("--n_feat_selected", action="store", type=int,
                        help="The number of selected features")
    args = parser.parse_args()
    #  print(args)
    utils.run_real_data(
        clf=eval(args.clf),
        clf_name=args.clf_name,
        n_iter=args.n_iter,
        n_samples_train=args.n_samples_train,
        n_feat_selected=args.n_feat_selected,
        dataset=args.dataset,
        verbose=verbose,
        random_state=random_state)

    #  utils.test_load_errors_mean_std(dataset=dataset)


def main():
    """docstring for main"""
    time_start = datetime.now()
    #  runner(dataset="synthetic_data_M1")
    runner_realdata()
    #  main_run_experiment_list(dataset="synthetic_data_M1")
    #  main_run_experiment_list_parallel(dataset="synthetic_data_M1")
    #  main_run_experiment_list(dataset="synthetic_data_M2")
    #  main_run_experiment_list(dataset="synthetic_data_M3")
    #  main_run_experiment_list(dataset="synthetic_data_M4")
    #  main_run_real_data_list(dataset="breast_cancer")
    #  main_run_real_data_list_parallel()
    #  main_run_real_data_list(dataset="lung_cancer")
    time_end = datetime.now()
    print("Time elapsed: {}".format(time_end - time_start))


if __name__ == "__main__":
    #  import ipdb
    #  ipdb.set_trace()
    import matplotlib
    import warnings
    from config import initialization
    initialization()
    print(__doc__)
    warnings.filterwarnings("ignore")
    matplotlib.style.use('ggplot')
    from logging_helper import setup_logging
    setup_logging()
    main()
