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
    write_list(clf_list, "clf", filename="clf_file")
    write_list(clf_name_list, "clf_name", filename="clf_name_file")
    write_list(n_samples_list, "n_samples", filename="n_samples_file")
    write_list(n_features_list, "n_features", filename="n_features_file")
    write_list(n_feat_selected_list, "n_feat_selected",
               filename="n_feat_selected_file")
    import subprocess
    import shlex
    cmd = "paste clf_file clf_name_file"
    with open("clf_clf_name_file", 'w') as f:
        subprocess.Popen(shlex.split(cmd), stdout=f)


def test_preproc(dataset="synthetic_data_M1"):
    """docstring for test_preproc"""
    preproc(dataset=dataset)
    assert os.path.exists('clf_name_file')


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


def main_run_real_data_list(
        dataset="breast_cancer", verbose=False, random_state=None):
    clf_list = [LinearDiscriminantAnalysis(), SVC(kernel='linear'),
                SVC(kernel='rbf'), KNeighborsClassifier(n_neighbors=3)]
    clf_name_list = ["LDA", "LSVM", "RBFSVM", "3NN"]
    n_iter = 100
    #  n_iter = 2
    n_features = 70
    n_samples_train_list = range(20, 101, 10)
    n_feat_selected_list = [2, 3, 5, 8, 10, 12, 15]

    clf_name_list, n_samples_train_list, _, n_feat_selected_list =\
        utils.init_params(dataset=dataset)

    for clf, clf_name in zip(clf_list, clf_name_list):
        for n_samples_train in n_samples_train_list:
            for n_feat_selected in n_feat_selected_list:
                base_name = utils.\
                    file_name_generator_real_data(clf_name,
                                                  n_samples_train,
                                                  n_features,
                                                  n_feat_selected)
                f_name = base_name + "_betafit.pdf"
                dir_name = "./results/figures/" + dataset
                f_name = os.path.join(dir_name, f_name)
                if os.path.exists(f_name):
                    print(f_name + " exists...")
                    continue
                utils.run_real_data(clf, clf_name, n_iter,
                                    n_samples_train=n_samples_train,
                                    n_feat_selected=n_feat_selected,
                                    verbose=verbose,
                                    random_state=random_state,
                                    dataset=dataset)
    utils.test_load_errors_mean_std(dataset=dataset)


def main():
    """docstring for main"""
    time_start = datetime.now()
    #  runner(dataset="synthetic_data_M1")
    #  runner(dataset="synthetic_data_M2")
    #  runner(dataset="synthetic_data_M3")
    #  runner(dataset="synthetic_data_M4")
    from utils import test_bvr_plots
    test_bvr_plots(dataset="synthetic_data_M1")
    #  test_bvr_plots(dataset="synthetic_data_M2")
    #  test_bvr_plots(dataset="synthetic_data_M3")
    #  test_bvr_plots(dataset="synthetic_data_M4")
    #  utils.test_load_errors_mean_std(dataset="synthetic_data_M1")
    #  utils.test_load_errors_mean_std(dataset="synthetic_data_M2")
    #  utils.test_load_errors_mean_std(dataset="synthetic_data_M3")
    #  utils.test_load_errors_mean_std(dataset="synthetic_data_M4")
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
