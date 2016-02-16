#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os


def my_makedirs(directory):
    """make directory if it does not exist."""
    if not os.path.exists(directory):
        print("Creating directory: {}".format(directory))
        os.makedirs(directory)


RESULTS = os.path.join(os.getcwd(), "results")
DIR_DATA = os.path.join(RESULTS, "data")
DIR_FIGURE = os.path.join(RESULTS, "figures")
BETA_BOX = os.path.join(DIR_FIGURE, "beta_box")
MEAN_STD = os.path.join(DIR_FIGURE, "mean_std")
BVR_PLOT = os.path.join(DIR_FIGURE, "bias_var_rms")


def initialization():
    my_makedirs(RESULTS)
    my_makedirs(DIR_DATA)
    my_makedirs(DIR_FIGURE)
    my_makedirs(BETA_BOX)
    my_makedirs(MEAN_STD)
    my_makedirs(BVR_PLOT)
