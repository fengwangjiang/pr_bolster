#!/usr/bin/env bash

ulimit -n 1024
CMD="python main_parallel_realdata.py --dataset {dataset} --clf {clf}\
    --clf_name {clf_name} --n_iter 100 --n_samples_train {n_samples_train} --n_feat_selected {n_feat_selected}"
CONFIG_DIR="./config/breast_cancer"
# python -c 'import main_parallel; main_parallel.test_preproc()'
# parallel --dry-run --verbose -k -j 6 --header : --colsep '\t' "${CMD}" \
parallel -k -j 6 --header : --colsep '\t' "${CMD}"\
    :::: ${CONFIG_DIR}/dataset_file :::: ${CONFIG_DIR}/clf_clf_name_file :::: ${CONFIG_DIR}/n_samples_train_file\
    :::: ${CONFIG_DIR}/n_feat_selected_file
