#!/usr/bin/env bash

ulimit -n 1024
CMD="python main_parallel.py --clf {clf} --clf_name {clf_name} --n_iter 100\
    --n_samples {n_samples} --n_features {n_features} --n_informative 15\
    --n_feat_selected {n_feat_selected} --train_size 0.1"
CONFIG_DIR="./config/synthetic_data_M1"
# python -c 'import main_parallel; main_parallel.test_preproc()'
# parallel --dry-run --verbose -k -j 6 --header : --colsep '\t' "${CMD}" \
parallel -k -j 7 --header : --colsep '\t' "${CMD}" \
    :::: ${CONFIG_DIR}/clf_clf_name_file :::: ${CONFIG_DIR}/n_samples_file\
    :::: ${CONFIG_DIR}/n_features_file :::: ${CONFIG_DIR}/n_feat_selected_file
