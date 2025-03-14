from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import json


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])

train_reader = InHospitalMortalityReader(dataset_dir='/raid/home/mohimenul/Data/in-hospital-mortality/train',
                                             listfile='/raid/home/mohimenul/Data/in-hospital-mortality/train/listfile.csv',
                                             period_length=48.0)

print('Reading data and extracting features ...')
(train_X, train_y, train_names) = read_and_extract_features(train_reader, 'all', 'all')

print('  train data shape = {}'.format(train_X.shape))
