import csv
import json
import os

import datasets
import pandas as pd
import numpy as np

ds = datasets.load_dataset('./wit_dataset_script.py', data_dir='./wit_data_dir/')
test_ds = ds['test']


def transform(example):

    example['pixel_values'] = np.load(example['pixels_file'])
    return example


test_ds = test_ds.map(transform)

for x in test_ds:
    print(x)
    break
