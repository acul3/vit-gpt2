import csv
import json
import os

import datasets
import pandas as pd
import numpy as np

ds = datasets.load_dataset('./wit_dataset_script.py', data_dir='./wit_data_dir/')
train_ds = ds['validation']


def transform(example):

    print(example)

    example['pixel_values'] = np.load(example['pixel_values'])

    return example


train_ds = train_ds.map(transform)

for x in train_ds:
    print(x)
    break
