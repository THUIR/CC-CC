# coding=utf-8
import pandas as pd
from collections import Counter, defaultdict
import os
import numpy as np
import socket
from scipy.stats import chi2_contingency

np.random.seed(2018)

DATASET_DIR = '../dataset'


def random_split_data(all_data_file, dataset_name, vt_ratio=0.1):
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('random_split_data', dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')
    vt_size = int(len(all_data) * vt_ratio)
    validation_set = all_data.sample(n=vt_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=vt_size).sort_index()
    train_set = all_data.drop(test_set.index)
    print(train_set)
    print(validation_set)
    print(test_set)
    train_set.to_csv(os.path.join(dir_name, dataset_name + '.train.csv'), index=False, sep='\t')
    validation_set.to_csv(os.path.join(dir_name, dataset_name + '.validation.csv'), index=False, sep='\t')
    test_set.to_csv(os.path.join(dir_name, dataset_name + '.test.csv'), index=False, sep='\t')
    return train_set, validation_set, test_set


def change_id(dataset_name='ml100k-r', item_cold_ratio=0.0, user_cold_ratio=0.0):
    print('change_id', dataset_name, item_cold_ratio, user_cold_ratio)
    random_dir = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(random_dir):
        print("Dataset doesn't exist:", random_dir)
        return
    if item_cold_ratio <= 0 and user_cold_ratio <= 0:
        return random_split_data(dataset_name)

    ci_name = dataset_name
    if item_cold_ratio > 0:
        ci_name += '-i%d' % int(item_cold_ratio * 100)
    if user_cold_ratio > 0:
        ci_name += '-u%d' % int(user_cold_ratio * 100)
    dir_name = os.path.join(DATASET_DIR, ci_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    train_set = pd.read_csv(os.path.join(random_dir, dataset_name + '.train.csv'), sep='\t')
    validation_set = pd.read_csv(os.path.join(random_dir, dataset_name + '.validation.csv'), sep='\t')
    test_set = pd.read_csv(os.path.join(random_dir, dataset_name + '.test.csv'), sep='\t')

    all_set = pd.concat([train_set, validation_set, test_set])
    max_uid, max_iid = all_set['uid'].max(), all_set['iid'].max()
    # print('origin uid-%d' % len(all_set['uid'].unique()), 'iid-%d' % len(all_set['iid'].unique()))
    # origin_iids = set(all_set['iid'].unique())

    if item_cold_ratio > 0:
        validation_cold_size = int(len(validation_set) * item_cold_ratio)
        validation_cold_rows = validation_set.sample(n=validation_cold_size)
        validation_cold_rows['iid'] = range(max_iid + 1, max_iid + validation_cold_size + 1)
        validation_set = pd.concat([validation_cold_rows, validation_set.drop(validation_cold_rows.index)])

        test_cold_size = int(len(test_set) * item_cold_ratio)
        test_cold_rows = test_set.sample(n=test_cold_size)
        test_cold_rows['iid'] = range(max_iid + validation_cold_size + 1,
                                      max_iid + validation_cold_size + test_cold_size + 1)
        test_set = pd.concat([test_cold_rows, test_set.drop(test_cold_rows.index)])
    if user_cold_ratio > 0:
        validation_cold_size = int(len(validation_set) * user_cold_ratio)
        validation_cold_rows = validation_set.sample(n=validation_cold_size)
        validation_cold_rows['uid'] = range(max_uid + 1, max_uid + validation_cold_size + 1)
        validation_set = pd.concat([validation_cold_rows, validation_set.drop(validation_cold_rows.index)])

        test_cold_size = int(len(test_set) * user_cold_ratio)
        test_cold_rows = test_set.sample(n=test_cold_size)
        test_cold_rows['uid'] = range(max_uid + validation_cold_size + 1,
                                      max_uid + validation_cold_size + test_cold_size + 1)
        test_set = pd.concat([test_cold_rows, test_set.drop(test_cold_rows.index)])

    train_set.to_csv(os.path.join(dir_name, ci_name + '.train.csv'), index=False, sep='\t')
    validation_set.to_csv(os.path.join(dir_name, ci_name + '.validation.csv'), index=False, sep='\t')
    test_set.to_csv(os.path.join(dir_name, ci_name + '.test.csv'), index=False, sep='\t')
    all_set = pd.concat([train_set, validation_set, test_set])

    print('new uid-%d' % len(all_set['uid'].unique()), 'iid-%d' % len(all_set['iid'].unique()))
    print(train_set)
    print(validation_set)
    print(test_set)
    # new_iids = set(all_set['iid'].unique())
    # print(len(origin_iids), len(new_iids))
    # print(len(new_iids & origin_iids))
    # print(len(new_iids - origin_iids))
    return train_set, validation_set, test_set


def change_feature(dataset_name='ml100k-r', ratio=0.0):
    print('change_feature', dataset_name, ratio)
    random_dir = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(random_dir):
        print("Dataset doesn't exist:", random_dir)
        return
    if ratio <= 0:
        return random_split_data(dataset_name)

    cf_name = dataset_name
    if ratio > 0:
        cf_name += '-f%d' % int(ratio * 100)
    dir_name = os.path.join(DATASET_DIR, cf_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    train_set = pd.read_csv(os.path.join(random_dir, dataset_name + '.train.csv'), sep='\t')
    validation_set = pd.read_csv(os.path.join(random_dir, dataset_name + '.validation.csv'), sep='\t')
    test_set = pd.read_csv(os.path.join(random_dir, dataset_name + '.test.csv'), sep='\t')

    all_set = pd.concat([train_set, validation_set, test_set])

    if ratio > 0:
        features = [c for c in all_set.columns if c.startswith('u_') or c.startswith('i_')]
        feature_max = np.max(all_set.values, axis=0)

        validation_values = validation_set.values
        validation_cf_size = int(validation_values.shape[0] * validation_values.shape[1] * ratio)
        validation_rows = np.random.randint(validation_values.shape[0], size=validation_cf_size)
        validation_cols = np.random.randint(1, len(features) + 1, size=validation_cf_size)
        # print(validation_rows)
        # print(validation_cols)
        # print(validation_set.loc[validation_rows[0], features[-validation_cols[0]]])
        for i, r in enumerate(validation_rows):
            c = -validation_cols[i]
            value = validation_values[r][c]
            if value > 0:
                validation_values[r][c] = 0
            # else:
            #     validation_values[r][c] = np.random.randint(1, feature_max[c] + 1)
        validation_set = pd.DataFrame(data=validation_values, columns=validation_set.columns)
        # print(validation_set.loc[validation_rows[0], features[-validation_cols[0]]])

        test_values = test_set.values
        test_cf_size = int(test_values.shape[0] * test_values.shape[1] * ratio)
        test_rows = np.random.randint(test_values.shape[0], size=test_cf_size)
        test_cols = np.random.randint(1, len(features) + 1, size=test_cf_size)
        # print(test_rows)
        # print(test_cols)
        # print(test_set.loc[test_rows[0], features[-test_cols[0]]])
        for i, r in enumerate(test_rows):
            c = -test_cols[i]
            value = test_values[r][c]
            if value > 0:
                test_values[r][c] = 0
            # else:
            #     test_values[r][c] = np.random.randint(1, feature_max[c] + 1)
        test_set = pd.DataFrame(data=test_values, columns=test_set.columns)
        # print(test_set.loc[test_rows[0], features[-test_cols[0]]])

    train_set.to_csv(os.path.join(dir_name, cf_name + '.train.csv'), index=False, sep='\t')
    validation_set.to_csv(os.path.join(dir_name, cf_name + '.validation.csv'), index=False, sep='\t')
    test_set.to_csv(os.path.join(dir_name, cf_name + '.test.csv'), index=False, sep='\t')

    return train_set, validation_set, test_set
