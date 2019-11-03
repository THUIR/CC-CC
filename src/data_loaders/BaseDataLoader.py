# coding=utf-8
import os
import pandas as pd
import numpy as np
from collections import Counter
import logging
from utils import utils


class BaseDataLoader(object):

    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../dataset/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml100k-r',
                            help='Choose a dataset.')
        parser.add_argument('--balance_data', type=int, default=0,
                            help='Whether balance the training data')
        parser.add_argument('--label', type=str, default='label',
                            help='name of dataset label column.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, path, dataset, load_data=True, sep='\t', label='label', append_id=True, include_id=False,
                 drop_features=None, seqs_features=None, seqs_sep=',', seqs_expand=True, balance_train=False,
                 filter_info=None):
        self.dataset = dataset
        self.path = os.path.join(path, dataset)
        self.train_file = os.path.join(self.path, dataset + '.train.csv')
        self.validation_file = os.path.join(self.path, dataset + '.validation.csv')
        self.test_file = os.path.join(self.path, dataset + '.test.csv')
        self.max_file = os.path.join(self.path, dataset + '.max.csv')
        self.word2vec_file = os.path.join(self.path, dataset + '.seq.npy')
        self.include_id = include_id
        self.append_id = append_id
        self.label = label
        self.sep = sep
        self.drop_features = drop_features
        self.seqs_features = [] if seqs_features is None else seqs_features
        self.seqs_sep = seqs_sep
        self.seqs_expand = seqs_expand
        self.word_embeddings = None
        self.load_data = load_data
        self.filter_info = filter_info
        self._load_data()
        self._load_max()
        self._load_words()
        self._select_features()

        self.train_data = None
        if self.train_df is not None:
            self.train_data = self.format_data(self.train_df)
            logging.info("# of train: %d" % len(self.train_data['Y']))
            logging.info("    " + str(Counter(self.train_data['Y']).most_common()))
        if balance_train:
            self.train_data = utils.balance_data(self.train_data)
            logging.info("# Balance training set: %d" % len(self.train_data['Y']))
        self.validation_data = None
        if self.validation_df is not None:
            self.validation_data = self.format_data(self.validation_df)
            logging.info("# of validation: %d" % len(self.validation_data['Y']))
            logging.info("    " + str(Counter(self.validation_data['Y']).most_common()))
        self.test_data = None
        if self.test_df is not None:
            self.test_data = self.format_data(self.test_df)
            logging.info("# of test: %d" % len(self.test_data['Y']))
            logging.info("    " + str(Counter(self.test_data['Y']).most_common()))

    def _load_data(self):
        self.train_df, self.validation_df, self.test_df = None, None, None
        if os.path.exists(self.train_file) and self.load_data:
            logging.info("load train csv...")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            if self.drop_features is not None:
                self.train_df = self.train_df.drop(columns=self.drop_features)
            self.train_df = self.filter_data(self.train_df, self.filter_info)
            if self.label not in self.train_df.columns:
                logging.error('No Labels In Training Data: ' + self.label)
            assert self.label in self.train_df.columns
        if os.path.exists(self.validation_file) and self.load_data:
            logging.info("load validation csv...")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            if self.drop_features is not None:
                self.validation_df = self.validation_df.drop(columns=self.drop_features)
            self.validation_df = self.filter_data(self.validation_df, self.filter_info)
        if os.path.exists(self.test_file) and self.load_data:
            logging.info("load test csv...")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            if self.drop_features is not None:
                self.test_df = self.test_df.drop(columns=self.drop_features)
            self.test_df = self.filter_data(self.test_df, self.filter_info)

    def _load_max(self):
        max_series = None
        if not os.path.exists(self.max_file):
            for df in [self.train_df, self.validation_df, self.test_df]:
                if df is not None:
                    df_max = df.max()
                    for seq in self.seqs_features:
                        seqs = df[seq].str.split(',')
                        seqs = seqs.apply(lambda x: len(x))
                        df_max[seq] = seqs.max() + 1
                    max_series = df_max if max_series is None else np.maximum(max_series, df_max)
            max_series.to_csv(self.max_file, sep=self.sep)
        else:
            max_series = pd.read_csv(self.max_file, sep=self.sep, header=None)
            max_series = max_series.set_index(0, drop=True).transpose()
            max_series = max_series.loc[1]
        self.column_max = max_series
        self.class_num = self.column_max['label'] + 1
        logging.info("# of class: %d" % self.class_num)

    def _load_words(self):
        self.dictionary = {}
        for seq_feature in self.seqs_features:
            logging.info('max length of %s: %d' % (seq_feature, self.column_max[seq_feature]))
            word_file = os.path.join(self.path, self.dataset + '.%s_word.csv' % seq_feature)
            words = pd.read_csv(word_file, sep='\t', header=None)
            words_dict = dict(zip(words[0].tolist(), words[1].tolist()))
            self.dictionary[seq_feature] = words_dict
            logging.info('max word id of %s: %d' % (seq_feature, max(self.dictionary[seq_feature].values())))
        if os.path.exists(self.word2vec_file):
            self.word_embeddings = np.load(self.word2vec_file)
            logging.info('load word embeddings:', self.word_embeddings.shape)

    def _select_features(self):
        exclude_features = [self.label] + self.seqs_features
        if not self.include_id:
            exclude_features += ['uid', 'iid']
        self.features = [c for c in self.column_max.keys() if c not in exclude_features]
        logging.info("# of features: %d" % len(self.features))

        self.feature_dims = 0
        self.feature_min, self.feature_max = [], []
        for f in self.features:
            self.feature_min.append(self.feature_dims)
            self.feature_dims += int(self.column_max[f] + 1)
            self.feature_max.append(self.feature_dims - 1)
        logging.info("# of feature dims: %d" % self.feature_dims)

        self.user_num, self.item_num = -1, -1
        self.user_features, self.item_features = [], []
        if 'uid' in self.column_max:
            self.user_num = int(self.column_max['uid'] + 1)
            logging.info("# of users: %d" % self.user_num)
            if self.include_id:
                self.user_features = [f for f in self.column_max.keys() if f.startswith('u')]
            else:
                self.user_features = [f for f in self.column_max.keys() if f.startswith('u_')]
            logging.info("# of user features: %d" % len(self.user_features))
        if 'iid' in self.column_max:
            self.item_num = int(self.column_max['iid'] + 1)
            logging.info("# of items: %d" % self.item_num)
            if self.include_id:
                self.item_features = [f for f in self.column_max.keys() if f.startswith('i')]
            else:
                self.item_features = [f for f in self.column_max.keys() if f.startswith('i_')]
            logging.info("# of item features: %d" % len(self.item_features))

    def format_data(self, df):
        df = df.copy()
        if self.label in df.columns:
            data = {'Y': np.array(df[self.label], dtype=np.float32)}
            df.drop([self.label], axis=1, inplace=True)
        else:
            logging.warning('No Labels In Data: ' + self.label)
            data = {'Y': np.zeros(len(df), dtype=np.float32)}
        ui_id = []
        if self.user_num > 0:
            ui_id.append('uid')
        if self.item_num > 0:
            ui_id.append('iid')
        ui_id = df[ui_id]

        base = 0
        for feature in self.features:
            df[feature] = df[feature].apply(lambda x: x + base)
            base += int(self.column_max[feature] + 1)

        if self.append_id:
            x = pd.concat([ui_id, df[self.features]], axis=1)
            data['X'] = x.values
        else:
            data['X'] = df[self.features].values

        for seq_feature in self.seqs_features:
            seqs = df[seq_feature].str.split(',')
            data[seq_feature + '_length'] = np.array(seqs.apply(lambda x: len(x)), dtype=np.int32)
            max_length = self.column_max[seq_feature]
            if not self.seqs_expand:
                seqs = seqs.apply(lambda x: np.array([int(n) for n in x]))
            else:
                seqs = seqs.apply(lambda x: [int(n) for n in x] + [0] * (max_length - len(x)))
            data[seq_feature] = np.array(seqs.tolist())
            # print(data[seq_feature])
            # print(data[seq_feature + '_length'])
        # data['X'] = np.array(data['X'], dtype=np.float32)
        # print(data['X'])
        # print(data['X'].shape)
        # print len(data['Y'])
        return data

    @staticmethod
    def filter_data(df, info=None):
        return df


def main():
    BaseDataLoader('../dataset/', 'test', sep='\t', label='rating', include_id=True, append_id=True,
                   seqs_features=['seq'], seqs_sep=',', seqs_expand=True)
    # data = LoadData('../dataset/', 'ml-100k-ci', label=args.label, sep=args.sep, append_id=True, include_id=False)
    return


if __name__ == '__main__':
    main()
