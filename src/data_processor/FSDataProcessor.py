# coding=utf-8
import copy
from utils import utils
from data_processor.BaseDataProcessor import BaseDataProcessor
import numpy as np
from collections import Counter
import logging


class FSDataProcessor(BaseDataProcessor):
    @staticmethod
    def parse_dp_args(parser):
        parser.add_argument('--fs_ratio', type=float, default=0.1,
                            help='Feature-Sampling ratio of each batch.')
        parser.add_argument('--fs_mode', type=str, default='afs',
                            help='Feature-Sampling model.')
        return parser

    def __init__(self, data_loader, model, runner, fs_ratio, mode='afs'):
        self.data_loader = data_loader
        self.model = model
        self.runner = runner
        self.mode = mode
        self.fs_ratio = fs_ratio

    def self_paced_fs_ratio(self, value, epoch):
        # base = 5
        # return value * (1 - base * base / (epoch * epoch + base * base))
        start, max_epoch = min(0.1, value), 21
        per = (value - start) / max_epoch
        epoch = min(epoch, max_epoch)
        return start + per * (epoch - 1)

    def self_paced_fp(self, value, epoch):
        if np.sum(value) >= 0:
            max_v = (np.max(value) * 1.1) / (1 + np.exp(-epoch + 3))
            result = value * ((value <= max_v) * 0.1 + (value > max_v) * 1) + 1e-5
            result = result / np.sum(result)
            return result
        return value

    def epoch_process_train(self, data, epoch):
        if data is None:
            return data
        fs_ratio = self.self_paced_fs_ratio(self.fs_ratio, epoch)
        if fs_ratio == 0:
            data['drop_pos'] = np.zeros(data['X'].shape, dtype=np.float32)
            return BaseDataProcessor.epoch_process_train(self, data, epoch)

        feature_num = self.model.feature_num
        feature_min, feature_max, base = [], [], 0
        for f in self.data_loader.features:
            feature_min.append(base)
            base += int(self.data_loader.column_max[f] + 1)
            feature_max.append(base - 1)

        if self.mode == 'rfs':
            logging.info('epoch_process_train: rfs')
            col_bias = data['X'].shape[1] - feature_num
            xs = data['X']
            fs_size = int(self.fs_ratio * xs.shape[0] * feature_num)
            rows = np.random.choice(list(range(xs.shape[0])) * feature_num, size=fs_size, replace=False)
            rows_cnt = Counter(rows)
            logging.info('fs size: %d' % fs_size)
            for i in range(xs.shape[0]):
                if rows_cnt[i] > 0:
                    cols = np.random.choice(range(feature_num), size=rows_cnt[i], replace=False)
                    for col in cols:
                        xs[i][col_bias + col] = feature_min[col]
            data['X'] = xs
            drop_pos = np.zeros(data['X'].shape, dtype=np.float32)
        elif self.mode == 'afs':
            logging.info('epoch_process_train: afs')
            lrp_results = self.runner.lrp(self.model, data)
            lrp_result = lrp_results[0]
            col_bias = data['X'].shape[1] - feature_num
            for i in range(col_bias):
                lrp_result[:, i] = 0
            f_p = np.mean(np.abs(lrp_result), axis=0) + 1e-5
            # f_p = np.nan_to_num(f_p)
            f_p = f_p / np.sum(f_p)
            f_p = self.self_paced_fp(f_p, epoch)
            xs = data['X']
            fs_size = int(fs_ratio * xs.shape[0] * feature_num)
            rows = np.random.choice(list(range(xs.shape[0])) * feature_num, size=fs_size, replace=False)
            rows_cnt = Counter(rows)
            logging.info('fs size: %d' % fs_size)
            for i in range(xs.shape[0]):
                if rows_cnt[i] > 0:
                    cols = np.random.choice(range(xs.shape[1]), size=min(rows_cnt[i], np.count_nonzero(f_p)),
                                            replace=False, p=f_p)
                    for col in cols:
                        xs[i][col] = feature_min[col - col_bias]
            data['X'] = xs
            drop_pos = np.zeros(data['X'].shape, dtype=np.float32)
        elif self.mode == 'afs-ui':
            logging.info('epoch_process_train: afs-ui')
            col_bias = data['X'].shape[1] - feature_num
            lrp_results = self.runner.lrp(self.model, data)
            lrp_result = lrp_results[0]
            for i in range(col_bias):
                lrp_result[:, i] = 0
            f_p = np.abs(lrp_result) + 1e-5
            # f_p = np.nan_to_num(f_p)
            f_p = f_p / np.sum(f_p, axis=1, keepdims=True)
            f_p = np.array(
                [self.self_paced_fp(f_p[i], epoch) for i in range(len(f_p))])
            xs = data['X']
            fs_size = int(fs_ratio * xs.shape[0] * feature_num)
            rows = np.random.choice(list(range(xs.shape[0])) * feature_num, size=fs_size, replace=False)
            rows_cnt = Counter(rows)
            logging.info('fs size: %d' % fs_size)
            for i in range(xs.shape[0]):
                if rows_cnt[i] > 0:
                    # try:
                    #     cols = np.random.choice(range(xs.shape[1]), size=min(rows_cnt[i], np.count_nonzero(f_p[i])),
                    #                             replace=False, p=f_p[i])
                    # except:
                    #     print(f_p[i], xs[i], lrp_results[2][i], lrp_results[3][i], lrp_results[4][i], lrp_results[5][i])
                    #     assert 1 == 2
                    cols = np.random.choice(range(xs.shape[1]), size=min(rows_cnt[i], np.count_nonzero(f_p[i])),
                                            replace=False, p=f_p[i])
                    for col in cols:
                        xs[i][col] = feature_min[col - col_bias]
            data['X'] = xs
            drop_pos = np.zeros(data['X'].shape, dtype=np.float32)
        else:
            drop_pos = np.zeros(data['X'].shape, dtype=np.float32)
        data['drop_pos'] = drop_pos
        return BaseDataProcessor.epoch_process_train(self, data, epoch)
