# coding=utf-8
import logging
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
import tensorflow as tf
from scipy.stats import chi2_contingency

LOWER_METRIC_LIST = ["rmse", 'mae']


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='../log/log.txt',
                        help='Logging file path')
    parser.add_argument('--result_file', type=str, default='../result/result.npy',
                        help='Result file path')
    parser.add_argument('--random_seed', type=int, default=2018,
                        help='Random seed of numpy and tensorflow.')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    return parser


def balance_data(data):
    pos_indexes = np.where(data['Y'] == 1)[0]
    copy_num = int((len(data['Y']) - len(pos_indexes)) / len(pos_indexes))
    if copy_num > 1:
        copy_indexes = np.tile(pos_indexes, copy_num)
        sample_index = np.concatenate([np.arange(0, len(data['Y'])), copy_indexes])
        for k in data:
            data[k] = data[k][sample_index]
    return data


def input_data_is_list(data):
    if type(data) is list or type(data) is tuple:
        print("input_data_is_list")
        new_data = {}
        for k in data[0]:
            new_data[k] = np.concatenate([d[k] for d in data])
        return new_data
    return data


def format_metric(metric):
    # print(metric, type(metric))
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            # print(type(m))
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)


def shuffle_in_unison_scary(data):
    rng_state = np.random.get_state()
    for d in data:
        np.random.set_state(rng_state)
        np.random.shuffle(data[d])
    return data


def best_result(metric, results_list):
    if metric in LOWER_METRIC_LIST:
        return min(results_list)
    return max(results_list)


def eva_termination(metric, valid):
    if len(valid) > 20 and metric in LOWER_METRIC_LIST and \
            valid[-1] >= valid[-2] >= valid[-3] >= valid[-4] >= valid[-5]:
        return True
    elif len(valid) > 20 and metric not in LOWER_METRIC_LIST and \
            valid[-1] <= valid[-2] <= valid[-3] <= valid[-4] <= valid[-5]:
        return True
    return False


# def batch_norm_layer(x, train_phase, scope_bn):
#     bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
#                           is_training=True, reuse=None, trainable=True, scope=scope_bn)
#     bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
#                               is_training=False, reuse=True, trainable=True, scope=scope_bn)
#     z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
#     return z


