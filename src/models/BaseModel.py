# coding=utf-8

import tensorflow as tf
import logging
from sklearn.metrics import *
import numpy as np


class BaseModel(object):
    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        parser.add_argument('--model_path', type=str,
                            default='../model/%s/%s.ckpt' % (model_name, model_name),
                            help='Model save path.')
        return parser

    @staticmethod
    def add_model_args(args):
        args.append_id = False
        args.include_id = True
        return args

    @staticmethod
    def evaluate_method(p, l, metrics):
        evaluations = []
        for metric in metrics:
            if metric == 'rmse':
                evaluations.append(np.sqrt(mean_squared_error(l, p)))
            elif metric == 'mae':
                evaluations.append(mean_absolute_error(l, p))
            elif metric == 'auc':
                evaluations.append(roc_auc_score(l, p))
            elif metric == 'f1':
                evaluations.append(f1_score(l, p))
            elif metric == 'accuracy':
                evaluations.append(accuracy_score(l, p))
            elif metric == 'precision':
                evaluations.append(precision_score(l, p))
            elif metric == 'recall':
                evaluations.append(recall_score(l, p))
        return evaluations

    def __init__(self, class_num, feature_num, random_seed=2018, model_path='../model/Model/Model.ckpt'):
        self.class_num = class_num
        self.feature_num = feature_num
        self.random_seed = random_seed
        self.model_path = model_path
        self.d_type = tf.float32
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

        self.check_list = []
        self.lrp_list = []

        self._init_placeholders()
        self.weights = self._init_weights()

        self._init_graph()
        self._init_lrp()
        self._init_loss()
        self._init_saver()

        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)
        
        self.optimizer = None
        self.init, self.sess = None, None
        if len(self.check_list) == 0:
            self.check_list.append(('loss', self.loss))
            self.check_list.append(('l2', self.l2))

    def _init_placeholders(self):
        with self.graph.as_default():
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * [u_id, i_id, u_fs, i_fs]
            self.train_labels = tf.placeholder(dtype=self.d_type, shape=[None])  # None
            self.dropout_keep = tf.placeholder(dtype=self.d_type)
            self.train_phase = tf.placeholder(tf.bool)

    def _init_weights(self):
        all_weights = dict()
        with self.graph.as_default():
            all_weights['w'] = tf.Variable(
                tf.random_normal([self.feature_num, 1], 0.0, 0.01, dtype=self.d_type), name='w')
            all_weights['b'] = tf.Variable(tf.random_normal([1, 1], 0.0, 0.01, dtype=self.d_type), name='b')
        return all_weights

    def count_variables(self):
        with self.graph.as_default():
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
        return total_parameters

    def _init_graph(self):
        with self.graph.as_default():
            self.out_matmul = tf.matmul(tf.cast(self.train_features, self.d_type), self.weights['w'])
            self.prediction = tf.add(self.out_matmul, self.weights['b'])
            self.prediction = tf.reduce_sum(self.prediction, axis=1)

    def _init_loss(self):
        with self.graph.as_default():
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.train_labels, self.prediction))))
            self.loss = self.rmse
            # must not be tf.global_variables() !!!
            self.l2_weights = list(tf.trainable_variables())
            self.var_list = list(tf.trainable_variables())
            self.l2 = 0
            for weights in self.l2_weights:
                self.l2 += tf.reduce_sum(tf.square(weights))

    def _init_saver(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(tf.global_variables())

    def _init_lrp(self):
        pass
