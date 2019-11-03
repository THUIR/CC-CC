# coding=utf-8

import tensorflow as tf
import logging
from sklearn.metrics import *
import numpy as np
from models.BaseModel import BaseModel
from utils import utils


class DeepModel(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='DeepModel'):
        parser.add_argument('--f_vector_size', type=int, default=64,
                            help='Size of feature vectors.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, class_num, feature_num, feature_dims, f_vector_size, layers,
                 random_seed, model_path):
        self.feature_dims = feature_dims
        self.f_vector_size = f_vector_size
        self.layers = layers
        BaseModel.__init__(self, class_num=class_num, feature_num=feature_num, random_seed=random_seed,
                           model_path=model_path)

    def _init_graph(self):
        with self.graph.as_default():
            # Model.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            # self.debug = self.train_features
            self.debug = nonzero_embeddings

            pre_layer = tf.reshape(nonzero_embeddings, shape=[-1, self.feature_num * self.f_vector_size])

            # ________ Deep Layers __________
            for i in range(0, len(self.layers)):
                pre_layer = tf.add(tf.matmul(pre_layer, self.weights['layer_%d' % i]),
                                   self.weights['bias_%d' % i])  # None * layer[i] * 1
                pre_layer = tf.layers.batch_normalization(pre_layer, training=self.train_phase, name='bn_%d' % i)
                pre_layer = tf.nn.relu(pre_layer)
                pre_layer = tf.nn.dropout(pre_layer, self.dropout_keep)  # dropout at each Deep layer
            pre_layer = tf.add(tf.matmul(pre_layer, self.weights['prediction']),
                               self.weights['prediction_bias'])  # None * 1
            deep_part = tf.reduce_sum(pre_layer, axis=1)

            # _________out _________
            self.prediction = deep_part

    def _init_weights(self):
        all_weights = dict()
        with self.graph.as_default():
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.feature_dims, self.f_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='feature_embeddings')  # feature_dims * f_vector_size
            pre_size = self.f_vector_size * self.feature_num
            for i, layer_size in enumerate(self.layers):
                all_weights['layer_%d' % i] = tf.Variable(
                    tf.random_normal([pre_size, self.layers[i]], 0.0, 0.01, dtype=self.d_type),
                    name='layer_%d' % i)
                all_weights['bias_%d' % i] = tf.Variable(
                    tf.random_normal([1, self.layers[i]], 0.0, 0.01, dtype=self.d_type),
                    name='bias_%d' % i)
                pre_size = self.layers[i]

            all_weights['prediction'] = tf.Variable(
                tf.random_normal([pre_size, 1], 0.0, 0.01, dtype=self.d_type))
            all_weights['prediction_bias'] = tf.Variable(
                tf.random_normal([1, 1], 0.0, 0.01, dtype=self.d_type),
                name='prediction_bias')
        return all_weights
