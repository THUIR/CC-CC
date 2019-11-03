# coding=utf-8

import tensorflow as tf
from models.WideDeep import WideDeep
from utils import utils


class FSWideDeep(WideDeep):
    def _init_placeholders(self):
        with self.graph.as_default():
            self.drop_pos = tf.placeholder(self.d_type, shape=[None, None])
            WideDeep._init_placeholders(self)

    def _init_graph(self):
        with self.graph.as_default():
            # Model.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)

            # Feature Sampling
            drop_f_pos = tf.expand_dims(self.drop_pos, axis=2)
            random_f_vectors = tf.random_normal(tf.shape(nonzero_embeddings), 0, 0.01, dtype=self.d_type)
            f_vectors = nonzero_embeddings * (1 - drop_f_pos) + random_f_vectors * drop_f_pos
            # self.check_list.append(('nonzero_embeddings', nonzero_embeddings))
            # self.check_list.append(('drop_f_pos', drop_f_pos))
            # self.check_list.append(('random_f_vectors', random_f_vectors))
            # self.check_list.append(('f_vectors', f_vectors))

            # self.debug = self.train_features
            self.debug = nonzero_embeddings

            pre_layer = tf.reshape(f_vectors, shape=[-1, self.feature_num * self.f_vector_size])

            self.lrp_layers = [pre_layer]
            # ________ Deep Layers __________
            for i in range(0, len(self.layers)):
                pre_layer = tf.add(tf.matmul(pre_layer, self.weights['layer_%d' % i]),
                                   self.weights['bias_%d' % i])  # None * layer[i] * 1
                pre_layer = tf.layers.batch_normalization(pre_layer, training=self.train_phase, name='bn_%d' % i)
                pre_layer = tf.nn.relu(pre_layer)
                pre_layer = tf.nn.dropout(pre_layer, self.dropout_keep)  # dropout at each Deep layer
                self.lrp_layers.append(pre_layer)
            pre_layer = tf.matmul(pre_layer, self.weights['prediction'])  # None * 1
            deep_part = tf.reduce_sum(pre_layer, axis=1)

            # cross part
            # tmp = tf.reshape(self.train_features, [-1, self.feature_num, 1])
            # tmp *= tf.ones([1, self.feature_num], tf.int32) * self.feature_dims
            # tmp += tf.expand_dims(self.train_features, axis=1)
            # cross_features = tf.reshape(tmp, [-1, self.feature_num * self.feature_num])
            # cross_bias = tf.reduce_sum(
            #     tf.reduce_sum(tf.nn.embedding_lookup(self.weights['cross_bias'], cross_features), axis=1),
            #     axis=1)
            self.cross_bias = tf.reduce_sum(
                tf.reduce_sum(tf.nn.embedding_lookup(self.weights['cross_bias'], self.train_features), axis=1),
                axis=1)

            # _________out _________
            self.prediction = deep_part + self.cross_bias
            # self.prediction = tf.sigmoid(self.prediction)
