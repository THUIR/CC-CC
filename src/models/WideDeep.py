# coding=utf-8

import tensorflow as tf
from models.DeepModel import DeepModel
from utils import utils


class WideDeep(DeepModel):
    def _init_graph(self):
        with self.graph.as_default():
            # Model.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            # self.debug = self.train_features
            self.debug = nonzero_embeddings

            pre_layer = tf.reshape(nonzero_embeddings, shape=[-1, self.feature_num * self.f_vector_size])
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
            # self.check_list.append(('cross_bias', self.cross_bias))
            # self.check_list.append(('prediction', self.prediction))

    def _init_weights(self):
        all_weights = dict()
        with self.graph.as_default():
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.feature_dims, self.f_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='feature_embeddings')  # feature_dims * f_vector_size
            # all_weights['cross_bias'] = tf.Variable(
            #     tf.random_normal([self.feature_dims * self.feature_dims, 1], 0.0, 0.01, dtype=self.d_type)
            #     , name='cross_bias')  # feature_dims^2 * 1
            all_weights['cross_bias'] = tf.Variable(
                tf.random_normal([self.feature_dims, 1], 0.0, 0.01, dtype=self.d_type)
                , name='cross_bias')  # feature_dims * 1
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
                tf.random_normal([pre_size, 1], 0.0, 0.01, dtype=self.d_type), name='prediction')
        return all_weights

    def _init_lrp(self):
        with self.graph.as_default():
            # print(tf.transpose(self.weights['w']))
            prediction = self.prediction + \
                         1e-5 * tf.cast(tf.logical_or(tf.equal(self.prediction, 0), tf.is_nan(self.prediction)),
                                        self.d_type)
            r_wide = tf.reshape(self.cross_bias / prediction, shape=[-1, 1])
            r_deep = 1 - r_wide

            f_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['cross_bias'], self.train_features), axis=2)
            cross_bias = self.cross_bias + \
                         1e-5 * tf.cast(tf.logical_or(tf.equal(self.cross_bias, 0), tf.is_nan(self.cross_bias)),
                                        self.d_type)
            r_wide_f = (f_bias / tf.reshape(cross_bias, shape=[-1, 1])) * r_wide
            # self.lrp_list.extend([tf.reduce_sum(r_wide_f, axis=1, keep_dims=True) - r_wide])

            for i in range(len(self.layers) + 1):
                ix = len(self.layers) - i
                if i == 0:
                    weights = self.weights['prediction']
                else:
                    weights = self.weights['layer_%d' % ix]
                z = tf.expand_dims(self.lrp_layers[ix], axis=1) * tf.transpose(weights)
                mo = tf.reduce_sum(z, axis=2, keep_dims=True)
                mo = mo + 1e-5 * tf.cast(tf.logical_or(tf.equal(mo, 0), tf.is_nan(mo)), self.d_type)
                w = z / mo
                # w = tf.where(tf.is_nan(w), tf.zeros_like(w), w)
                r_deep = tf.expand_dims(r_deep, axis=1)
                r_deep = tf.matmul(r_deep, w)
                r_deep = tf.reduce_sum(r_deep, axis=1)

            r_deep_fs = []
            for i in range(self.feature_num):
                start, end = i * self.f_vector_size, (i + 1) * self.f_vector_size
                r_deep_fs.append(tf.reduce_sum(r_deep[:, start:end], axis=1, keep_dims=True))
            r_all = tf.concat(r_deep_fs, axis=1) + r_wide_f

            self.lrp_list.extend([r_all, tf.reduce_sum(r_all, axis=1, keep_dims=True)])
            # self.lrp_list.append(r_all)
            # self.lrp_list.extend([self.a_cf_u, self.a_cf_i, r_wide_f, tf.reduce_sum(r_wide_f, axis=1, keep_dims=True)])
