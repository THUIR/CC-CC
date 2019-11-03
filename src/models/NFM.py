# coding=utf-8

import tensorflow as tf
from models.DeepModel import DeepModel
from utils import utils


class NFM(DeepModel):
    def _init_graph(self):
        with self.graph.as_default():
            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)

            summed_features_emb = tf.reduce_sum(nonzero_embeddings, axis=1)  # None * K
            # get the element-multiplication
            summed_features_emb_square = tf.square(summed_features_emb)  # None * K

            # _________ square_sum part _____________
            squared_features_emb = tf.square(nonzero_embeddings)
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1)  # None * K

            # ________ fm __________
            fm = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
            fm = tf.layers.batch_normalization(fm, training=self.train_phase, name='bn_fm')
            fm = tf.nn.dropout(fm, self.dropout_keep)  # dropout at the bilinear interactin layer

            # ________ Deep Layers __________
            for i in range(0, len(self.layers)):
                fm = tf.add(tf.matmul(fm, self.weights['layer_%d' % i]),
                            self.weights['bias_%d' % i])  # None * layer[i] * 1
                fm = tf.layers.batch_normalization(fm, training=self.train_phase,
                                                   name='bn_%d' % i)  # None * layer[i] * 1
                fm = tf.nn.relu(fm)
                fm = tf.nn.dropout(fm, self.dropout_keep)  # dropout at each Deep layer
            fm = tf.matmul(fm, self.weights['prediction'])  # None * 1

            # _________out _________
            bilinear = tf.reduce_sum(fm, axis=1)
            feature_bias = tf.reduce_sum(
                tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features), axis=1),
                axis=1)
            bias = self.weights['bias']
            self.prediction = bilinear + feature_bias + bias

    def _init_weights(self):
        all_weights = dict()
        with self.graph.as_default():
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.feature_dims, self.f_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='feature_embeddings')  # feature_dims * f_vector_size
            all_weights['feature_bias'] = tf.Variable(
                tf.random_normal([self.feature_dims, 1], 0.0, 0.01, dtype=self.d_type)
                , name='feature_bias')  # feature_dims * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.0, dtype=self.d_type), name='bias')  # 1 * 1

            pre_size = self.f_vector_size
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
            r_all = tf.ones(shape=tf.shape(self.train_features), dtype=self.d_type) / self.feature_num
            self.lrp_list.append(r_all)
