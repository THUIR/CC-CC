# coding=utf-8

import tensorflow as tf
from models.NFM import NFM
from utils import utils


class FSNFM(NFM):
    def _init_placeholders(self):
        with self.graph.as_default():
            self.drop_pos = tf.placeholder(self.d_type, shape=[None, None])
            NFM._init_placeholders(self)

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

            summed_features_emb = tf.reduce_sum(f_vectors, axis=1)  # None * K
            # get the element-multiplication
            summed_features_emb_square = tf.square(summed_features_emb)  # None * K

            # _________ square_sum part _____________
            squared_features_emb = tf.square(f_vectors)
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
