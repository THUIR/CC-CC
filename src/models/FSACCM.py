# coding=utf-8

import tensorflow as tf
from models.ACCM import ACCM
from utils import utils


class FSACCM(ACCM):
    def _init_placeholders(self):
        with self.graph.as_default():
            self.drop_pos = tf.placeholder(self.d_type, shape=[None, None])
            ACCM._init_placeholders(self)

    def _init_graph(self):
        with self.graph.as_default():
            # Model.
            u_ids = self.train_features[:, 0]
            i_ids = self.train_features[:, 1]

            # cold sampling
            drop_u_pos = tf.cast(tf.multinomial(tf.log([[1 - self.cs_ratio, self.cs_ratio]]),
                                                tf.shape(u_ids)[0]), dtype=self.d_type)
            drop_i_pos = tf.cast(tf.multinomial(tf.log([[1 - self.cs_ratio, self.cs_ratio]]),
                                                tf.shape(i_ids)[0]), dtype=self.d_type)
            drop_u_pos = tf.reshape(drop_u_pos, shape=[-1])
            drop_i_pos = tf.reshape(drop_i_pos, shape=[-1])
            drop_u_pos_zero = tf.zeros(shape=tf.shape(drop_u_pos), dtype=self.d_type)
            drop_i_pos_zero = tf.zeros(shape=tf.shape(drop_i_pos), dtype=self.d_type)

            drop_u_pos = tf.cond(self.train_phase, lambda: drop_u_pos, lambda: drop_u_pos_zero)
            drop_i_pos = tf.cond(self.train_phase, lambda: drop_i_pos, lambda: drop_i_pos_zero)
            drop_u_pos_v = tf.reshape(drop_u_pos, shape=[-1, 1])
            drop_i_pos_v = tf.reshape(drop_i_pos, shape=[-1, 1])
            drop_f_pos = tf.expand_dims(self.drop_pos[:, 2:], axis=2)

            # bias
            self.u_bias = tf.nn.embedding_lookup(self.weights['user_bias'], u_ids) * (1 - drop_u_pos)
            self.i_bias = tf.nn.embedding_lookup(self.weights['item_bias'], i_ids) * (1 - drop_i_pos)

            # cf part
            cf_u_vectors = tf.nn.embedding_lookup(self.weights['uid_embeddings'], u_ids)
            cf_i_vectors = tf.nn.embedding_lookup(self.weights['iid_embeddings'], i_ids)

            random_u_vectors = tf.random_normal(tf.shape(cf_u_vectors), 0, 0.01, dtype=self.d_type)
            random_i_vectors = tf.random_normal(tf.shape(cf_i_vectors), 0, 0.01, dtype=self.d_type)

            cf_u_vectors = random_u_vectors * drop_u_pos_v + cf_u_vectors * (1 - drop_u_pos_v)
            cf_i_vectors = random_i_vectors * drop_i_pos_v + cf_i_vectors * (1 - drop_i_pos_v)

            self.cf_prediction = tf.reduce_sum(tf.multiply(cf_u_vectors, cf_i_vectors), axis=1)

            # cb part
            f_vectors = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features[:, 2:])
            random_f_vectors = tf.random_normal(tf.shape(f_vectors), 0, 0.01, dtype=self.d_type)
            f_vectors = f_vectors * (1 - drop_f_pos) + random_f_vectors * drop_f_pos
            uf_vectors = f_vectors[:, :self.user_feature_num]
            if_vectors = f_vectors[:, self.user_feature_num:]
            uf_layer = tf.reshape(uf_vectors, (-1, self.f_vector_size * self.user_feature_num))
            if_layer = tf.reshape(if_vectors, (-1, self.f_vector_size * self.item_feature_num))

            self.lrp_layers_u, self.lrp_layers_i = [uf_layer], [if_layer]
            for i in range(0, len(self.cb_hidden_layers) + 1):
                uf_layer = tf.add(tf.matmul(uf_layer, self.weights['cb_user_layer_%d' % i]),
                                  self.weights['cb_user_bias_%d' % i])
                if_layer = tf.add(tf.matmul(if_layer, self.weights['cb_item_layer_%d' % i]),
                                  self.weights['cb_item_bias_%d' % i])
                if i < len(self.cb_hidden_layers):
                    uf_layer = tf.layers.batch_normalization(uf_layer, training=self.train_phase, name='u_bn_%d' % i)
                    uf_layer = tf.nn.relu(uf_layer)
                    uf_layer = tf.nn.dropout(uf_layer, self.dropout_keep)
                    if_layer = tf.layers.batch_normalization(if_layer, training=self.train_phase, name='i_bn_%d' % i)
                    if_layer = tf.nn.relu(if_layer)
                    if_layer = tf.nn.dropout(if_layer, self.dropout_keep)
                    self.lrp_layers_u.append(uf_layer)
                    self.lrp_layers_i.append(if_layer)
            cb_u_vectors, cb_i_vectors = uf_layer, if_layer
            self.cb_prediction = tf.reduce_sum(tf.multiply(cb_u_vectors, cb_i_vectors), axis=1)

            # attention
            ah_cf_u = tf.add(tf.matmul(cf_u_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            ah_cf_u = tf.tanh(ah_cf_u)
            # ah_cf_u = tf.nn.relu(ah_cf_u)
            a_cf_u = tf.reduce_sum(tf.multiply(ah_cf_u, self.weights['attention_pre']), axis=1)
            # a_cf_u = tf.minimum(tf.maximum(a_cf_u, -10), 10)
            a_cf_u = tf.exp(a_cf_u)
            ah_cb_u = tf.add(tf.matmul(cb_u_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            ah_cb_u = tf.tanh(ah_cb_u)
            # ah_cb_u = tf.nn.relu(ah_cb_u)
            a_cb_u = tf.reduce_sum(tf.multiply(ah_cb_u, self.weights['attention_pre']), axis=1)
            a_cb_u = tf.exp(a_cb_u)
            # a_cb_u = tf.minimum(tf.maximum(a_cb_u, -10), 10)
            a_sum = a_cf_u + a_cb_u

            self.a_cf_u = tf.reshape(a_cf_u / a_sum, shape=[-1, 1])
            self.a_cb_u = tf.reshape(a_cb_u / a_sum, shape=[-1, 1])

            ah_cf_i = tf.add(tf.matmul(cf_i_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            ah_cf_i = tf.tanh(ah_cf_i)
            # ah_cf_i = tf.nn.relu(ah_cf_i)
            a_cf_i = tf.reduce_sum(tf.multiply(ah_cf_i, self.weights['attention_pre']), axis=1)
            # a_cf_i = tf.minimum(tf.maximum(a_cf_i, -10), 10)
            a_cf_i = tf.exp(a_cf_i)
            ah_cb_i = tf.add(tf.matmul(cb_i_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            ah_cb_i = tf.tanh(ah_cb_i)
            # ah_cb_i = tf.nn.relu(ah_cb_i)
            a_cb_i = tf.reduce_sum(tf.multiply(ah_cb_i, self.weights['attention_pre']), axis=1)
            a_cb_i = tf.exp(a_cb_i)
            # a_cb_i = tf.minimum(tf.maximum(a_cb_i, -10), 10)
            a_sum = a_cf_i + a_cb_i

            self.a_cf_i = tf.reshape(a_cf_i / a_sum, shape=[-1, 1])
            self.a_cb_i = tf.reshape(a_cb_i / a_sum, shape=[-1, 1])

            # prediction
            self.bias = self.u_bias + self.i_bias + self.weights['global_bias']

            # self.debug = cf_u_vectors
            self.u_vector = self.a_cf_u * cf_u_vectors + self.a_cb_u * cb_u_vectors
            self.i_vector = self.a_cf_i * cf_i_vectors + self.a_cb_i * cb_i_vectors
            self.prediction = self.bias + tf.reduce_sum(tf.multiply(self.u_vector, self.i_vector), axis=1)
