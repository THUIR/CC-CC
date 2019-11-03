# coding=utf-8

import tensorflow as tf
from models.RecModel import RecModel
from utils import utils


class BiasedMF(RecModel):
    def _init_graph(self):
        with self.graph.as_default():
            # Model.
            u_ids = self.train_features[:, 0]
            i_ids = self.train_features[:, 1]

            u_bias = tf.nn.embedding_lookup(self.weights['user_bias'], u_ids)
            i_bias = tf.nn.embedding_lookup(self.weights['item_bias'], i_ids)

            # cf part
            cf_u_vectors = tf.nn.embedding_lookup(self.weights['uid_embeddings'], u_ids)

            cf_i_vectors = tf.nn.embedding_lookup(self.weights['iid_embeddings'], i_ids)
            self.cf_prediction = tf.reduce_sum(tf.multiply(cf_u_vectors, cf_i_vectors), axis=1)
            self.cf_vector = tf.concat(values=[cf_u_vectors, cf_i_vectors], axis=1)

            # prediction
            self.prediction = u_bias + i_bias + self.weights['global_bias']
            self.prediction += self.cf_prediction

    def _init_weights(self):
        all_weights = dict()
        with self.graph.as_default():
            all_weights['user_bias'] = tf.Variable(tf.constant(0.1, shape=[self.user_num], dtype=self.d_type))
            all_weights['item_bias'] = tf.Variable(tf.constant(0.1, shape=[self.item_num], dtype=self.d_type))
            all_weights['global_bias'] = tf.Variable(0.1, dtype=self.d_type)

            all_weights['uid_embeddings'] = tf.Variable(
                tf.random_normal([self.user_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='uid_embeddings')  # user_num * ui_vector_size
            all_weights['iid_embeddings'] = tf.Variable(
                tf.random_normal([self.item_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='iid_embeddings')  # item_num * ui_vector_size
        return all_weights
