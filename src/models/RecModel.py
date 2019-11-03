# coding=utf-8

import tensorflow as tf
from models.BaseModel import BaseModel
from utils import utils


class RecModel(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='RecModel'):
        parser.add_argument('--u_vector_size', type=int, default=64,
                            help='Size of user vectors.')
        parser.add_argument('--i_vector_size', type=int, default=64,
                            help='Size of item vectors.')
        return BaseModel.parse_model_args(parser, model_name)

    @staticmethod
    def add_model_args(args):
        args.append_id = True
        args.include_id = False
        return args

    def __init__(self, class_num, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 random_seed, model_path):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num
        BaseModel.__init__(self, class_num=class_num, feature_num=feature_num, random_seed=random_seed,
                           model_path=model_path)

    def _init_graph(self):
        with self.graph.as_default():
            # Model.
            u_ids = self.train_features[:, 0]
            i_ids = self.train_features[:, 1]

            # cf part
            cf_u_vectors = tf.nn.embedding_lookup(self.weights['uid_embeddings'], u_ids)
            cf_i_vectors = tf.nn.embedding_lookup(self.weights['iid_embeddings'], i_ids)
            self.cf_prediction = tf.reduce_sum(tf.multiply(cf_u_vectors, cf_i_vectors), axis=1)
            self.cf_vector = tf.concat(values=[cf_u_vectors, cf_i_vectors], axis=1)

            # prediction
            self.prediction = self.cf_prediction

    def _init_weights(self):
        all_weights = dict()
        with self.graph.as_default():
            all_weights['uid_embeddings'] = tf.Variable(
                tf.random_normal([self.user_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='uid_embeddings')  # user_num * ui_vector_size
            all_weights['iid_embeddings'] = tf.Variable(
                tf.random_normal([self.item_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='iid_embeddings')  # item_num * ui_vector_size
        return all_weights
