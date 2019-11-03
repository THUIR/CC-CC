# coding=utf-8

import tensorflow as tf
from models.RecModel import RecModel
from utils import utils


class ACCM(RecModel):
    @staticmethod
    def parse_model_args(parser, model_name='ACCM'):
        parser.add_argument('--f_vector_size', type=int, default=64,
                            help='Size of feature vectors.')
        parser.add_argument('--cb_hidden_layers', type=str, default='[]',
                            help="Number of CB part's hidden layer.")
        parser.add_argument('--attention_size', type=int, default=16,
                            help='Size of attention layer.')
        parser.add_argument('--cs_ratio', type=float, default=0.1,
                            help='Cold-Sampling ratio of each batch.')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, class_num, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 user_feature_num, item_feature_num, feature_dims,
                 f_vector_size, cb_hidden_layers, attention_size, cs_ratio,
                 random_seed, model_path):
        self.user_feature_num = user_feature_num
        self.item_feature_num = item_feature_num
        self.feature_dims = feature_dims
        self.f_vector_size = f_vector_size
        self.cb_hidden_layers = cb_hidden_layers
        self.attention_size = attention_size
        self.cs_ratio = cs_ratio
        RecModel.__init__(self, class_num=class_num, feature_num=feature_num, user_num=user_num, item_num=item_num,
                          u_vector_size=u_vector_size, i_vector_size=i_vector_size,
                          random_seed=random_seed, model_path=model_path)

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
            u_fs = self.train_features[:, 2:2 + self.user_feature_num]
            i_fs = self.train_features[:, 2 + self.user_feature_num:]
            uf_vectors = tf.nn.embedding_lookup(self.weights['feature_embeddings'], u_fs)
            if_vectors = tf.nn.embedding_lookup(self.weights['feature_embeddings'], i_fs)
            uf_layer = tf.reshape(uf_vectors, (-1, self.f_vector_size * self.user_feature_num))
            if_layer = tf.reshape(if_vectors, (-1, self.f_vector_size * self.item_feature_num))

            self.lrp_layers_u, self.lrp_layers_i = [uf_layer], [if_layer]
            for i in range(0, len(self.cb_hidden_layers) + 1):
                uf_layer = tf.add(tf.matmul(uf_layer, self.weights['cb_user_layer_%d' % i]),
                                  self.weights['cb_user_bias_%d' % i])
                if_layer = tf.add(tf.matmul(if_layer, self.weights['cb_item_layer_%d' % i]),
                                  self.weights['cb_item_bias_%d' % i])
                uf_layer = tf.layers.batch_normalization(uf_layer, training=self.train_phase, name='u_bn_%d' % i)
                if_layer = tf.layers.batch_normalization(if_layer, training=self.train_phase, name='i_bn_%d' % i)
                if i < len(self.cb_hidden_layers):
                    uf_layer = tf.nn.relu(uf_layer)
                    uf_layer = tf.nn.dropout(uf_layer, self.dropout_keep)
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

    def _init_weights(self):
        all_weights = dict()
        with self.graph.as_default():
            all_weights['user_bias'] = tf.Variable(tf.constant(0.0, shape=[self.user_num], dtype=self.d_type))
            all_weights['item_bias'] = tf.Variable(tf.constant(0.0, shape=[self.item_num], dtype=self.d_type))
            all_weights['global_bias'] = tf.Variable(0.1, dtype=self.d_type)

            all_weights['uid_embeddings'] = tf.Variable(
                tf.random_normal([self.user_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='uid_embeddings')  # user_num * ui_vector_size
            all_weights['iid_embeddings'] = tf.Variable(
                tf.random_normal([self.item_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='iid_embeddings')  # item_num * ui_vector_size

            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.feature_dims, self.f_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='feature_embeddings')  # feature_dims * f_vector_size

            user_pre_size = self.user_feature_num * self.f_vector_size
            for i, layer_size in enumerate(self.cb_hidden_layers):
                all_weights['cb_user_layer_%d' % i] = tf.Variable(
                    tf.random_normal([user_pre_size, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                    name='cb_user_layer_%d' % i)
                all_weights['cb_user_bias_%d' % i] = tf.Variable(
                    tf.random_normal([1, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                    name='cb_user_bias_%d' % i)
                user_pre_size = self.cb_hidden_layers[i]
            all_weights['cb_user_layer_%d' % len(self.cb_hidden_layers)] = tf.Variable(
                tf.random_normal([user_pre_size, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='cb_user_layer_%d' % len(self.cb_hidden_layers))
            all_weights['cb_user_bias_%d' % len(self.cb_hidden_layers)] = tf.Variable(
                tf.random_normal([1, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='cb_user_bias_%d' % len(self.cb_hidden_layers))

            item_pre_size = self.item_feature_num * self.f_vector_size
            for i, layer_size in enumerate(self.cb_hidden_layers):
                all_weights['cb_item_layer_%d' % i] = tf.Variable(
                    tf.random_normal([item_pre_size, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                    name='cb_item_layer_%d' % i)
                all_weights['cb_item_bias_%d' % i] = tf.Variable(
                    tf.random_normal([1, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                    name='cb_item_bias_%d' % i)
                item_pre_size = self.cb_hidden_layers[i]
            all_weights['cb_item_layer_%d' % len(self.cb_hidden_layers)] = tf.Variable(
                tf.random_normal([item_pre_size, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='cb_item_layer_%d' % len(self.cb_hidden_layers))
            all_weights['cb_item_bias_%d' % len(self.cb_hidden_layers)] = tf.Variable(
                tf.random_normal([1, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
                name='cb_item_bias_%d' % len(self.cb_hidden_layers))

            all_weights['attention_weights'] = tf.Variable(
                tf.random_normal([self.ui_vector_size, self.attention_size], 0.0, 0.01, dtype=self.d_type),
                name='attention_weights')
            all_weights['attention_bias'] = tf.Variable(
                tf.random_normal([1, self.attention_size], 0.0, 0.01, dtype=self.d_type),
                name='attention_bias')
            all_weights['attention_pre'] = tf.Variable(
                tf.random_normal([self.attention_size], 0.0, 0.01, dtype=self.d_type),
                name='attention_pre')
        return all_weights

    def _init_loss(self):
        with self.graph.as_default():
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.train_labels, self.prediction))))
            self.rmse_cb = tf.sqrt(tf.reduce_mean(tf.square(
                tf.subtract(self.train_labels, self.bias + self.cb_prediction))))
            self.rmse_cf = tf.sqrt(tf.reduce_mean(tf.square(
                tf.subtract(self.train_labels, self.bias + self.cf_prediction))))
            self.loss = self.rmse + self.rmse_cb + self.rmse_cf
            self.l2_weights = list(tf.global_variables())
            self.var_list = list(tf.global_variables())
            self.l2 = 0
            for weights in self.l2_weights:
                self.l2 += tf.reduce_sum(tf.square(weights))

    def _init_lrp(self):
        with self.graph.as_default():
            # print(tf.transpose(self.weights['w']))
            ui_multiply = tf.multiply(self.u_vector, self.i_vector)
            prediction = self.prediction + \
                         1e-5 * tf.cast(tf.logical_or(tf.equal(self.prediction, 0), tf.is_nan(self.prediction)),
                                        self.d_type)
            r_bias, r_cc = self.bias / prediction, ui_multiply / tf.reshape(prediction, shape=[-1, 1])
            bias = self.bias + 1e-5 * tf.cast(tf.logical_or(tf.equal(self.bias, 0), tf.is_nan(self.bias)), self.d_type)
            r_u_bias = tf.reshape((self.u_bias / bias) * r_bias, shape=[-1, 1])
            r_i_bias = tf.reshape((self.i_bias / bias) * r_bias, shape=[-1, 1])
            r_g_bias = tf.reshape((self.weights['global_bias'] / bias) * r_bias, shape=[-1, 1])

            # self.lrp_list.extend([r_g_bias + r_u_bias + r_i_bias + tf.reduce_sum(r_cc, axis=1, keep_dims=True) + r_wide])

            r_cc_u, r_cc_i = r_cc / 2, r_cc / 2
            r_cf_u, r_cb_u = tf.reduce_sum(r_cc_u * self.a_cf_u, axis=1, keep_dims=True), r_cc_u * self.a_cb_u
            r_cf_i, r_cb_i = tf.reduce_sum(r_cc_i * self.a_cf_i, axis=1, keep_dims=True), r_cc_i * self.a_cb_i
            # self.lrp_list.extend([r_cf_u + r_cb_u + r_cf_i + r_cb_i - r_cc])

            # self.lrp_list.extend([tf.reduce_sum(r_cb_u, axis=1, keep_dims=True),
            #                       tf.reduce_sum(r_cb_i, axis=1, keep_dims=True)])
            for i in range(len(self.cb_hidden_layers) + 1):
                ix = len(self.cb_hidden_layers) - i

                z_u = tf.expand_dims(self.lrp_layers_u[ix], axis=1) * \
                      tf.transpose(self.weights['cb_user_layer_%d' % ix])
                mo_u = tf.reduce_sum(z_u, axis=2, keep_dims=True)
                mo_u = mo_u + 1e-5 * tf.cast(tf.logical_or(tf.equal(mo_u, 0), tf.is_nan(mo_u)), self.d_type)
                w_u = z_u / mo_u
                # w_u = tf.where(tf.is_nan(w_u), tf.zeros_like(w_u), w_u)
                r_cb_u = tf.expand_dims(r_cb_u, axis=1)
                r_cb_u = tf.matmul(r_cb_u, w_u)
                r_cb_u = tf.reduce_sum(r_cb_u, axis=1)

                z_i = tf.expand_dims(self.lrp_layers_i[ix], axis=1) * \
                      tf.transpose(self.weights['cb_item_layer_%d' % ix])
                mo_i = tf.reduce_sum(z_i, axis=2, keep_dims=True)
                mo_i = mo_i + 1e-5 * tf.cast(tf.logical_or(tf.equal(mo_i, 0), tf.is_nan(mo_i)), self.d_type)
                w_i = z_i / mo_i
                # w_i = tf.where(tf.is_nan(w_i), tf.zeros_like(w_i), w_i)
                r_cb_i = tf.expand_dims(r_cb_i, axis=1)
                r_cb_i = tf.matmul(r_cb_i, w_i)
                r_cb_i = tf.reduce_sum(r_cb_i, axis=1)
            # self.lrp_list.extend([tf.reduce_sum(r_cb_u, axis=1, keep_dims=True),
            #                       tf.reduce_sum(r_cb_i, axis=1, keep_dims=True)])

            r_ui = tf.concat([r_u_bias + r_cf_u, r_i_bias + r_cf_i], axis=1)

            r_cb_fs = []
            # self.lrp_list.extend([r_cb_u, r_fm_u, r_cb_i, r_fm_i])
            for i in range(self.user_feature_num):
                start, end = i * self.f_vector_size, (i + 1) * self.f_vector_size
                r_cb_fs.append(tf.reduce_sum(r_cb_u[:, start:end], axis=1, keep_dims=True))
            for i in range(self.item_feature_num):
                start, end = i * self.f_vector_size, (i + 1) * self.f_vector_size
                r_cb_fs.append(tf.reduce_sum(r_cb_i[:, start:end], axis=1, keep_dims=True))
            r_cb_fs = tf.concat(r_cb_fs, axis=1)

            # r_all = tf.concat([r_ui, r_cb_fs], axis=1) + r_g_bias / (self.feature_num + 2)
            r_g_bias = r_g_bias - \
                       1e-5 * tf.cast(tf.logical_or(tf.equal(r_g_bias, 1.0), tf.is_nan(r_g_bias)), self.d_type)
            r_all = tf.concat([r_ui, r_cb_fs], axis=1) / (1 - r_g_bias)
            # r_all = tf.nn.softmax(tf.abs(r_all), dim=1)

            # self.lrp_list.extend([r_g_bias +
            #                       tf.reduce_sum(r_ui, axis=1, keep_dims=True) +
            #                       tf.reduce_sum(r_cb_fs, axis=1, keep_dims=True)])
            # self.lrp_list.extend([r_all, tf.reduce_sum(r_all, axis=1, keep_dims=True)])
            self.lrp_list.append(r_all)
            # self.lrp_list.extend([r_cb_u, r_cb_i, r_cb_fs, r_g_bias])
