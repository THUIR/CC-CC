# coding=utf-8

import tensorflow as tf
import logging
from time import time
from utils import utils
from tqdm import tqdm
import gc
import numpy as np
from data_processor.BaseDataProcessor import BaseDataProcessor
import copy


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 16,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-4,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='Adagrad',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='RMSE', check_epoch=10):
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.keep_prob = 1 - dropout
        self.no_dropout = 1.0
        self.l2_weight = l2
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.time = None

        self.train_results, self.valid_results, self.test_results = [], [], []

    def _build_sess(self, model):
        with model.graph.as_default():
            optimizer_name = self.optimizer_name.lower()
            if optimizer_name == 'gd':
                logging.info("Optimizer: GD")
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)
            elif optimizer_name == 'adagrad':
                logging.info("Optimizer: Adagrad")
                optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate, initial_accumulator_value=1e-8)
            elif optimizer_name == 'adam':
                logging.info("Optimizer: Adam")
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            else:
                logging.error("Unknown Optimizer: " + self.optimizer_name)
                assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                model.optimizer = optimizer.minimize(model.loss + model.l2 * self.l2_weight, var_list=model.var_list)
                # model.optimizer = optimizer.minimize(model.loss + model.l2 * self.l2_weight)

            model.init = [tf.global_variables_initializer(), tf.local_variables_initializer(),
                          tf.tables_initializer()]
            # model.init = [tf.global_variables_initializer()]
            # model.init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.allow_soft_placement = True
            # config.log_device_placement = True
            model.sess = tf.Session(config=config)
            # for init in model.init:
            model.sess.run(model.init)

            # model.sess.run(tf.variables_initializer(optimizer.variables()))
            # model.sess.run(tf.variables_initializer(
            #     [var for var in tf.global_variables() if self.optimizer_name in var.name]))

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def get_feed_dict(self, model, data, batch_i, batch_size, train):
        batch_start = batch_i * batch_size
        batch_end = min(len(data['Y']), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        feed_dict = {model.train_features: data['X'][batch_start: batch_start + real_batch_size],
                     model.train_labels: data['Y'][batch_start:batch_start + real_batch_size],
                     model.dropout_keep: self.keep_prob if train else self.no_dropout, model.train_phase: train}
        if hasattr(model, 'drop_pos'):
            if train or 'drop_pos' in data:
                feed_dict[model.drop_pos] = data['drop_pos'][batch_start: batch_start + real_batch_size]
            else:
                feed_dict[model.drop_pos] = np.zeros(shape=feed_dict[model.train_features].shape, dtype=np.float32)
        return feed_dict

    def predict(self, model, data):
        if model.sess is None:
            self._build_sess(model)
        data = utils.input_data_is_list(data)
        num_example = len(data['Y'])
        total_batch = int((num_example + self.eval_batch_size - 1) / self.eval_batch_size)
        assert num_example > 0

        gc.collect()
        predictions = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1):
            feed_dict = self.get_feed_dict(model, data, batch, self.eval_batch_size, False)
            prediction = model.sess.run(model.prediction, feed_dict=feed_dict)
            predictions.append(prediction)
        predictions = np.concatenate(predictions)
        gc.collect()
        return predictions

    def fit(self, model, data, epoch=-1):  # fit the results for an input set
        if model.sess is None:
            self._build_sess(model)
        num_example = len(data['Y'])
        total_batch = int((num_example + self.batch_size - 1) / self.batch_size)
        gc.collect()
        for batch in tqdm(range(total_batch), leave=False, desc='Epoch %5d' % epoch, ncols=100, mininterval=1):
            feed_dict = self.get_feed_dict(model, data, batch, self.batch_size, True)
            opt = model.sess.run(model.optimizer, feed_dict=feed_dict)
        gc.collect()

    def train(self, model, train_data, validation_data=None, test_data=None, data_processor=None):
        assert train_data is not None
        if model.sess is None:
            self._build_sess(model)
        if data_processor is None:
            data_processor = BaseDataProcessor()
        self._check_time(start=True)

        init_train = self.evaluate(model, train_data) \
            if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = self.evaluate(model, validation_data) \
            if validation_data is not None else [-1.0] * len(self.metrics)
        init_test = self.evaluate(model, test_data) \
            if test_data is not None else [-1.0] * len(self.metrics)
        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
            utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
            self._check_time()) + ','.join(self.metrics))

        try:
            for epoch in range(self.epoch):
                gc.collect()
                self._check_time()
                epoch_train_data = copy.deepcopy(train_data)
                epoch_train_data = data_processor.epoch_process_train(epoch_train_data, epoch=epoch + 1)
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    self.check(model, epoch_train_data)
                self.fit(model, epoch_train_data, epoch=epoch + 1)
                del epoch_train_data
                training_time = self._check_time()

                # output validation
                train_result = self.evaluate(model, train_data) \
                    if train_data is not None else [-1.0] * len(self.metrics)
                valid_result = self.evaluate(model, validation_data) \
                    if validation_data is not None else [-1.0] * len(self.metrics)
                test_result = self.evaluate(model, test_data) \
                    if test_data is not None else [-1.0] * len(self.metrics)
                testing_time = self._check_time()

                self.train_results.append(train_result)
                self.valid_results.append(valid_result)
                self.test_results.append(test_result)

                logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                             % (epoch + 1, training_time, utils.format_metric(train_result),
                                utils.format_metric(valid_result), utils.format_metric(test_result),
                                testing_time) + ','.join(self.metrics))

                if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                    self.save_model(model)
                if utils.eva_termination(self.metrics[0], self.valid_results):
                    logging.info("Early stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                self.save_model(model)

        # Find the best validation result across iterations
        best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        self.load_model(model)

    def evaluate(self, model, data):  # evaluate the results for an input set
        if model.sess is None:
            self._build_sess(model)
        data = utils.input_data_is_list(data)
        predictions = self.predict(model, data)
        labels = data['Y']
        return model.evaluate_method(predictions, labels, self.metrics)

    def check(self, model, data):
        if model.sess is None:
            self._build_sess(model)
        data = utils.input_data_is_list(data)
        feed_dict = self.get_feed_dict(model, data, 0, self.eval_batch_size, False)

        # print(model.check_list)
        if len(model.check_list) > 0:
            tmp = model.sess.run([c[1] for c in model.check_list], feed_dict=feed_dict)
            for i, t in enumerate(tmp):
                logging.info("     " + ' '.join([model.check_list[i][0], str(np.array(t)), str(np.array(t).shape)]))

        loss, l2 = model.sess.run([model.loss, model.l2], feed_dict=feed_dict)
        l2 = l2 * self.l2_weight
        if not (loss * 0.005 < l2 < loss * 0.1):
            logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))

    def lrp(self, model, data):
        if model.sess is None:
            self._build_sess(model)
        batch_size = self.eval_batch_size
        data = utils.input_data_is_list(data)
        num_example = len(data['Y'])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        if len(model.lrp_list) > 0:
            gc.collect()
            lrps = [[] for i in model.lrp_list]
            for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1):
                feed_dict = self.get_feed_dict(model, data, batch, batch_size, False)
                lrp_tmps = model.sess.run(model.lrp_list, feed_dict=feed_dict)
                for i, lrp in enumerate(lrp_tmps):
                    lrps[i].append(lrp)
            # for lrp in lrps[0]:
            #     logging.debug(lrp.shape)
            lrps = [np.concatenate(lrp, axis=0) for lrp in lrps]
            for i, lrp in enumerate(lrps):
                logging.debug('lrp %d: %s %s' % (i, lrps[i].shape, lrps[i]))
            gc.collect()
            return lrps
        return []

    def load_model(self, model, model_path=None):
        if model.sess is None:
            self._build_sess(model)
        if model_path is None:
            model_path = model.model_path
        with model.graph.as_default():
            model.saver.restore(model.sess, model_path)
            logging.info('Load model from ' + model_path)

    def save_model(self, model, model_path=None):
        if model.sess is None:
            self._build_sess(model)
        if model_path is None:
            model_path = model.model_path
        with model.graph.as_default():
            model.saver.save(model.sess, model_path)
            logging.info('Save model to ' + model_path)
