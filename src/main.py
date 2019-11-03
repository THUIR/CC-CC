# coding=utf-8

import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
import os
import copy

from utils import utils
from data_loaders.BaseDataLoader import BaseDataLoader
from data_processor.BaseDataProcessor import BaseDataProcessor
from data_processor.FSDataProcessor import FSDataProcessor
from runners.BaseRunner import BaseRunner
# from runners.CSRunner import CSRunner
from models.BaseModel import BaseModel
from models.DeepModel import DeepModel
from models.NFM import NFM
from models.FSNFM import FSNFM
from models.WideDeep import WideDeep
from models.FSWideDeep import FSWideDeep
from models.RecModel import RecModel
from models.BiasedMF import BiasedMF
from models.ACCM import ACCM
from models.FSACCM import FSACCM
from models.CCCC import CCCC
from models.FSCCCC import FSCCCC


def main():
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BaseModel',
                             help='Choose model to run.')
    # init_parser.add_argument('--runner_name', type=str, default='BaseRunner',
    #                          help='Choose runner to run.')
    init_args, init_extras = init_parser.parse_known_args()

    model_name = eval(init_args.model_name)
    init_args.runner_name = 'BaseRunner'
    runner_name = eval(init_args.runner_name)

    parser = argparse.ArgumentParser(description='')
    parser = utils.parse_global_args(parser)
    parser = BaseDataLoader.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)

    args, extras = parser.parse_known_args()
    args = model_name.add_model_args(args)

    log_file_name = [init_args.model_name, args.dataset, str(args.random_seed), init_args.runner_name,
                     'optimizer=' + args.optimizer, 'lr=' + str(args.lr), 'l2=' + str(args.l2),
                     'dropout=' + str(args.dropout), 'batch_size=' + str(args.batch_size)]
    log_file_name = '__'.join(log_file_name).replace(' ', '__')
    if args.log_file == '../log/log.txt':
        args.log_file = '../log/%s.txt' % log_file_name
    if args.result_file == '../result/result.npy':
        args.result_file = '../result/%s.npy' % log_file_name
    if args.model_path == '../model/%s/%s.ckpt' % (init_args.model_name, init_args.model_name):
        args.model_path = '../model/%s/%s.ckpt' % (init_args.model_name, log_file_name)

    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)
    logging.info(args)

    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data = BaseDataLoader(path=args.path, dataset=args.dataset, sep=args.sep, label=args.label,
                          append_id=args.append_id, include_id=args.include_id,
                          balance_train=args.balance_data > 0)

    if init_args.model_name in ['BaseModel']:
        model = model_name(class_num=data.class_num, feature_num=len(data.features), random_seed=args.random_seed,
                           model_path=args.model_path)
    elif init_args.model_name in ['DeepModel', 'NFM', 'WideDeep']:
        model = model_name(class_num=data.class_num, feature_num=len(data.features),
                           feature_dims=data.feature_dims, f_vector_size=args.f_vector_size, layers=eval(args.layers),
                           random_seed=args.random_seed, model_path=args.model_path)
    elif init_args.model_name in ['FSNFM', 'FSWideDeep']:
        model = model_name(class_num=data.class_num, feature_num=len(data.features),
                           feature_dims=data.feature_dims, f_vector_size=args.f_vector_size, layers=eval(args.layers),
                           random_seed=args.random_seed, model_path=args.model_path)
    elif init_args.model_name in ['RecModel', 'BiasedMF', 'CSRecModel']:
        model = model_name(class_num=data.class_num, feature_num=len(data.features),
                           user_num=data.user_num, item_num=data.item_num,
                           u_vector_size=args.u_vector_size, i_vector_size=args.i_vector_size,
                           random_seed=args.random_seed, model_path=args.model_path)
    elif init_args.model_name in ['ACCM', 'CCCC', 'FSACCM', 'FSCCCC']:
        model = model_name(class_num=data.class_num, feature_num=len(data.features),
                           user_num=data.user_num, item_num=data.item_num,
                           user_feature_num=len(data.user_features), item_feature_num=len(data.item_features),
                           feature_dims=data.feature_dims, f_vector_size=args.f_vector_size,
                           cb_hidden_layers=eval(args.cb_hidden_layers), attention_size=args.attention_size,
                           u_vector_size=args.u_vector_size, i_vector_size=args.i_vector_size, cs_ratio=args.cs_ratio,
                           random_seed=args.random_seed, model_path=args.model_path)
    else:
        logging.error('Unknown Model: ' + init_args.model_name)
        return

    if init_args.runner_name in ['BaseRunner']:
        runner = runner_name(
            optimizer=args.optimizer, learning_rate=args.lr,
            epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            dropout=args.dropout, l2=args.l2,
            metrics=args.metric, check_epoch=args.check_epoch)
    else:
        logging.error('Unknown Runner: ' + init_args.runner_name)
        return

    dp_parser = FSDataProcessor.parse_dp_args(argparse.ArgumentParser(description=''))
    dp_args, extras = dp_parser.parse_known_args()
    data_processor = FSDataProcessor(data, model, runner, fs_ratio=dp_args.fs_ratio, mode=dp_args.fs_mode)
    logging.info({**vars(args), **vars(dp_args)})

    logging.info('Test Before Training = ' + utils.format_metric(runner.evaluate(model, data.test_data))
                 + ' ' + ','.join(runner.metrics))
    if args.load > 0:
        runner.load_model(model)
    if args.train > 0:
        runner.train(model, data.train_data, data.validation_data, data.test_data, data_processor=data_processor)
    logging.info('Test After Training = ' + utils.format_metric(runner.evaluate(model, data.test_data))
                 + ' ' + ','.join(runner.metrics))

    np.save(args.result_file, runner.predict(model, data.test_data))
    logging.info('Save Test Results to ' + args.result_file)

    runner.lrp(model, data.train_data)
    runner.lrp(model, data.test_data)
    logging.debug(runner.evaluate(model, data.test_data))
    logging.debug(runner.evaluate(model, data.test_data))
    return


if __name__ == '__main__':
    main()
