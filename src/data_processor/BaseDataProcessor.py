# coding=utf-8
import copy
from utils import utils


class BaseDataProcessor(object):
    @staticmethod
    def parse_dp_args(parser):
        return parser

    def epoch_process_train(self, data, epoch):
        if data is None:
            return data
        utils.shuffle_in_unison_scary(data)
        return data
