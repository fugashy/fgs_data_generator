# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import numpy as np


class GroundtruthModurator(ABC):
    @abstractmethod
    def modurate(self, data):
        raise NotImplementedError('To developer, inherit this class')


class NormalDistribution(GroundtruthModurator):
    @staticmethod
    def create(config_dict):
        std_dev = conf_dict['std_dev']
        return NormalDistribution(std_dev)

    def __init__(self, std_dev):
        self._modurate = lambda data: np.random.normal(
                np.array(data), np.array(std_dev))

    def modurate(self, data):
        return self._modurate(data)


class Transformer(GroundtruthModurator):
    @staticmethod
    def create(config_dict):
        rot_rad = conf_dict['rot_rad']
        trans = conf_dict['trans']
        return Transformer(rot_rad, trans)

    def __init__(self, rot_rad, trans):
        rot_mat = np.array(
            [
                [np.cos(rot_rad), -np.sin(rot_rad)],
                [np.sin(rot_rad), np.cos(rot_rad)]
            ])
        translate = lambda xy: np.array([xy[0] + trans[0], xy[1] + trans[1]])
        self._transform = lambda data: np.array([translate(rot_mat @ xy) for xy in data])

    def modurate(self, data):
        return self._transform(data)
