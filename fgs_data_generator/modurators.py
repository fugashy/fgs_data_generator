# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import numpy as np

def create(conf_dict):
    if conf_dict['type'] == 'normal_distribution':
        return NormalDistribution.create(conf_dict)
    elif conf_dict['type'] == 'rigid_motion':
        return RigidMotion.create(conf_dict)
    else:
        raise NotImplementedError(
                '{} is not implemented.'.format(conf_dict['type']))


class ModelModurator(ABC):
    @abstractmethod
    def modurate(self, data):
        raise NotImplementedError('To developer, inherit this class')


class NormalDistribution(ModelModurator):
    @staticmethod
    def create(config_dict):
        std_dev = config_dict['std_dev']
        return NormalDistribution(std_dev)

    def __init__(self, std_dev):
        self._modurate = lambda data: np.random.normal(
                np.array(data), np.array(std_dev))

    def modurate(self, data):
        return self._modurate(data)


class RigidMotion(ModelModurator):
    @staticmethod
    def create(config_dict):
        rot_rad = config_dict['rot_rad']
        trans = config_dict['trans']
        return RigidMotion(rot_rad, trans)

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
