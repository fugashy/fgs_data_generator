# -*- coding: utf-8 -*-
import cv2
import numpy as np


def create(modurator, conf_dict):
    if conf_dict['type'] == 'curve2d':
        return Curve2d.create(modurator, conf_dict)
    elif conf_dict['type'] == 'ellipse':
        return Ellipse.create(modurator, conf_dict)
    elif conf_dict['type'] == 'const':
        return Const.create(modurator, conf_dict)
    elif conf_dict['type'] == 'line':
        return Line.create(modurator, conf_dict)
    elif conf_dict['type'] == 'michaelis_menten':
        return MichaelisMenten.create(modurator, conf_dict)
    elif conf_dict['type'] == 'cos':
        return Cos.create(modurator, conf_dict)
    else:
        raise NotImplementedError(
                '{} is not implemented.'.format(conf_dict['type']))


class Data2d():
    def __init__(self, modurator):
        self._modurator = modurator

        self._gt = None
        self._obs = None

    @property
    def gt(self):
        return self._gt

    @property
    def obs(self):
        return self._obs

    def save(self, filepath):
        file_handle = cv2.FileStorage(filepath, cv2.FileStorage_WRITE)
        file_handle.write('ground_truth', self._gt)
        file_handle.write('observation', self._obs)

    def _generate(self, generate_data):
        self._gt = generate_data()
        self._obs = self._modurator.modurate(self._gt)

class Curve2d(Data2d):
    u"""
    """
    @staticmethod
    def create(modurator, conf_dict):
        start = conf_dict['start']
        end = conf_dict['end']
        step = conf_dict['step']
        a = conf_dict['a']
        b = conf_dict['b']
        c = conf_dict['c']
        # 3次関数or2次関数
        if 'd' in conf_dict and conf_dict['d']:
            d = conf_dict['d']
            p = [a, b, c, d]
            return Curve2d(modurator, start, end, step, p)
        else:
            p = [a, b, c]
            return Curve2d(modurator, start, end, step, p)

    def __init__(self, modurator, s, e, st, p):
        super(Curve2d, self).__init__(modurator)
        if len(p) == 3:
            f = lambda x: p[0] * x**2 + p[1] * x + p[2]
        elif len(p) == 4:
            f = lambda x: p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3]
        else:
            raise Exception('Invalid parameter length.')

        generate_data = lambda : np.array(
            [[x, f(x)] for x in np.arange(s, e, st)])
        self._generate(generate_data)


class Ellipse(Data2d):
    @staticmethod
    def create(modurator, conf_dict):
        a = conf_dict['a']
        b = conf_dict['b']
        rotation = conf_dict['rotation']
        translation = conf_dict['translation']
        theta_start = conf_dict['theta_start']
        theta_end = conf_dict['theta_end']
        theta_step = conf_dict['theta_step']
        return Ellipse(
                modurator, a, b, rotation, translation,
                theta_start, theta_end, theta_step)

    def __init__(self, modurator, a, b, r, trs, s, e, st):
        super(Ellipse, self).__init__(modurator)
        theta_range = np.arange(s, e, st)

        xy = lambda t: np.array(
            [a * np.cos(t) + trs[0], b * np.sin(t) + trs[1]])
        rot_mat = np.array(
            [
                [np.cos(r), -np.sin(r)],
                [np.sin(r), np.cos(r)]
            ])
        generate_data = lambda: np.array(
            [rot_mat @  xy(t) for t in theta_range])
        self._generate(generate_data)


class Const(Data2d):
    @staticmethod
    def create(modurator, conf_dict):
        translation = conf_dict['translation']
        num = conf_dict['num']
        return Const(modurator, translation, num)

    def __init__(self, modurator, trs, num):
        super(Const, self).__init__(modurator)
        generate_data = lambda : np.array([trs for i in range(num)])
        self._generate(generate_data)


class Line(Data2d):
    @staticmethod
    def create(modurator, conf_dict):
        a = conf_dict['a']
        b = conf_dict['b']
        start = conf_dict['start']
        end = conf_dict['end']
        step = conf_dict['step']
        return Line(modurator, a, b, start, end, step)

    def __init__(self, modurator, a, b, start, end, step):
        super(Line, self).__init__(modurator)
        x_range = np.arange(start, end, step)
        f = lambda x: a * x + b
        generate_data = lambda: np.array([[x, f(x)] for x in x_range])
        self._generate(generate_data)


class MichaelisMenten(Data2d):
    @staticmethod
    def create(modurator, conf_dict):
        b1 = conf_dict['b1']
        b2 = conf_dict['b2']
        start = conf_dict['start']
        end = conf_dict['end']
        step = conf_dict['step']
        return MichaelisMenten(modurator, b1, b2, start, end, step)

    def __init__(self, modurator, b1, b2, s, e, st):
        super(MichaelisMenten, self).__init__(modurator)
        x_range = np.arange(s, e, st)
        f = lambda x: b1 * x / (b2 + x)
        generate_data = lambda: np.array([[x, f(x)] for x in x_range])
        self._generate(generate_data)


class Cos(Data2d):
    @staticmethod
    def create(modurator, conf_dict):
        a = conf_dict['a']
        b = conf_dict['b']
        start = conf_dict['start']
        end = conf_dict['end']
        step = conf_dict['step']
        return Cos(modurator, a, b, start, end, step)

    def __init__(self, modurator, a, b, start, end, step):
        super(Cos, self).__init__(modurator)
        x_range = np.arange(start, end, step)
        f = lambda x: a * np.cos(b * x)
        generate_data = lambda: np.array([[x, f(x)] for x in x_range])
        self._generate(generate_data)
