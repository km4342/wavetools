#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base data set loader

TODO
----
- zip データへの対応
- 条件が完全でない場合必ずジェネレータが返却されるバグへの対応

Updates
-------
 (17.11.04) initial making
 (17.11.26) load() 及び _file_paths() における 入力 conditions が全条件を持たない場合
            ワイルドカード判定ができないことによる，ジェネレータの返却ができなくなるバグを
            修正
 (18.11.29) `spalib`に依らない実装

Last update: Thu Nov 29 2018
@author t-take
"""
from __future__ import division, absolute_import, print_function

import os
import re
import types
import warnings
import wave
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np


__all__ = ['InvalidLabelError', 'BaseDataSetReader']


def _path2label(path):
    return os.path.splitext(os.path.basename(path))[0]


def _have_wildcard(label, conditions):
    if label is None:
        return '*' in conditions.values()
    elif conditions is None:
        return '*' in label
    else:
        return '*' in label or '*' in conditions.keys()


def readwav(filepath):
    """wave read"""

    with wave.open(str(filepath), 'r') as wfp:
        framerate = wfp.getframerate()
        sampwidth = wfp.getsampwidth()
        rawdata = wfp.readframes(wfp.getnframes())

    numdata = np.frombuffer(rawdata, dtype=get_npformat_from(sampwidth))
    numdata = numdata / (2 ** (sampwidth * 8 - 1))

    return numdata, framerate


def get_npformat_from(width, unsigned=False):
    """
    Returns a Numpy format constant for the specified `width`.

    Parameters
    ----------
    width : int
        The desired sample width in bytes (1, 2, or 4)
    unsigned : bool, default=False
        For 1 byte width, specifies signed or unsigned format.

    Return
    ------
    Numpy's data type
    """

    if width == 1:
        if unsigned:
            return np.uint8
        else:
            return np.int8
    elif width == 2:
        return np.int16
    elif width == 4:
        return np.int32
    else:
        raise ValueError("Invalid width: %d" % width)


class InvalidLabelError(ValueError):
    pass


class BaseDataSetReader(metaclass=ABCMeta):
    """Base class for DataSetReader modules."""

    def __init__(self, home_directory):
        super(BaseDataSetReader, self).__init__()
        self.home_directory = Path(home_directory)
        self.samplingrate = -1

        if not self.home_directory.is_dir():
            raise FileNotFoundError(
                'Data set directory is not exist. `%s`' % self.home_directory)

        self.path_template = '{}.wav'

    @abstractmethod
    def _label2condition(self, label):
        """Return label matched conditions."""
        pass

    @abstractmethod
    def warning_checker(self, label):
        """waring when label matched"""
        pass

    def _condition_converter(self, **conditions):
        """Convert condition for matching label."""
        return conditions

    def _file_paths(self, **conditions):
        """Get the paths matched conditions."""

        # Make path under the subdirectory
        condition_pattern = self.path_template.format(**conditions)

        # replace for `*` if `*` followed by two or more
        condition_pattern = re.sub(r'\*{2,}', '*', condition_pattern)

        for path in self.home_directory.glob(condition_pattern):
            if not path.is_file():
                continue
            yield str(path)

    def get_paths(self, label=None, **conditions):
        """
        Get the paths matched label or conditions.

        Parameters
        ----------
        label : str or None, default to None
            The data label to show data file
        [conditions]
            each class's conditions

        Return
        ------
        path_generator : pathlib.PosixPath or generator
            Path of mutching conditions
            If wildcard in label or conditions, return the generator.
            Otherwise return the path.
        """

        if label is not None:
            conditions = self._label2condition(label)
        conditions = self._condition_converter(**conditions)
        path_generator = self._file_paths(**conditions)

        if not _have_wildcard(label, conditions):   # only path
            try:
                path_generator = next(path_generator)
            except StopIteration:   # required file does not exist
                raise FileNotFoundError('The file corresponding to the label'
                                        '`{}` does not exist.'.format(label))

        return path_generator

    def _load(self, path_generator):

        for path in path_generator:
            self.warning_checker(_path2label(path))
            wav, fs = readwav(path)

            if fs != self.samplingrate:
                msg = ('The samplingrate `{fs}` of loaded file "{filename}" '
                       'is not requirement value {rate}.').format(
                           fs=fs, filename=_path2label(path),
                           rate=self.samplingrate)
                warnings.warn(msg, UserWarning)

            yield wav

    def load(self, label=None, **conditions):
        """
        Load the .wav files from label.

        Parameters
        ----------
        label : str or None, default to None
            The data label to show data file
        [conditions]
            each class's conditions

        Return
        ------
        wav_generator : numpy.ndarray or generator
            The wave data of matching label or conditions
            If there is wildcard in label or conditions, return the generator.
            Otherwise return the ndarray.
        """

        path_generator = self.get_paths(label, **conditions)

        if isinstance(path_generator, types.GeneratorType):
            return self._load(path_generator)
        else:
            try:
                wav = next(self._load([path_generator]))
            except StopIteration:   # required file does not exist
                raise InvalidLabelError('Invalid label {}'.format(label))
            else:
                return wav
