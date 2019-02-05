#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITMIT data set I/O

Module (Third-party library)
======
-

TODO
----
- wav以外のファイルタイプの読み出し

Update Contents
---------------
 (19.02.01) Initialize

Last update: Fri Feb 1 2019
@author t-take
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

from .base import BaseDataSetReader, InvalidLabelError, _path2label


TIMIT_DATA_PATH = '/Volumes/share/ExperimentalSpeechDB/TIMIT'


class TIMIT(BaseDataSetReader):
    """
    TIMIT Acoustic-Phonetic Continuous Speech Corpus

    Parameter
    ---------
    home_directory : str, default='/Volumes/share/ExperimentalSpeechDB/TIMIT'
        TIMIT data directory path

    Attributes
    ----------
    samplingrate : int
        sampling rate (Hz)
    path_template : str
        .wav file path template
    """
    def __init__(self, home_directory=TIMIT_DATA_PATH):
        super(TIMIT, self).__init__(home_directory=home_directory)
        self.samplingrate = 16000

        self.path_template = os.path.join(
            '{useage}', '{dialect}', '{sex}{speaker_id}',
            '{sentence_id}.{file_type}'
        )

        # read the speakers information data
        spinfo_path = self.home_directory / 'DOC' / 'SPKRINFO.TXT'
        try:
            with open(spinfo_path, 'r') as fp:
                txt = fp.read().split('\n')
                names = re.split(r'\s+', txt[37][1:])
                fp.seek(0)  # pd.read_csvに渡すために先頭にシーク
                self.speakers_information = pd.read_csv(
                    fp, sep=r'\s+', names=names,
                    skiprows=lambda l: txt[l].startswith(';'),)

        except OSError:
            warnings.warn('The speakers information file `%s` does not exist.'
                          % spinfo_path)

    def _label2condition(self, label):

        sex_spid, senid = label.split('/')
        conditions = {
            'useage': '*',
            'dialect': '*',
            'sex': sex_spid[0],
            'speaker_id': sex_spid[1:],
            'sentence_id': senid,
            'file_type': 'WAV'
        }

        return conditions

    def _condition_converter(self, **conditions):
        keylist = ['useage', 'dialect', 'sex', 'speaker_id', 'sentence_id',
                   'file_type',]
        conditions_ = {key: '*' for key in keylist}
        conditions_.update(conditions)
        return conditions_

    def warning_checker(self, label):
        """Warning checker of the given label"""

    def _load(self, path_generator):

        for path in path_generator:
            self.warning_checker(_path2label(path))
            wav, info = sphread(path)

            if info['sample_rate'] != self.samplingrate:
                msg = ('The samplingrate `{fs}` of loaded file "{filename}" '
                       'is not requirement value {rate}.').format(
                           fs=info['sample_rate'], filename=_path2label(path),
                           rate=self.samplingrate)
                warnings.warn(msg, UserWarning)

            yield wav

    def sppaths(self, code):
        """
        Get the paths of a speaker

        Parameters
        ----------
        code : str
            speaker code

        Return
        ------
        generator
            return variable generate the paths of a speaker
        """

        try:
            spinfo = self.speakers_information.loc[
                self.speakers_information['ID'] == code].values[0]
        except IndexError:
            raise InvalidLabelError('speaker code is not exist: {%s}' % code)

        return self.get_paths(
            usage='TEST' if spinfo[3] == 'TST' else 'TRAIN',
            dialect='DR{}'.format(spinfo[2]),
            sex=spinfo[1],
            speaker_id=spinfo[0],
            sentence_id='*',
            file_type='WAV',
        )

    def sploads(self, code):
        """
        Load the waveform data of a speaker

        Parameters
        ----------
        code : str
            speaker code

        Return
        ------
        generator
            return value generate the wavefom (numpy.ndarray) of a speaker
        """

        try:
            spinfo = self.speakers_information.loc[
                self.speakers_information['ID'] == code].values[0]
        except IndexError:
            raise InvalidLabelError('speaker code is not exist: {%s}' % code)

        return self.load(
            usage='TEST' if spinfo[3] == 'TST' else 'TRAIN',
            dialect='DR{}'.format(spinfo[2]),
            sex=spinfo[1],
            speaker_id=spinfo[0],
            sentence_id='*',
            file_type='WAV',
        )


def sphread(filepath):
    """
    sphデータ読み取り

    Parameter
    ---------
    filepath : str or pathlib.Path
        data file path (.sph)
    Return
    ------
    numpy.ndarray, shape=(`nframes`, `nchannels`)
        wave data
    """

    with open(filepath, 'rb') as fp:
        # ヘッダ読み取り
        if fp.read(8) != b'NIST_1A\n':
            raise OSError('file does not start with `NIST_1A` id')
        headsize = int(fp.read(8).strip())

        end = b'end_head'
        headinfo = {}
        # すでに読み取った16bitを除外して`headsize`分を読み込み
        for line in fp.read(headsize - 16).splitlines():
            if line.startswith(end):
                break

            line = line.decode('latin-1')
            key, size, contents = line.split(' ')
            if size[:2] == '-i':
                contents = int(contents)
            headinfo.update({key: contents})

        # データ読み込み
        datasize = headinfo['sample_count'] * headinfo['sample_n_bytes']
        if headinfo['sample_n_bytes'] == 1:
            npformat = np.uint8
        elif headinfo['sample_n_bytes'] == 2:
            npformat = np.int16
        elif headinfo['sample_n_bytes'] == 4:
            npformat = np.int32
        else:
            raise RuntimeError('Unrecognized bytes count: {}'
                               .format(headinfo['sample_n_bytes']))

        data = np.frombuffer(fp.read(datasize), dtype=npformat)
        data = (data.reshape((-1, headinfo['channel_count']))
                / 2 ** (headinfo['sample_sig_bits'] - 1))

        return data, headinfo
