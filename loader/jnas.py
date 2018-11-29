#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JNAS data set I/O

Module (Third-party library)
======
- pandas

TODO
----
- add enough comments
- fix JNAS().spinfo's mistake the uninvaild label

Update Contents
---------------
 (17.11.26) debug the keyword miss in JNAS()._condition_converter()
 (17.11.27) add get_sppaths method and sploads method
 (18.01.23) update to label search in JNAS().spinfo()
 (18.02.05) disuse JNAS_NOTICES_PATH
            label bug (`*P00` can not load) is fixed
 (18.03.22) add comments
 (18.08.23) get_sppaths -> sppaths

@author t-take
"""


import os
import re
import warnings

import pandas as pd

from .base import BaseDataSetReader, InvalidLabelError


JNAS_DATA_PATH = '/Volumes/share/ExperimentalSpeechDB/JNAS'


def spinfo_from_speakers_en(speakers_en_path):
    """
    Get the speakers information from speakers_en.txt

    Parameter
    ---------
    speakers_en_path : string
        speakers_en.txt path in JNAS

    Return
    ------
    spinfo : pandas.DataFrame
        speakers information
    """

    pattern = r'\s{1,9}'    # split pattern
                            # There are empty element of age,
                            # so split by 9 space.
    columns = []            # DataFrame columns

    with open(speakers_en_path, 'r') as f:
        for n, textline in enumerate(f):

            # split the data (The lastest element is always empty.)
            elements = re.split(pattern, textline)[:-1]

            # To skip the not need line
            # Speaker information data has 7 elements.
            if len(elements) != 7:
                continue

            if not columns:     # , then contents show columns.
                # Since '-' and '/' can not be used for column names,
                # replace with '_'.
                columns.extend([re.sub(r'(-|/)', '_', s) for s in elements])
                spinfo = pd.DataFrame(columns=columns)
                continue

            spinfo = spinfo.append(
                pd.Series(data=elements, index=columns),
                ignore_index=True)

            if n > 312:    # There is no datas after 312 line.
                break

    return spinfo


class JNAS(BaseDataSetReader):
    """
    ASJ Continuous Speech Corpus
    -- Japanese Newspaper Article Sentences (JNAS) --

    Parameter
    ---------
    home_directory : str, default='/Volumes/share/実験用音声データベース/JNAS'
        JNAS data directory path

    Attributes
    ----------
    samplingrate : int
        sampling rate (Hz)

    path_template : str
        .wav file path template

    speakers_information : pandas.DataFrame
        speakers information that is written in speakers_en.txt
    """
    def __init__(self, home_directory=JNAS_DATA_PATH):
        super(JNAS, self).__init__(home_directory=home_directory)
        self.samplingrate = 16000

        self.path_template = os.path.join(
            '*', 'WAVES_{mic}', '{sex}{text_id}', '{text_set_norm}',
            '{text_set}{sex}{text_id}{subset}{sen_id}_{mic}.wav')

        # read the speakers information data
        spinfo_path = os.path.join(
            home_directory, 'Vol1', 'DOCS', 'speakers_en.txt')
        try:
            self.speakers_information = spinfo_from_speakers_en(spinfo_path)
        except OSError:
            warnings.warn('The speakers information file `%s` does not exist.'
                          % os.path.join(spinfo_path), UserWarning)

    def _label2condition(self, label):

        text_set, inner_label = label[0], label[1:]
        if text_set == 'N':
            conditions = {'text_set': 'N'}
            patterns = [('sex', r'(M|F|\*?)'),
                        ('text_id', r'([0-9]{3}|P[0-9]{2}|\*?)'),
                        ('sen_id', r'([0-9]{3}|\*?)'),
                        ('underbar', r'_'),
                        ('mic', r'(DT|HS|\*?)')]

        elif text_set == 'B':
            conditions = {'text_set': 'B'}
            patterns = [('sex', r'(M|F|\*?)'),
                        ('text_id', r'([0-9]{3}|P[0-9]{2}|\*?)'),
                        ('subset', r'([A-J]|\*?)'),
                        ('sen_id', r'([0-9]{2}|\*?)'),
                        ('underbar', r'_'),
                        ('mic', r'(DT|HS|\*?)')]

        else:
            raise InvalidLabelError('Invalid label {}'.format(label))

        before_wildcard = False

        for condition_key, pattern in patterns:
            if condition_key == 'underbar':
                inner_label = inner_label[1:]
                continue

            matched = re.match(r'^{}'.format(pattern), inner_label).group()

            if matched == '' and before_wildcard:
                raise InvalidLabelError('Invalid label {}'.format(label))
            elif matched == '':
                conditions[condition_key] = '*'
                before_wildcard = True
            else:
                conditions[condition_key] = matched
                before_wildcard = False

            inner_label = inner_label[len(matched):]

        return conditions

    def _condition_converter(self, **conditions):

        converted_conditions = {'text_set': '*', 'sex': '*', 'text_id': '*',
                                'subset': '*', 'sen_id': '*', 'mic': '*',
                                'text_set_norm': '*'}
        converted_conditions.update(conditions)

        if converted_conditions['text_set'] in 'NB':
            conditions['text_set_norm'] = \
                'NP' if conditions['text_set'] == 'N' else 'PB'
        if 'text_id' in conditions.keys() and not conditions['text_id'] == '*':
            converted_conditions['text_id'] = str(
                conditions['text_id']).zfill(3)
        if 'sen_id' in conditions.keys() and not conditions['sen_id'] == '*':
            converted_conditions['sen_id'] = str(conditions['sen_id']).zfill(2)

        return converted_conditions

    def warning_checker(self, label):
        """Warning checker of the given label"""
        pass

    def spinfo(self, label=None, **conditions):
        """
        Get the speakers information.

        Parameters
        ----------
        label : str or None, default to None
            The data label to show data file

        sp_code : str, (Mxxx or Fxxx)
            The speaker ID is speaker's label shown <SEX> + <TEXT-ID>.
            Where <SEX> is one-character code of speaker sex
            (M: male, F: female). And <TEXT-ID> is three-character id of
            newspaper text-set
            (001-150: SC text-set, P01-P05: paragraph text-set).

        m_f : str, (M or F)
            one-charecter code of speaker sex (M: male, F: female)

        age : str, (xx-xx)
            two-character id  of speaker age
            (10-19, 20-29, 30-39, 40-49, 50-59, 60_and_up)

        news_text : str (xxx or Pxx)
            three-character id of sentence
            (001-150: SC text-set, P01-P05: paragraph text-set)

        PB_text : str, (X)
            subset name of ATR 503 PB-sentences (A, B, C, ..., J)

        rec_site : str, (xxx)
            recorded site of sentences
            (please check the rec_site_en.txt or rec_site_jp.txt)

        rec_date : str, (x/xx/xx)
            recorded date of sentences
            (please check the rec_site_en.txt or rec_site_jp.txt)

        Return
        ------
        information : pandas.DataFrame
            speakers information
        """
        information = self.speakers_information

        if label is not None:
            _cond = self._label2condition(label)
            conditions = {
                'sp_code': _cond['sex'] + _cond['text_id'],
                'm_f': _cond['sex'],
                'news_text': str(int(_cond['text_id'])),
                'PB_text': _cond['subset'],
            }

        for key, item in conditions.items():
            # chack the invalid keyword
            if key not in ['sp_code', 'm_f', 'age', 'news_text', 'PB_text',
                           'rec_site', 'rec_date']:
                raise KeyError(
                    "The conditions keyword must be ['sp_code', 'm_f', 'age',"
                    "'news_text', 'PB_text', 'rec_site', 'rec_date']")

            information = information[information[key] == item]

        return information

    def sppaths(self, code, sen_id='*', text_set='PB', mic='HS'):
        """
        Get the paths of a speaker

        Parameters
        ----------
        code : str, (Mxxx or Fxxx)
            speaker code

        text_set : str, ('NP' or 'PB')
            If NP, get news text speech path
            If PB, get phoneme balance speech path

        mic : str, ('DT' or 'HS')
            If DT, get path recorded by desktop mic
            If PB, get path recorded by head set mic

        Return
        ------
        path_generator : generator
            path_generator generate the paths of a speaker
        """

        try:
            sp_data = self.spinfo(sp_code=code).values[0]
        except IndexError:
            raise InvalidLabelError('speaker code is not exist: {%s}' % code)

        if text_set == 'NP':
            text_set = 'N'
        if text_set == 'PB':
            text_set = 'B'

        path_generator = self.get_paths(
            text_set=text_set,
            sex=sp_data[1],
            text_id=sp_data[3],
            subset=sp_data[4],
            sen_id=sen_id,
            mic=mic,
        )

        return path_generator

    def sploads(self, code, sen_id='*', text_set='PB', mic='HS'):
        """
        Load the waveform data of a speaker

        Parameters
        ----------
        code : str, (Mxxx or Fxxx)
            speaker code

        text_set : str, ('NP' or 'PB')
            If NP, get news text speech path
            If PB, get phoneme balance speech path

        mic : str, ('DT' or 'HS')
            If DT, get path recorded by desktop mic
            If PB, get path recorded by head set mic

        Return
        ------
        wave_generator : generator
            wave_generator generate the waveform (ndarray) of a speaker
        """

        try:
            sp_data = self.spinfo(sp_code=code).values[0]
        except IndexError:
            raise InvalidLabelError('speaker code is not exist: {%s}' % code)

        if text_set == 'NP':
            text_set = 'N'
        if text_set == 'PB':
            text_set = 'B'

        wave_generator = self.load(
            text_set=text_set,
            sex=sp_data[1],
            text_id=sp_data[3],
            subset=sp_data[4],
            sen_id=sen_id,
            mic=mic,
        )

        return wave_generator
