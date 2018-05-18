# -*- coding: utf-8 -*-

import os
import time
import sys
import numpy as np
import pyaudio
import wave
import struct
from scipy import io
from tqdm import tqdm


CHUNK = 1024
FORMAT = pyaudio.paInt16


def device_info():
    """get the device information"""

    audio = pyaudio.PyAudio()

    count = audio.get_device_count()

    devices = []

    for i in range(count):
        devices.append(audio.get_device_info_by_index(i))

    for i, dev in enumerate(devices):
        print(i, dev['name'])


def import_wav(name):
    """import wav files"""

    wave_file = wave.open(name, 'r')

    channel = wave_file.getnchannels()  # Channel
    fs = wave_file.getframerate()       # Sampling frequency
    nframes = wave_file.getnframes()    # Number of frames

    data = wave_file.readframes(wave_file.getnframes())
    data = np.frombuffer(data, dtype="int16") / 32767.0

    wave_file.close()

    return data, fs


def save(data, name, ch=1, nbit=16, fs=48000, type='wav'):
    """save files

    Parameters:
    -----------
    data: array-like
        input signal

    name: str
        output file name

    ch: int (default: 1ch)
        number of channels

    nbit: int (default: 16bit)
        number of bit

    fs: int (default: 48000)
        sampling frequency

    type: str
        ----------------------------------
        If type is 'wav' save as wav files
        If type is 'npy' save as npy files
        If type is 'mat' save as mat files
        ----------------------------------
    """
    # name, ext = os.path.splitext(name)
    # type = ext.replace('.', '')

    if type is 'wav':
        data = np.clip(data, -1, 1)

        data = [int(x * 32767.0) for x in data]

        binwave = struct.pack('h' * len(data), *data)

        wave_write = wave.open(name, 'w')
        p = (ch, nbit // 8, fs, len(binwave), 'NONE', 'not compressed')

        wave_write.setparams(p)
        wave_write.writeframes(binwave)
        wave_write.close()

    elif type is 'npy':
        np.save(name, data)

    elif type is 'mat':
        io.savemat(name, {"data": data})

    else:
        raise TypeError("type: 'wav', 'npy', 'mat'")

    print("save complete %s" % (name))


class Audio(object):
    """play and record

    Parameters:
    -----------
    in_data: array-like
        input data

    in_device: int (default: 2)
        input device index

    out_device: int (default: 1)
        output device index

    rec_time: float (default: 3.0)
        recording time

    ch: int (default: 8ch)
        number of channels

    fs: int
        sampling frequency ## 8kHz - 48kHz

    input: boolen (default: True)
        If input is True can record audio

    output: boolen (default: True)
        If output is True can play audio

    """
    def __init__(self, in_data=[], in_device=2, out_device=1, rec_time=3.0,
                 ch=8, fs=48000, input=True, output=True):

        self.audio = pyaudio.PyAudio()

        self.in_data = in_data
        self.in_device = in_device
        self.out_device = out_device
        self.rec_time = rec_time
        self.ch = ch
        self.fs = fs
        self.input = input
        self.output = output

        self._set_params()

    def __repr__(self):
        return repr(self.fs)

    def _set_params(self):

        play_data = self.in_data

        if play_data == []:
            play_data = np.zeros((int(self.rec_time * self.fs), 1),
                                 dtype=np.int16)
        else:
            data_max = np.abs(play_data).min()
            if not data_max < 1:
                play_data = 0.8 * play_data / data_max
                msg = "Play data is normalized"
                print(msg)

        self.play_data = play_data * (2.0**15)

        self.frames = []

    def _set_stream(self):

        self.stream = self.audio.open(format=FORMAT,
                                      channels=self.ch,
                                      rate=self.fs,
                                      input=self.input,
                                      output=self.output,
                                      input_device_index=self.in_device,
                                      output_device_index=self.out_device,
                                      frames_per_buffer=CHUNK)

    def _terminate(self):

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


def play(data, device=1, ch=1, fs=48000):
    """play audio

    Parameters:
    -----------
    data: array-like
        input data

    device: int (default: 1)
        output device index

    ch: int (default: 1ch)
        number of channels

    fs: int
        sampling frequency ## 8kHz - 48kHz
    """

    pa = Audio(in_data=data, out_device=device, ch=ch, fs=fs,
               input=False, output=True)

    byte_data = _encode(pa.play_data)

    pa._set_stream()

    print("playing...")
    pa.stream.write(byte_data)

    pa._terminate()


def record(device=2, rec_time=3.0, ch=8, fs=48000):
    """record a few seconds of audio

    Parameters:
    -----------
    device: int (default: 2)
        input device index

    rec_time: float (default: 3.0)
        recording time

    ch: int (default: 8ch)
        number of channels

    fs: int
        sampling frequency ## 8kHz - 48kHz

    Returns:
    --------
    rec_data: np.ndarray, shape(`chunk_size`, `channels`)
        recording data that has been decoded
    """

    pa = Audio(in_device=device, rec_time=rec_time, ch=ch, fs=fs,
               input=True, output=False)

    pa._set_stream()

    print("recording...")

    for _ in tqdm(range(int(fs / CHUNK * rec_time))):
        data = pa.stream.read(CHUNK)
        pa.frames.append(data)

    print("finished recording")

    pa._terminate()

    rec_data = _decode(pa.frames, ch)

    # if preview is True:
    #
    #     play(rec_data, device=1, ch=1)

    return rec_data


def _decode(in_data, channels):
    """Convert a byte stream into a numpy array (`chunk_size`, `channels`)

    Parameters:
    -----------
    in_data: bytes
        input byte stream data

    channels: int
        number of channels

    Returns:
    --------
    result: np.ndarray, shape (`chunk_size`, `channels`)
        converted numpy array
    """

    recbuff = b''.join(in_data)  # string concatenation
    result = np.fromstring(recbuff, dtype=np.int16)

    result = result / (2.0**15)

    data = np.clip(result, -1, 1)

    result = np.reshape(result, (-1, channels))
    return result


def _encode(in_data):
    """Convert a numpy array into a byte stream for PyAudio

    Parameters:
    -----------
    in_data: np.ndarray, shape (`chunk_size`, `channels`)
        converted numpy array

    Returns:
    --------
    result: bytes
        input byte stream data
    """

    result = in_data.astype(np.int16).tostring()
    return result
