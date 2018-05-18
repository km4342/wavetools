# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests

from .analysis import stft
from datetime import datetime

EPS = np.spacing(1)

def notify(msg="Terminated!", img=None):
    "LINE Notify"
    url = "https://notify-api.line.me/api/notify"
    token = os.environ.get("LINE_NOTIFY")
    headers = {"Authorization": "Bearer "+token}

    time = datetime.now().strftime("%y/%m/%d %H:%M")

    payload = {"message": time+"\n"+msg}

    if img:
        files = {"imageFile": open(img, "rb")}
        r = requests.post(url, headers=headers, params=payload, files=files)
    else:
        r = requests.post(url, headers=headers, params=payload)


def plot(data, size=4096, shift=1024, fs=48000, show=True):
    "easy show waveform and spectrogram"

    # axis = np.arange(0, len(data) / fs, 1 / fs)
    axis = np.linspace(0, len(data) / fs, len(data))

    stft_data, taxis, faxis = _spectrogram(data, size=size, shift=shift,
                                           win=np.hanning, fs=fs)

    taxis, faxis = np.meshgrid(taxis, faxis)

    sns.set_style('white')
    sns.set_context('poster')

    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[20, 1], height_ratios=[1, 2])

    ax1 = plt.Subplot(fig, gs[0, 0])
    fig.add_subplot(ax1)
    ax1.plot(axis, data)
    ax1.axis([0, max(axis), -1, 1])
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Amplitude")

    ax2 = plt.Subplot(fig, gs[1, 0])
    fig.add_subplot(ax2)
    pcm = ax2.pcolormesh(taxis, faxis, 10 * np.log10(stft_data),
                         cmap='inferno')
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Frequency [Hz]")

    ax3 = plt.Subplot(fig, gs[1, 1])
    fig.add_axes(ax3)
    cbar = plt.colorbar(pcm, cax=ax3)
    cbar.set_label('Power[dB]', labelpad=15, rotation=270)

    plt.tight_layout()

    if show is True:
        plt.show()


def spect(data, size, shift, name="", fs=48000, show=True,
          cmap='inferno'):
    """plot spectrogram"""

    stft_data, taxis, faxis = _spectrogram(data, size=size, shift=shift,
                                           win=np.hanning, fs=fs)

    taxis, faxis = np.meshgrid(taxis, faxis)

    plt.figure(figsize=(10, 6))
    sns.set_style('white')
    sns.set_context('poster')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.pcolormesh(taxis, faxis, 10 * np.log10(stft_data),
                   vmin=-200, vmax=0, cmap=cmap)
    plt.title(name)
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    cbar = plt.colorbar()
    cbar.set_label('Power[dB]', labelpad=15, rotation=270)
    plt.tight_layout()

    if show is True:
        plt.show()


def _spectrogram(data, size, shift, win, fs):
    "calculate spectrogram for plot"

    stft_data = stft(data, size=size, shift=shift, win=win, fs=fs)
    stft_data = abs(stft_data)**2

    frames = np.shape(stft_data)[1]

    taxis = np.arange(0, shift / fs * frames, shift / fs)
    faxis = np.arange(0, fs / 2, fs / size)

    return stft_data, taxis, faxis


def plot_wave(data, name="", fs=48000, show=True):
    """plot waveform"""

    axis = np.arange(0, len(data) / fs, 1 / fs)

    plt.figure()
    sns.set_style('white')
    sns.set_context('poster')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(axis, data)
    plt.axis([0, max(axis), -1, 1])
    plt.title(name)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")

    if show is True:
        plt.show()


def plot_spectrum(data, name="", fs=48000, show=True):
    """plot spectrum"""

    if not np.isreal(data).all():
        data = np.abs(data)**2

    size = np.shape(data)[0] * 2

    axis = np.arange(0, fs / 2, fs / size)

    plt.figure(figsize=(10, 6))
    sns.set_style('white')
    sns.set_context('poster')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(axis, 10 * np.log10(data + EPS))
    plt.axis([1, 8000, -100, 0])
    plt.title(name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.tight_layout()

    if show is True:
        plt.show()


def plot_spect(data, size, shift, name="", fs=48000, show=True,
               cmap='inferno'):
    """plot spectrogram"""

    if not np.isreal(data).all():
        data = np.abs(data)**2

    frames = np.shape(data)[1]

    taxis = np.arange(0, shift / fs * frames, shift / fs)
    faxis = np.arange(0, fs / 2, fs / size)

    taxis, faxis = np.meshgrid(taxis, faxis)

    plt.figure(figsize=(10, 6))
    sns.set_style('white')
    sns.set_context('poster')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.pcolormesh(taxis, faxis, 10 * np.log10(data),
                   vmin=-200, vmax=0, cmap=cmap)
    plt.title(name)
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    cbar = plt.colorbar()
    cbar.set_label('Power[dB]', labelpad=15, rotation=270)
    plt.tight_layout()

    if show is True:
        plt.show()


def plot_dendrogram(tree):
    """plot dendrogram"""
    plt.figure(figsize=(10, 6))
    sns.set_style('white')
    sns.set_context('poster')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel("basis index")
    plt.ylabel("distance")
    dendrogram(tree)

    if show is True:
        plt.show()


def plot_map(data, name="", xticks=5, yticks=5, show=True):
    """plot heatmap"""

    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.tick_params(labelbottom="off",bottom="off") # x軸の削除
    plt.tick_params(labelleft="off",left="off") # y軸の削除
    plt.title(name)
    plt.imshow(data, vmin=0, vmax=np.floor(np.max(data)), cmap='jet',
               origin="lower")
    plt.xticks(range(0, data.shape[1], xticks))
    plt.yticks(range(0, data.shape[0], yticks))
    plt.gca().grid(linestyle='')

    if show is True:
        plt.show()
