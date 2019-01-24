# -*- coding: utf-8 -*-

import wave
import numpy as np
from numba import jit, i8, f8, c8, c16


def cepstrum(data, size, time, fs=48000, dim=20):
    """cepstrum analysis"""

    if not np.isreal(data).all():
        data = np.abs(data)**2

    data = 10 * np.log10(data)

    taxis = np.arange(0, time, 1 / fs)

    cps = np.real(np.fft.ifft(data))
    quefrency = taxis - min(taxis)

    cepdim = dim
    cpslif = np.array(cps)

    cpslif[cepdim: len(cpslif) - cepdim + 1] = 0

    ceps = np.fft.fft(cpslif)

    return ceps


def stft(data, size, shift=None, fs=48000, win=np.hanning):
    """Short-Time Fourier Transform"""

    if size % 2 != 0:
        raise ValueError("Please set 'FFT size' as an even number.")
    if not shift:
        shift = size//2

    win = win(size)

    rawlength = len(data)

    xf = np.zeros(rawlength + size, dtype="float64")
    xf[:rawlength] = data

    length = len(xf)  # length(update)

    # number of frames
    frames = int(np.floor((length - size + shift) / shift))

    stft_x = np.zeros([size, frames], dtype="complex64")

    for m in range(frames):
        start = m * int(shift)
        stft_x[:, m] = np.fft.fft(xf[start: start + size] * win)

    spectrogram = stft_x[0: size // 2, :]  # Spectrogram

    return spectrogram


def istft(data, length, shift, win=np.hanning, fs=48000):
    """Inverse Short-Time Fourier Transform"""

    freq, frames = np.shape(data)  # Frequency, number of Frames

    size = freq * 2  # frame size

    win = win(size)

    syntheWin = _synthesized_window(shift, win)  # Synthesized Window

    spect = np.zeros(size, dtype="complex64")
    wave = np.zeros(frames * shift + size, dtype="float64")

    for m in range(frames):
        start = m * int(shift)
        spect[: freq] = data[:, m]
        wave[start: start + size] = (wave[start: start + size] +
                                     np.fft.ifft(spect).real * syntheWin * 2)

    waveform = wave[0:length]  # discard zeroPadding

    return waveform


def reconst(data, length, size, shift, win=np.hanning, fs=48000, iter=100):
    """reconstruct spectrogram to estimate phaseSpectrum"""

    win = win(size)

    shape = np.shape(data)  # matrix shape

    amplitude = np.sqrt(data)  # amplitudeSpectrum
    phase = np.random.uniform(-np.pi, np.pi, shape)  # phaseSpectrum

    spect = np.zeros(shape, dtype="float64")
    spect = amplitude * np.exp(1j * phase)  # default spectrogram

    for _ in range(iter):

        waveform = istft(spect, length, shift, win, fs)

        approx_spect = stft(waveform, size, shift, win, fs)

        bias = approx_spect / abs(approx_spect)
        bias[np.isnan(bias)] = 1  # replace NaN

        spect = amplitude * bias  # replace amplitude

    return spect


@jit(f8[:](i8, f8[:]))
def _synthesized_window(shift, win):
    """create synthesized window for ISTFT"""

    size = len(win)
    syntheWin = np.zeros(size, dtype="float64")

    for n in range(shift):
        amp = 0.0
        for q in range(int(size / shift)):
            amp = amp + win[n + (q - 1) * shift] ** 2
        for q in range(int(size / shift)):
            syntheWin[n + (q - 1) * shift] = win[n + (q - 1) * shift] / amp

    return syntheWin
