# -*- coding: utf-8 -*-

import time
import numpy as np

from scipy import fftpack, signal
from numba import jit, i8, f8
from tqdm import tqdm_notebook as tqdm

EPS = np.spacing(1)


@jit(f8[:](f8[:, :], i8))
def corr_delay(data, mic_num):
    """calculate a delay using a correlation"""

    length = np.shape(data)[0]
    nch = np.shape(data)[1]
    estim_delay = np.zeros(nch, dtype="float64")

    print("Calculate delay")
    for ch in tqdm(range(nch)):
        corr = np.correlate(data[:, mic_num], data[:, ch], "full")
        estim_delay[ch] = corr.argmax() - (length - 1)

    return estim_delay


# def _mssft_time2(data, shift):
#     """shift the signal"""
#
#     length = np.shape(data)[0]
#
#     if shift > 0:
#         data = np.concatenate((data[length - shift:length],
#                                data[0:length - shift]))
#     else:
#         shift = -shift
#         data = np.concatenate((data[shift:length], data[0:shift]))
#
#     return data

@jit(f8[:](f8[:, :]))
def delay_sum(data):
    """Delay and Sum beamforming"""

    length = np.shape(data)[0]
    nch = np.shape(data)[1]

    lag1 = corr_delay(data, mic_num=3)
    lag2 = corr_delay(data, mic_num=7)

    data_delay1 = np.zeros(np.shape(data), dtype="float64")
    data_delay2 = np.zeros(np.shape(data), dtype="float64")

    # 遅延修正
    for ch in range(nch):
        data_delay1[:, ch] = np.roll(data[:, ch], int(lag1[ch]))
        data_delay2[:, ch] = np.roll(data[:, ch], int(lag2[ch]))

    data_sum1 = np.zeros((length, 1), dtype="float64")
    data_sum2 = np.zeros((length, 1), dtype="float64")

    # 遅延信号の総和
    for ch in range(nch):
        data_sum1[:, 0] = data_sum1[:, 0] + data_delay1[:, ch]
        data_sum2[:, 0] = data_sum2[:, 0] + data_delay2[:, ch]

    data_sum1 = data_sum1 / nch
    data_sum2 = data_sum2 / nch

    data_sum1 = np.reshape(data_sum1, (np.shape(data_sum1)[0], ))
    data_sum2 = np.reshape(data_sum2, (np.shape(data_sum2)[0], ))

    return data_sum1, data_sum2


# @jit((f8[:], f8[:])f8[:, :])
def null_beam(data):
    """Null beamformer"""

    length = np.shape(data)[0]
    nch = np.shape(data)[1]
    timeDiff1 = np.zeros(nch)
    timeDiff2 = np.zeros(nch)
    fs = 48000
    d1 = 0.0283
    c = 343
    angle1 = np.radians(50)
    angle2 = -np.radians(50)
    for i in range(nch):
        timeDiff1[i] = d1 * np.sin(angle1) / c * (i - 1)

    lag1 = np.round(timeDiff1 * fs)

    for i in range(nch):
        timeDiff2[i] = d1 * np.sin(angle2) / c * (i - 1)

    lag2 = np.round(timeDiff2 * fs)

    # # 遅延量計算
    # lag = corr_delay(data, mic_num=mic_num)

    data_delay1 = np.zeros(np.shape(data), dtype="float64")
    data_delay2 = np.zeros(np.shape(data), dtype="float64")

    # 遅延修正
    for ch in range(nch):
        data_delay1[:, ch] = np.roll(data[:, ch], int(lag1[ch]))
        data_delay2[:, ch] = np.roll(data[:, ch], int(lag2[ch]))

    data_sum1 = np.zeros((length, 1), dtype="float64")
    data_sum2 = np.zeros((length, 1), dtype="float64")

    # 偶数番目の符号を反転
    data_delay1[:, 1] = data_delay1[:, 1] * (-1)
    data_delay2[:, 1] = data_delay2[:, 1] * (-1)

    # 遅延信号の総和
    for ch in range(nch):
        data_sum1[:, 0] = data_sum1[:, 0] + data_delay1[:, ch]
        data_sum2[:, 0] = data_sum2[:, 0] + data_delay2[:, ch]

    data_sum1 = data_sum1 / nch
    data_sum2 = data_sum2 / nch

    data_sum1 = np.reshape(data_sum1, (np.shape(data_sum1)[0], ))
    data_sum2 = np.reshape(data_sum2, (np.shape(data_sum2)[0], ))

    return data_sum1, data_sum2


def mix_snr(speech, noise, snr=0):
    """
    mix two sound sources with a desired SNR

    Parameters:
    -----------
      speech: input speech sound source
       noise: sound source 2
         snr: desired SNR[dB] (default: 0)

    Returns:
    --------
          mix: mixed signal with specified SNR
       signal: sound source in the mixture signal 1
        noise: sound source in the mixture signal 2
         coef: mixture coefficient

    """

    # delete silent point
    n_filter = 2000
    ave = np.convolve(abs(speech).flatten(), np.ones(n_filter)/n_filter, mode='same')
    candidacy = ave[np.argsort(ave)[-int(len(ave) * 0.05)]]
    threshold = 20 * np.log10(candidacy.mean()) - 21.0
    _speech = np.delete(speech, obj=np.where(20 * np.log10(ave) < threshold))

    inSNR = cul_snr(_speech, noise)
    coef = 10**((inSNR - snr) / 20)

    # print("INPUT SNR: {}[dB]".format(inSNR))

    signal = speech
    noise = noise * coef

    mix = signal + noise

    normCoef = np.max(abs(mix))

    if normCoef >= 1:
        mix = mix / normCoef
        signal = signal / normCoef
        noise = noise / normCoef
        print('The signals are normalized.')

    return mix, signal, noise, coef


def cul_snr(data1, data2, axis=None):
    """calculate input SNR"""

    return 20 * np.log10(np.sum(abs(data1), axis=axis) / np.sum(abs(data2), axis=axis))


def cal_sp(play_num, time=20, fs=48000, device=2):
    """
    スピーカキャリブレーション用の信号再生
    1000Hzの正弦波信号を再生する

    Parameters
    ----------
    play_num : int
        再生するスピーカの番号

    time : int, optional
        信号を再生する長さ[sec.]

    fs : int, optinal
        サンプリング周波数

    device : int, optional
        再生デバイス番号

    Last update: Thu Nov 17 2016
    @author t-take
    """

    FREQ = 1000
    # 再生する信号(1000Hzの正弦波)
    sinwave = np.sin(2 * np.pi * FREQ * np.arange(0, time, 1 / fs)) * 0.8

    # 再生
    audio.play(sinwave, fs, firstch=play_num, lastch=play_num,
               device_index=device)


def make_tsp(point, eff_point=None):
    """
    TSP信号の生成

    Parameters
    ----------
    point : int
        TSP信号の長さ

    eff_point : int
        TSP信号信号の実効長

    Return
    ------
    : ndarray, shape(`time series`)
        TSP信号

    Reference
    ---------
    1 - 「音響インパルス応答計測の基礎」, 金田豊, 2013.08.28.
        http://www.asp.c.dendai.ac.jp/ASP/IRseminor2013.pdf

    Last update: Sun May 28 2017
    @author t-take
    """
    # 定数パラメータの設定
    if eff_point is None:
        eff_point = point * 150 // 1024
    a = 2.0 * np.pi * eff_point / ((point // 2)**2)   # 指数ベキの中の係数
    shift = point // 2 - eff_point * 2              # 円状シフト点数

    tsp_f = np.zeros(point, dtype=np.complex)   # 周波数領域でのTSP信号

    # 周波数領域でのTSP信号の生成
    tsp_f[:point // 2 + 1] = np.exp(-1j * a * np.arange(0, point // 2 + 1)**2)
    tsp_f[point // 2 + 1:] = np.exp(1j * a *
                                    np.arange(point // 2 - 1, 0, -1)**2)

    # 周波数領域のTSP信号を時間信号に変換
    tsp_t = fftpack.ifft(tsp_f).real        # 逆フーリエ変換で時間信号に変換，実数部のみを抽出
    tsp_t = np.roll(tsp_t, shift)       # shift点数分回転(circshift)
    tsp_t = tsp_t / abs(tsp_t).max()    # 正規化

    return tsp_t


def multi_tsp(n_ch, n_add, tsp_shift=2**16, len_tsp=2**18, eff_point=None):
    """
    多チャンネルTSP信号の作成

    Parameters
    ----------
    n_ch : int
        チャンネル数

    n_add : int
        同期加算回数

    tsp_shift : int
        シフト点数

    len_tsp : int
        1つのTSP長

    eff_point : int or None
        1つのTSPの実効長

    Return
    ------
    : ndarray, shape(`time series`, `channel`)
        多チャンネルTSP信号

    Last update: Tue Nov 22 2016
    @author t-take
    """

    if eff_point is None:
        eff_point = len_tsp * 150 // 1024

    # 1chのTSP信号作成
    tsp_sig = make_tsp(len_tsp, eff_point)

    # マルチチャンネル化
    rtn_val = np.zeros((tsp_shift * n_add * n_ch + len_tsp, n_ch))
    temp = 0
    for add in range(n_add):
        for ch in range(n_ch):
            rtn_val[temp:len_tsp + temp, ch] += tsp_sig   # TSP信号の格納
            temp += tsp_shift                          # タイミングのインクリメント
    return rtn_val


def impres(rectsp, tsp, n_sp, n_mic, tsp_shift, n_add):
    """
    インパルス応答推定
    impluse response estimation

    Parameters
    ----------
    rectsp : ndarray, shape(`time series`, `chunnel of loud speakers`)
        観測されたTSP信号

    tsp : ndarray, shape(`time series`)
        測定に使用した１つのTSP信号
        すなわち，同期加算なし，チャンネルによるシフトもなしのTSP信号
        すなわち，make_tspで作成されるようなTSP信号

    n_sp : int
        観測に使用したスピーカの数

    n_mic : int
        観測に使用したマイクの数

    n_add : int
        観測時のTSP信号の同期加算回数

    Return
    ------
    : ndarray, shape(`time series`, `ch of mic`, `ch of speaker`)
        インパルス応答

    Last update: Tue May 30 2017
    @author t-take
    """

    len_tsp = len(tsp)  # １つのtsp信号の長さ
    # １つのTSP信号に対する観測信号とインパルス応答
    sync_tsp = np.zeros((len(tsp), n_mic, n_sp))

    # 録音されたTSP信号をチャンネルごとに分解
    for add in range(n_add):
        for mic in range(n_mic):
            tim = 0
            for sp in range(n_sp):
                sync_tsp[:, mic, sp] = (sync_tsp[:, mic, sp] +
                                        rectsp[tim:tim + len_tsp, mic])
                tim += tsp_shift
    sync_tsp = sync_tsp / n_add

    # 周波数領域のTSP信号
    tsp_f = fftpack.fft(tsp)
    tsp_fh = np.conj(tsp_f)     # 周波数領域のTSP信号の共役転置(クロススペクトル)

    # クロススペクトル法によるインパルス応答の推定
    impres = np.zeros_like(sync_tsp)
    for mic in range(n_mic):
        for sp in range(n_sp):
            _res = fftpack.fft(sync_tsp[:, mic, sp], len_tsp)
            impres[:, mic, sp] = (fftpack.ifft((tsp_fh * _res) /
                                               (tsp_fh * tsp_f)).real)

    impres = impres[:tsp_shift]

    return impres


def conv_impres(indata, impres, fs=48000):
    "convolution impulse response"
    d_len, n_components = np.shape(indata)
    i_len, n_mic, n_spekers = np.shape(impres)

    length = d_len + i_len - 1

    data_ = np.zeros((length, n_mic), dtype="float64")

    for m in range(n_mic):
        for n in range(n_components):
            data_[:, m] += signal.fftconvolve(indata[:, n], impres[:, m, n])

    data = data_[:d_len, :]

    return data


def corr_sort(basis, activation, sepa_nb=60, learn_nb=30):
    "sort basis matrix"

    _R_sum = np.zeros((2, sepa_nb), dtype=np.float32)
    R_sum = np.zeros(sepa_nb, dtype=np.float32)
    # R_sort = np.zeros(sepa_nb, dtype=np.float32)
    # basis_sort = np.zeros(np.shape(basis), dtype=np.float32)
    # active_sort = np.zeros(np.shape(activation), dtype=np.float32)

    R = np.corrcoef(basis.T)

    for nb in range(sepa_nb):
        _R_sum[0, nb] = np.sum(R[0:learn_nb, nb])
        _R_sum[1, nb] = np.sum(R[learn_nb:sepa_nb, nb])
        R_sum[nb] = _R_sum[1, nb] - _R_sum[0, nb]

    n_index = np.where(R_sum < 0)
    p_index = np.where(R_sum > 0)

    # index = np.argsort(R_sum[2, :])
    #
    # for nb in range(sepa_nb):
    #     basis_sort[:, nb] = basis[:, index[nb]]
    #     active_sort[nb, :] = activation[index[nb], :]
    #
    # __R = np.corrcoef(basis_sort.T)
    # __R_sum = np.zeros(sepa_nb)
    #
    # for nb in range(sepa_nb):
    #     R_sum[0, nb] = np.sum(__R[0:learn_nb, nb])
    #     R_sum[1, nb] = np.sum(__R[learn_nb:sepa_nb, nb])
    #     R_sum[2, nb] = R_sum[1, nb] - R_sum[0, nb]
    #
    # __R_sum = R_sum[2, :]

    basis_1_sort = np.zeros(np.shape(basis), dtype="float64")
    basis_2_sort = np.zeros(np.shape(basis), dtype="float64")

    basis_1_sort[:, n_index] = basis[:, n_index]
    basis_2_sort[:, p_index] = basis[:, p_index]

    approx1 = np.dot(basis_1_sort, activation)
    approx2 = np.dot(basis_2_sort, activation)

    return approx1, approx2

def pre_emphasis_filtering(data, p=0.97):
    """高域協調フィルタリング"""
    return signal.lfilter([1, -p], 1, data)

def freq_pre_emphasis_filtering(data, nfft, p=0.97):
    """周波数領域における高域協調フィルタリング"""
    # フィルタ係数
    pre = [1.0, -p]
    # フィルタ応答
    wpre, hpre = signal.freqz(pre, worN=nfft // 2)
    # フィルタ畳み込み（周波数領域）
    return wpre[:, np.newaxis] * data

def _hz_to_mel(freq):
    """Convert Hz to Mels"""

    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mels):
    """Convert mel bin numbers to frequencies"""
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def mel_freq(n_mels=128, fmin=0.0, fmax=11025.0):
    """Compute the center frequencies of mel bands.

    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        number of Mel bins

    fmin      : float >= 0 [scalar]
        minimum frequency (Hz)

    fmax      : float >= 0 [scalar]
        maximum frequency (Hz)

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        vector of n_mels frequencies in Hz which are uniformly spaced on the
        Mel axis.
    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return _mel_to_hz(mels)


def melfilter(fs, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """
    MFCC抽出のための三角フィルタバンクの作成

    Parameters:
    -----------
    fs: int
        サンプリング周波数
    n_fft: int
        FFT点数
    n_mels:
        フィルタ数
    fmin:

    fmax:

    Return
    ------
    fbank: np.ndarray, shape(n_mels, 1 + n_fft/2)
        Mel filter bank
    """
    if fmax is None:
        fmax = float(fs) / 2

    n_mels = int(n_mels)
    fbank = np.zeros((n_mels, int(1 + n_fft // 2)))

    # 中心周波数
    fftfreqs = np.linspace(0, float(fs) / 2, int(1 + n_fft // 2),
                           endpoint=True)
    # 中心メル周波数
    mel_f = mel_freq(n_mels + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        fbank[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    fbank *= enorm[:, np.newaxis]

    return fbank[:, 1:]


def extract_mfcc(data, fs, nfft, nceps):
    """
    メル周波数ケプストラム係数(mel-cepstrum frequency cepstrum coefficient: MFCC)
    抽出

    Parameters
    ----------
    data : np.ndarray, shape(data sample, n_basis)
        MFCCを抽出するデータ
    fs : int
        サンプリング周波数[Hz]
    nfft : int
        窓長
    nceps : int
        メルケプストラムの何次元目まで用いるか
    Return
    ------
    mfcc : np.ndarray, shape(n_ceps, n_basis)
        MFCC
    """
    # # フィルタ係数
    # pre = [1.0, -0.97]
    # # フィルタ応答
    # wpre, hpre = signal.freqz(pre, worN=nfft // 2)
    # # フィルタ畳み込み（周波数領域）
    # data = wpre[:, np.newaxis] * data

    # フィルタ畳み込み（周波数領域）
    data = freq_pre_emphasis_filtering(data, p=0.97)
    # メルフィルタの生成
    fbank = melfilter(fs, nfft, n_mels=40)
    # メルフィルタリング
    mspec = np.log10(np.dot(fbank, data) + EPS)
    # 離散コサイン変換
    ceps = fftpack.realtransforms.dct(mspec, type=2,
                                      norm='ortho', axis=0)
    # MFCC算出
    mfcc = ceps[:nceps, :]

    return mfcc
