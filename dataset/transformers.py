import numpy as np
import librosa
import torch


class AudioRead:
    def __init__(self, sr=22050, offset=0.0, duration=None):
        self.sr = sr
        self.offset = offset
        self.duration = duration

    def __call__(self, x):
        y, sr = librosa.load(x, sr=self.sr, duration=self.duration, offset=self.offset)
        return y


class Zscore:
    def __init__(self, divide_sigma=False):
        self.divide_sigma = divide_sigma

    def __call__(self, x):
        x -= x.mean()
        if self.divide_sigma:
            x /= x.std()
        return x


class PadAudio:
    def __init__(self, sr=22050, pad_to=30):
        """
        Pad the input audio with zeros.
        If the input is longer than the desired length, trim the audio.
        :param pad_to: the desired length of audio (seceond)
        """
        self.pad_to = pad_to
        self.sr = sr

    def __call__(self, x):
        target_len = int(self.pad_to * self.sr)
        pad_len = abs(len(x) - target_len)
        if len(x) < target_len:  # pad
            x = np.hstack([x, np.zeros(pad_len)])
        elif len(x) > target_len:  # trim
            x = x[:target_len]

        return x


class Spectrogram:
    def __init__(self, sr=22050, n_fft=2048, hop_size=735, n_band=64, spec_type='mel'):
        """
        Derive spectrogram. Currently accept linear and Mel spectrogram.
        As the default input duration of spectrograms is 0.5s to the VAE,
        the default values of sr and hop_size are such that a 0.5s spectrogram has 15 frames,
        which is small enough ([f, t] = [64, 15]) to keep small number of parameters of the VAE.
        :param n_fft: size of short-time fourier transform
        :param hop_size: short-time window hop size
        :param n_band: number of frequency bins, ignored if spec_type='linear'
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_band = n_band
        self.spec_type = spec_type

    def __call__(self, x):
        assert self.spec_type in ['lin', 'mel', 'cqt'], "spec_type should be in ['lin', 'mel', 'cqt']"
        if self.spec_type == 'lin':
            S = librosa.core.stft(y=x, n_fft=self.n_fft, hop_length=self.hop_size)
            S = np.abs(S) ** 2  # power spectrogram

        elif self.spec_type == 'mel':
            S = librosa.feature.melspectrogram(y=x, sr=self.sr, n_fft=self.n_fft,
                                               hop_length=self.hop_size, n_mels=self.n_band)
            # melspectrogram has raised np.abs(S)**power, default power=2
            # so power_to_db is directly applicable
            S = librosa.core.power_to_db(S, ref=np.max)
        else:
            # TODO: implement CQT
            raise NotImplementedError

        return S


class MinMaxNorm:
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        x -= x.mean()
        x_min = x.min()
        x_max = x.max()
        nom = x - x_min
        den = x_max - x_min

        if abs(den) > 1e-4:
            return (self.max_val - self.min_val) * (nom / den) + self.min_val
        else:
            return nom


class SpecChunking:
    def __init__(self, duration=0.5, sr=22050, hop_size=735, reverse=False):
        """
        Slice spectrogram into non-overlapping chunks. Discard chunks shorter than the specified duration.
        :params duration: the duration (in sec.) of each spectrogram chunk
        :params sr: sampling frequency used to read waveform; used to calculate the size of each spectrogram chunk
        :parms hop_size: hop size used to derive spectrogram; used to calculate the size of each spectrogram chunk
        :params reverse: reverse the spectrogram before chunking;
                         set True if the end is more important than the begin of spectrogram
        TODO:
            [] Allow an input argument to indicate the overlapping amount between chunks
        """
        self.duration = duration
        self.sr = sr
        self.hop_size = hop_size
        self.chunk_size = int(sr * duration) // hop_size
        self.reverse = reverse

    def __call__(self, x):
        time_dim = 1  # assume input spectrogram with shape n_freqBand * n_contextWin
        n_contextWin = x.shape[time_dim]  # context window size of the input spectrogram
        # TODO: overlapping window size; with the amount of overlap as an input argument
        indices = np.arange(self.chunk_size, n_contextWin, self.chunk_size)  # currently non-overlapping chunking

        # reverse to keep the end content of spectrogram intact in the later discard
        # this is only used when the end is more important than the begin of spectrogram
        if self.reverse:
            x = np.flip(x, time_dim)

        x_chunk = np.split(x, indices_or_sections=indices, axis=time_dim)

        # reverse back if self.reverse=True
        if self.reverse:
            x_chunk = [np.flip(i, time_dim) for i in x_chunk[::-1]]

        # discard those short chunks
        x_chunk = [x_i for x_i in x_chunk if x_i.shape[time_dim] == self.chunk_size]

        return np.array(x_chunk)


class LoadNumpyAry:
    def __call__(self, x):
        return np.load(x)
