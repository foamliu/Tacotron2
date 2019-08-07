import argparse
import logging

import torch


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}

    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Tacotron2')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--max_norm', default=1, type=float, help='Gradient norm threshold to clip')
    # minibatch
    parser.add_argument('--batch_size', default=32, type=int)
    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
    parser.add_argument('--l2', default=0.0, type=float, help='weight decay (L2)')
    # logging
    parser.add_argument('--print_freq', default=10, type=int, help='Frequency of printing training infomation')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file, feature='fbank', dim=40, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10, save_feature=None):
    y, sr = librosa.load(input_file, sr=None)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == 'fbank':  # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature, tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')


# Based on:
# - https://github.com/candlewill/Griffin_lim/blob/master/utils/audio.py
# - https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import librosa.filters
import numpy as np
from scipy import signal

from scipy.io import wavfile
import config


def load_wav(path, sr):
    # Maybe use scipy.io.wavfile.read() + norm(divide max int16) to impl this?
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, config.sample_rate, wav.astype(np.int16))


def spectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(np.abs(D)) - config.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + config.ref_level_db)  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S ** config.power))  # Reconstruct phase


def melspectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)


def inv_melspectrogram(melspectrogram):
    S = _mel_to_linear(_db_to_amp(_denormalize(melspectrogram)))  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S ** 1.5))  # Reconstruct phase


# Based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(config.griffin_lim_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y):
    return librosa.stft(y=y, n_fft=config.fft_size, hop_length=get_hop_size())


def _istft(y):
    return librosa.istft(y, hop_length=get_hop_size())


# Conversions:

_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(mel_spectrogram):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    # return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))
    return np.maximum(hparams.floor_freq, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis():
    # n_fft = (hparams.num_freq - 1) * 2
    # return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)
    return librosa.filters.mel(config.sample_rate, n_fft=config.fft_size, n_mels=config.num_mels, fmin=config.min_freq,
                               fmax=config.max_freq)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _preemphasis(x):
    return signal.lfilter([1, -config.preemphasis], [1], x)


def _inv_preemphasis(x):
    return signal.lfilter([1], [1, -config.preemphasis], x)


def _normalize(S):
    return np.clip(
        (2 * config.max_abs_value) * ((S - config.min_level_db) / (-config.min_level_db)) - config.max_abs_value,
        -config.max_abs_value, config.max_abs_value)


def _denormalize(D):
    return (((np.clip(D, -config.max_abs_value,
                      config.max_abs_value) + config.max_abs_value) * -config.min_level_db / (
                     2 * config.max_abs_value))
            + config.min_level_db)


def get_hop_size():
    hop_size = config.hop_size
    if hop_size is None:
        assert config.frame_shift_ms is not None
        hop_size = int(config.frame_shift_ms / 1000 * config.sample_rate)
    return hop_size
