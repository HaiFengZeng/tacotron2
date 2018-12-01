import random
import numpy as np
import torch
import torch.utils.data
from hparams import create_hparams
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
import os


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, shuffle=True):
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, hparams.sort_by_length)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio = load_wav_to_torch(filename, self.sampling_rate)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, hparams, fixed_length=False):
        self.n_frames_per_step = hparams.n_frames_per_step
        self.max_encode_step = hparams.max_encode_step
        self.max_decode_step = hparams.max_decode_steps
        self.fixed_length = fixed_length
        self.hp = hparams

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0] if not self.fixed_length else self.max_encode_step
        if self.fixed_length:
            max_input_len = 200
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec with extra single zero vector to mark the end
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch]) + 1 if not self.fixed_length else self.max_decode_step
        if self.hp.start_pad:
            max_target_len += 1
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        if self.fixed_length:
            max_target_len = 1000
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        if self.hp.start_pad:
            for i in range(len(ids_sorted_decreasing)):
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, 1:mel.size(1) + 1] = mel
                gate_padded[i, 1 + mel.size(1):] = 1
                output_lengths[i] = mel.size(1) + 1
        else:
            for i in range(0, len(ids_sorted_decreasing)):
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, :mel.size(1)] = mel
                gate_padded[i, mel.size(1):] = 1
                output_lengths[i] = mel.size(1)
        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
