import os
from hparams import create_hparams, load_hparams
import matplotlib
import shutil

matplotlib.use("Agg")
import matplotlib.pylab as plt
# import IPython.display as ipd

import numpy as np
import torch

from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text
from scipy.io.wavfile import write

MAX_WAV_VALUE = 32768.0


def load_inference_model(hparams, name, ckpt_step, cuda=True):
    ckpt_path = './output-{}/checkpoint_{}'.format(name, ckpt_step)
    ckpt = torch.load(ckpt_path)['state_dict']
    ckpt_ = dict()
    for k in ckpt.keys():
        if k.startswith('module'):
            ckpt_['.'.join(k.split('.')[1:])] = ckpt[k]
        else:
            ckpt_[k] = ckpt[k]
    model = Tacotron2(hparams) if hparams.model == 'tacotron2' else Transformer(hparams)
    model.load_state_dict(ckpt_)
    model.eval()
    if cuda:
        model = model.cuda()
        model.eval()
    return model


def prepare_inputs(hparams, text=None, cuda=True):
    '''
    prepare input for model
    :param text: english sentence
    :return: input_sequence,input_length,input_pos size = [1,T]
    '''
    if not text:
        # sample a sentence from validation datasets
        with open(hparams.validation_files) as f:
            texts = f.readlines()
        index = np.random.randint(0, len(texts) - 1)
        text = texts[index]
    input_seq = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    input_seq = torch.LongTensor(input_seq)
    input_pos = torch.LongTensor([range(1, input_seq.size(1) + 1)])
    if cuda:
        return input_seq.cuda(), input_pos.cuda()
    return input_seq, input_pos


def get_mel(filename, hparams):
    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)
    audio = load_wav_to_torch(filename, hparams.sampling_rate)
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def get_texts(audio_texts=None, _type='train'):
    if not audio_texts:
        audio_texts = 'filelists/ljs_audio_text_{}_filelist.txt'.format(_type)
        with open(audio_texts, encoding='utf-8') as f:
            lines = f.readlines()[:2]
            return list(map(lambda x: x.strip('\n').split('|')[0], lines)), list(
                map(lambda x: x.strip('\n').split('|')[-1], lines))
    else:
        with open(audio_texts, encoding='utf-8') as f:
            lines = f.readlines()[:2]
            return None, list(map(lambda x: x.strip('\n'), lines))


def inference_texts(model, hp, target_texts, step, model_name, vocoder, waveglow, f_type='mel', _type='train',
                    postnet=True):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    sample_rate = 22050
    original_audio, texts = target_texts
    save_target = 'generate/{}-step-{}'.format(model_name, step)
    stft = TacotronSTFT(
        hp.filter_length, hp.hop_length, hp.win_length,
        hp.n_mel_channels, hp.sampling_rate, hp.mel_fmin,
        hp.mel_fmax)

    os.makedirs(save_target, exist_ok=True)
    for i, text in enumerate(texts):
        print(text)
        if original_audio:
            target_name = '{}-target-{}.wav'.format(_type, i)
            path = os.path.join(save_target, target_name)
            shutil.copy2(original_audio[i], path, )
        inputs = prepare_inputs(hp, text)
        if torch.cuda.device_count() > 1:
            with torch.no_grad():
                predict = model.module.inference(inputs, postnet=postnet)
        else:
            with torch.no_grad():
                predict = model.inference(inputs, postnet=postnet)
        name = '{}-{}-{}-{}.wav'.format(_type, f_type, i, vocoder)

        path = os.path.join(save_target, name)
        if vocoder == 'griffin_lim':
            mel_decompress = stft.spectral_de_normalize(predict)
            mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
            spec_from_mel_scaling = 1000
            spec_from_mel = torch.mm(mel_decompress[0], stft.mel_basis)
            spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
            spec_from_mel = spec_from_mel * spec_from_mel_scaling
            print(spec_from_mel.size())
            waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), stft.stft_fn, 60)
            write(path, sample_rate, waveform[0].data.cpu().numpy())
        elif vocoder == 'waveglow' and waveglow:
            with torch.no_grad():
                audio = MAX_WAV_VALUE * waveglow.infer(predict, sigma=1.0)[0]
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            write(path, sample_rate, audio)


def test_inference():
    hparams = create_hparams()
    name = 'position-encoding-train'
    h_path = './presets/{}.json'.format(name)
    hparams = load_hparams(h_path, hparams)
    step = 770 * 1000
    # model = load_inference_model(hparams, name, step)
    texts = get_texts('./sentence.txt')
    seq, pos = prepare_inputs(hparams, text=texts[0])
    model = load_inference_model(hparams, name, step)
    encoder_outputs = model.encoder.inference(seq, pos)
    print(encoder_outputs)


def inference():
    hparams = create_hparams()
    _type = 'test'
    name = 'pff'
    # name = 'position-encoding-train'
    vocoder = 'griffin_lim'
    # vocoder = 'waveglow'
    # name = 'multi-gpu'
    step = 320 * 1000
    waveglow_path = '/data1/hfzeng/work/tacotron/original_tacotron/waveglow/waveglow_old.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.remove_weightnorm()
    waveglow.cuda().eval()
    # texts = get_texts('./sentence.txt')
    h_path = './presets/{}.json'.format(name)
    hparams = load_hparams(h_path, hparams)
    model = load_inference_model(hparams, name, step)
    for _type in ['val']:#, 'val', 'test']:
        texts = get_texts(_type=_type)
        inference_texts(model, hparams, texts, step, name, vocoder=vocoder, waveglow=waveglow, _type=_type,
                        postnet=False)


if __name__ == '__main__':
    inference()
    # test_inference()
