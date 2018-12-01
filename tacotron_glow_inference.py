from hparams import create_hparams
from tacotron_glow import WaveGlow
from inference import get_texts
import numpy as np
import torch
from text import text_to_sequence
from plotting_utils import save_spectrogram_image


def prepare_inputs(hparams, text=None, cuda=True):
    '''
    prepare input for model
    :param text: english sentence
    :return: input_sequence,input_length,input_pos size = [1,T]
    '''
    padding = 200
    if not text:
        # sample a sentence from validation datasets
        with open(hparams.validation_files) as f:
            texts = f.readlines()
        index = np.random.randint(0, len(texts) - 1)
        text = texts[index]
    input_seq = np.array(text_to_sequence(text, ['english_cleaners']))
    input_seq = np.pad(input_seq, (0, padding - len(input_seq)), 'constant')
    input_seq = torch.LongTensor(input_seq).unsqueeze(0)
    if cuda:
        return input_seq.cuda()
    return input_seq


def load_tacotorn_glow_model(ckpt_path, cuda=True):
    tacotron_glow = torch.load(ckpt_path)['model']
    tacotron_glow.remove_weightnorm()
    if cuda:
        tacotron_glow.cuda().eval()
    else:
        tacotron_glow.eval()
    return tacotron_glow


def inference(model, text_list):
    for text in text_list:
        inputs = prepare_inputs(text)
        mel = model.inference()


if __name__ == '__main__':
    hparams = create_hparams()
    text = 'once you know that you must put the crosshairs on the target and that is all that is necessary.'
    input_seq = prepare_inputs(hparams, text)
    print(input_seq.size())

    ckpt_path = './checkpoints/waveglow_10000'
    model = load_tacotorn_glow_model(ckpt_path)
    mel = model.infer(input_seq)
    mel = mel.contiguous().view(mel.size(0), 80, -1)
    mel = mel.data.cpu().numpy()[0]
    save_spectrogram_image(mel)
