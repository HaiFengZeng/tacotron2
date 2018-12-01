import tensorflow as tf
from text import symbols
import argparse, json
import os
import torch


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=5000,
        inference_step=2000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="file://distributed.dpt",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        dataset='cmp',
        model='transformer',  # [transformer,tacotron2]
        encoder_type='tacotron_encoder',
        enable_debug=True,
        wn=False,  # wn: weight normalization,bn batch normalization

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        # training_files='train_datasets/train_ecmp.txt',
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        # validation_files='train_datasets/test_ecmp.txt',
        validation_files='filelists/ljs_audio_text_test_filelist.txt',
        text_cleaners=['english_cleaners'],
        sort_by_length=False,
        start_pad=True,  # start with left padding for
        position_encoding_trainable=True,
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,  # if None, half the sampling rate

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=256,

        dropout=0.01,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=256,
        max_encode_step=200,
        num_head=8,
        encoder_layers=3,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=200,
        max_decode_steps=1000,
        gate_threshold=0.6,
        decoder_layers=3,
        decode_embedding_dim=256,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,
        attention_inner_dim=256,
        attention_plot={
            "enc-attn": [0, -1],
            "dec-attn": [0, -1],
            "enc-dec-attn": [0, -1]
        },

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-9,
        grad_clip_thresh=2,
        batch_size=4,
        mask_padding=False,  # set model's padded outputs to padded values,
        reset_optimizer=False,
        #######Model####################
        checkpoint_path='',

        ###########################
        # Loss Hyperparameters    #
        ###########################
        diff_loss_weights=20,  # if diff loss weights <0,set it to 0.
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams


def save_hparams_json(filename, hparams):
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('There is no directory to save {}'.format(filename))

    with open(filename, 'w') as f:
        json.dump(hparams.values(), f, sort_keys=True, separators=(',', ': '), indent=4)


def load_hparams(filename, hparams):
    """ load hparams from json file """
    with open(filename) as f:
        hparams.parse_json(f.read())
    return hparams


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
    return 'Hyperparameters:\n' + '\n'.join(hp)


if __name__ == '__main__':
    hparams = create_hparams()
    # print('CUDA:', torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Name of logging directory.', default='Tacotron')
    parser.add_argument('--mode',
                        help='type contains[new(gernare a single json config),'
                             'override(override a single json file),'
                             'all(update all preset)].',
                        default='new')
    args = parser.parse_args()
    os.makedirs('./presets', exist_ok=True)
    _path = './presets/{name}.json'.format(name=args.name)
    if args.mode == 'new':
        save_hparams_json(_path, hparams)
    elif args.mode == 'override':
        hparams = load_hparams(_path, hparams)
        save_hparams_json(_path, hparams)
    elif args.mode == 'all':
        for _p in os.listdir('./presets'):
            _path = os.path.join('./presets', _p)
            hparams = load_hparams(_path, hparams)
            save_hparams_json(_path, hparams)
