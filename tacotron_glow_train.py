# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import os
import torch

# =====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
# =====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from tacotron_glow import WaveGlow, WaveGlowLoss

from data_utils import TextMelCollate, TextMelLoader
from hparams import create_hparams
from model import Tacotron2
from logger import WaveGlowLogger


# Encoder is used as a condition extractor before upsample

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    eval_iteration = checkpoint_dict['iteration']
    iteration = checkpoint_dict['eval_iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, iteration, eval_iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, eval_iteration, filepath, hparams=None):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config, hparams=hparams).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'eval_iteration': eval_iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, batch_size, seed, checkpoint_path, hparams):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    # =====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(sigma)
    if num_gpus >= 1:
        model = WaveGlow(**waveglow_config, hparams=hparams).cuda()
    else:
        model = WaveGlow(**waveglow_config, hparams=hparams)

    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    # =====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if one exists
    iteration, eval_iteration = 0, 0

    if checkpoint_path != "":
        model, optimizer, iteration, eval_iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1
        eval_iteration += 1
    # trainset = Mel2Samp(**data_config)

    trainset = TextMelLoader(
        audiopaths_and_text='./filelists/ljs_audio_text_train_filelist.txt', hparams=hparams)
    testset = TextMelLoader(
        audiopaths_and_text='./filelists/ljs_audio_text_test_filelist.txt', hparams=hparams)

    collate_fn = TextMelCollate(hparams, fixed_length=True)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1,
                              collate_fn=collate_fn,
                              shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    test_loader = DataLoader(testset, num_workers=1,
                             collate_fn=collate_fn,
                             shuffle=False,
                             sampler=train_sampler,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=True)
    log_path = os.path.join(output_directory, 'log-event')
    os.makedirs(log_path, exist_ok=True)
    logger = WaveGlowLogger(log_path)
    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    model.train()
    tacotron2 = Tacotron2(hparams)
    batch_parser = tacotron2.parse_batch
    # we use tacotron-2's pipeline
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            x, y = batch_parser(batch)
            text_padded, input_lengths, mel_padded, max_len, output_lengths = x
            # print(text_padded.size(), mel_padded.size())
            mel_padded, gate_padded = y
            outputs = model((text_padded, mel_padded))

            loss = criterion(outputs)
            logger.log_loss('train/loss', loss, iteration)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            print("{}:\t{:.9f}".format(iteration, reduced_loss))
            iteration += 1

        # model.eval()
        # for i, batch in enumerate(test_loader):
        #     x, y = batch_parser(batch)
        #     text_padded, input_lengths, mel_padded, max_len, output_lengths = x
        #     mel_padded, gate_padded = y
        #     outputs = model((text_padded, mel_padded))
        #     loss = criterion(outputs)
        #     logger.log_loss('eval/loss', loss, iteration)
        #     eval_iteration += 1

        if rank == 0:
            checkpoint_path = "{}/waveglow_epoch_{}".format(output_directory, epoch)
            save_checkpoint(model, optimizer, learning_rate, iteration, eval_iteration, checkpoint_path,
                            hparams=hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='waveglow.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument('-l', '--logdir', type=str, default='./waveglow-log/log', help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    hparams = create_hparams()
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name, **train_config, hparams=hparams)
