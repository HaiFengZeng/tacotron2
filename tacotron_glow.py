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
import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from layers import ConvNorm


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(0.5)

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        for conv in self.convolutions:
            x = self.dropout(F.relu(conv(x)))
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = self.dropout(F.relu(conv(x)))

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        # Compute log determinant
        W = self.conv.weight.squeeze()
        log_det_W = batch_size * n_of_groups * torch.logdet(W)

        if reverse:
            # Reverse computation
            W_inverse = W.inverse()
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == 'torch.cuda.HalfTensor':
                W_inverse = W_inverse.half()
            z = F.conv1d(z, W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            z = self.conv(z)
            return z, log_det_W


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                 kernel_size):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        assert (n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_layers = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()
        self.cond_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels, 1)
            cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            self.cond_layers.append(cond_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
                res_layer = torch.nn.utils.weight_norm(res_layer, name='weight')
                self.res_layers.append(res_layer)

            skip_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
            skip_layer = torch.nn.utils.weight_norm(skip_layer, name='weight')
            self.skip_layers.append(skip_layer)

    def forward(self, forward_input):
        spec, text_condition = forward_input
        spec = self.start(spec)

        for i in range(self.n_layers):
            in_act = self.in_layers[i](spec)
            # print(in_act.size())
            condition = self.cond_layers[i](text_condition)
            in_act = in_act + condition

            t_act = torch.nn.functional.tanh(in_act[:, :self.n_channels, :])
            s_act = torch.nn.functional.sigmoid(in_act[:, self.n_channels:, :])
            acts = t_act * s_act

            if i < self.n_layers - 1:
                res_acts = self.res_layers[i](acts)
                spec = res_acts + spec

            if i == 0:
                output = self.skip_layers[i](acts)
            else:
                output = self.skip_layers[i](acts) + output
        return self.end(output)


class WaveGlow(torch.nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every,
                 n_early_size, WN_config, hparams):
        super(WaveGlow, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.embedding = torch.nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        self.encoder = Encoder(hparams)
        self.upsample = torch.nn.ConvTranspose1d(hparams.encoder_embedding_dim,
                                                 hparams.encoder_embedding_dim,
                                                 60, stride=50, padding=5)
        assert (n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group

        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            # print(k, n_half, n_remaining_channels)
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, hparams.encoder_embedding_dim, **WN_config))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, forward_input):
        """
        forward_input[0] = mel_text_conditionrogram:  batch x n_mel_channels x frames
        forward_input[1] = text_input_seq: batch x encoding_channels x time
        """
        text_seq, spec = forward_input
        embedding_input = self.embedding(text_seq).transpose(2, 1)
        text_condition = self.encoder(embedding_input)  # rnn encoding
        # print('encoding: ', text_condition.size())
        text_condition = self.upsample(text_condition.transpose(2, 1))

        spec = spec.transpose(2, 1).unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spec = spec.contiguous().view(spec.size(0), spec.size(-1), -1)

        output_spec = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_spec.append(spec[:, :self.n_early_size, :])
                spec = spec[:, self.n_early_size:, :]

            spec, log_det_W = self.convinv[k](spec)
            log_det_W_list.append(log_det_W)

            n_half = int(spec.size(1) / 2)
            spec_0 = spec[:, :n_half, :]
            spec_1 = spec[:, n_half:, :]

            output = self.WN[k]((spec_0, text_condition))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            spec_1 = torch.exp(log_s) * spec_1 + b
            log_s_list.append(log_s)

            spec = torch.cat([spec_0, spec_1], 1)

        output_spec.append(spec)
        return torch.cat(output_spec, 1), log_s_list, log_det_W_list

    def infer(self, inputs, sigma=1.0):
        # text_seq, inputs_length = inputs
        embedding = self.embedding(inputs).transpose(2, 1)
        text_condition = self.encoder(embedding)
        text_condition = self.upsample(text_condition.transpose(2, 1))
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        text_condition = text_condition[:, :, :-time_cutoff]
        print('text_condition size=', text_condition.size())
        if text_condition.type() == 'torch.cuda.HalfTensor':
            spec = torch.cuda.HalfTensor(text_condition.size(0), self.n_remaining_channels,
                                         text_condition.size(2)).normal_()
        else:
            spec = torch.cuda.FloatTensor(text_condition.size(0), self.n_remaining_channels,
                                          text_condition.size(2)).normal_()

        spec = torch.autograd.Variable(sigma * spec)

        for k in reversed(range(self.n_flows)):
            n_half = int(spec.size(1) / 2)
            spec_0 = spec[:, :n_half, :]
            spec_1 = spec[:, n_half:, :]

            output = self.WN[k]((spec_0, text_condition))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            spec_1 = (spec_1 - b) / torch.exp(s)
            spec = torch.cat([spec_0, spec_1], 1)

            spec = self.convinv[k](spec, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                if text_condition.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(text_condition.size(0), self.n_early_size,
                                              text_condition.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(text_condition.size(0), self.n_early_size,
                                               text_condition.size(2)).normal_()
                spec = torch.cat((sigma * z, spec), 1)

        spec = spec.permute(0, 2, 1).contiguous().view(spec.size(0), -1).data
        return spec

    def remove_weightnorm(self):
        waveglow = copy.deepcopy(self)
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_layers = remove(WN.res_layers)
            WN.skip_layers = remove(WN.skip_layers)
        self = waveglow


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
