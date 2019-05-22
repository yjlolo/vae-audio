import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, BaseVAE


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def spec_conv2d(n_layer=3, n_channel=[1, 32, 16, 8], n_freqBand=64, filter_size=[1, 3, 3], stride=[1, 2, 2]):
    """
    Construction of conv. layers. Note the current implementation always effectively turn to 1-D conv,
    inspired by https://arxiv.org/pdf/1704.04222.pdf.
    :param n_layer: number of conv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 )
    :param n_freqBand: number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified conv. layers
    TODO:
        [] directly use nn.Conv1d for implementation
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg

    # construct first layer
    conv_layers = [[
        nn.Conv2d(n_channel[0], n_channel[1], (n_freqBand, filter_size[0]), (1, stride[0])),
        nn.BatchNorm2d(n_channel[1]),
        nn.Tanh()
    ]]
    # construct the rest
    for i in range(n_layer - 1):
        in_channel = n_channel[1:][i]
        out_channel = n_channel[1:][i + 1]
        conv_layers.append([
            nn.Conv2d(in_channel, out_channel, (1, filter_size[1:][i]), (1, stride[1:][i])),
            nn.BatchNorm2d(out_channel),
            nn.Tanh()
        ])

    conv_layers = [j for i in conv_layers for j in i]

    return nn.Sequential(*conv_layers)


def spec_deconv2d(n_layer=3, n_channel=[1, 32, 16, 8], n_freqBand=64, filter_size=[1, 3, 3], stride=[1, 2, 2]):
    """
    Construction of deconv. layers. Input the arguments in normal conv. order.
    E.g., n_channel = [1, 32, 16, 8] gives deconv. layers of [8, 16, 32, 1].
    :param n_layer: number of deconv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 )
    :param n_freqBand: number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified deconv. layers.
    TODO:
        [] directly use nn.Conv1d for implementation
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg

    n_channel, filter_size, stride = n_channel[::-1], filter_size[::-1], stride[::-1]

    deconv_layers = []
    for i in range(n_layer - 1):
        in_channel = n_channel[i]
        out_channel = n_channel[i + 1]
        deconv_layers.append([
            nn.ConvTranspose2d(in_channel, out_channel, (1, filter_size[i]), (1, stride[i])),
            nn.BatchNorm2d(out_channel),
            nn.Tanh()
        ])
    # Construct the output layer
    deconv_layers.append([
        nn.ConvTranspose2d(n_channel[-2], n_channel[-1], (n_freqBand, filter_size[-1]), (1, stride[-1])),
        nn.Tanh()  # check the effect of with or without BatchNorm in this layer
    ])
    deconv_layers = [j for i in deconv_layers for j in i]

    return nn.Sequential(*deconv_layers)


def fc(n_layer, n_channel, activation='tanh', batchNorm=True):
    """
    Construction of fc. layers.
    :param n_layer: number of fc. layers
    :param n_channel: in/output number of neurons for each layer ( len(n_channel) = n_layer + 1 )
    :param activation: allow either 'tanh' or None for now
    :param batchNorm: True|False, indicate apply batch normalization or not
    TODO:
        [] allow different activations and batch normalization functions
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    assert activation in [None, 'tanh'], "Only implement 'tanh' for now"

    fc_layers = []
    for i in range(n_layer):
        layer = [nn.Linear(n_channel[i], n_channel[i + 1])]
        if batchNorm:
            layer.append(nn.BatchNorm1d(n_channel[i + 1]))
        if activation:
            layer.append(nn.Tanh())
        fc_layers.append(layer)

    fc_layers = [j for i in fc_layers for j in i]

    return nn.Sequential(*fc_layers)


class SpecVAE(BaseVAE):
    def __init__(self, input_size=(1, 64, 15), latent_dim=32, is_featExtract=False,
                 n_convLayer=3, n_convChannel=[1, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2],
                 n_fcLayer=1, n_fcChannel=[256]):
        """
        Construction of VAE
        :param input_size: (n_channel, n_freqBand, n_contextWin);
                           assume a spectrogram input of size (n_freqBand, n_contextWin)
        :param latent_dim: the dimension of the latent vector
        :param is_featExtract: if True, output z as mu; otherwise, output z derived from reparameterization trick
        """
        super(SpecVAE, self).__init__(input_size, latent_dim, is_featExtract)
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.is_featExtract = is_featExtract

        self.n_freqBand = input_size[1]
        self.n_contextWin = input_size[2]

        # Construct encoder and Gaussian layers
        self.encoder = spec_conv2d(n_convLayer, n_convChannel, self.n_freqBand, filter_size, stride)
        self.flat_size, self.encoder_outputSize = self.infer_flat_size()
        self.encoder_fc = fc(n_fcLayer, [self.flat_size, *n_fcChannel], activation='tanh', batchNorm=True)
        self.mu_fc = fc(1, [n_fcChannel[-1], latent_dim], activation=None, batchNorm=False)
        self.logvar_fc = fc(1, [n_fcChannel[-1], latent_dim], activation=None, batchNorm=False)

        # Construct decoder
        self.decoder_fc = fc(n_fcLayer + 1, [self.latent_dim, *n_fcChannel[::-1], self.flat_size],
                             activation='tanh', batchNorm=True)
        self.decoder = spec_deconv2d(n_convLayer, n_convChannel, self.n_freqBand, filter_size, stride)

    def infer_flat_size(self):
        encoder_output = self.encoder(torch.ones(1, *self.input_size))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]

    def encode(self, x):
        h = self.encoder(x)
        h2 = self.encoder_fc(h.view(-1, self.flat_size))
        mu = self.mu_fc(h2)
        logvar = self.logvar_fc(h2)
        mu, logvar, z = self._infer_latent(mu, logvar)

        return mu, logvar, z

    def decode(self, x):
        fc_output = self.decoder_fc(x)
        y = self.decoder(fc_output.view(-1, *self.encoder_outputSize))
        return y

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        x_recon = self.decode(z)
        # print(x_recon.size(), mu.size(), var.size(), z.size())
        return x_recon, mu, logvar, z
