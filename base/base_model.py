import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)


class BaseVAE(BaseModel):
    def __init__(self, input_size, latent_dim, is_featExtract=False):
        super(BaseVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.is_featExtract = is_featExtract

    def infer_flat_size(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def _infer_latent(self, mu, logvar):
        if self.is_featExtract:
            """
            Infer mu as the feature vector
            """
            return mu, logvar, mu

        sigma = torch.sqrt(torch.exp(logvar))
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size())

        z = mu + sigma * eps  # reparameterization trick

        return mu, logvar, z

    def forward(self, x):
        raise NotImplementedError
