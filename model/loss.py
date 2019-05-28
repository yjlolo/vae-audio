import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def kld_loss(q_mu, q_logvar):
    """
    KL divergence between q(z | x) and p(z)
    in standard VAEs, the prior p(z) is a standard Gaussian.
    :params q_mu: posterior mean
    :params q_logvar: posterior log-variance
    """
    # if mu is None:
    #     mu = torch.zeros_like(q_mu)
    # if logvar is None:
    #     logvar = torch.zeros_like(q_logvar)

    # return -0.5 * (1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar))

    return -0.5 * torch.mean(1 + q_logvar - (torch.pow(q_mu, 2) + torch.exp(q_logvar)))


def mse_loss(output, target):
    """
    Reconstruction loss
    """
    return F.mse_loss(output, target, reduction='elementwise_mean')


def vae_loss(q_mu, q_logvar, output, target):
    return mse_loss(output, target), kld_loss(q_mu, q_logvar)
