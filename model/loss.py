import numpy as np
import torch
import torch.nn.functional as F
from base import approx_qy_x


def vae_loss(q_mu, q_logvar, output, target):
    return mse_loss(output, target), kld_gauss(q_mu, q_logvar)


def gmvae_loss(output, target, logLogit_qy_x, qy_x, q_mu, q_logvar, mu_lookup, logvar_lookup, n_component):
    """
    Basic GMVAE loss (https://arxiv.org/abs/1611.05148)
    """
    return mse_loss(output, target),\
        kld_latent(qy_x, q_mu, q_logvar, mu_lookup, logvar_lookup),\
        kld_class(logLogit_qy_x, qy_x, n_component)


def mse_loss(output, target, avg_batch=True):
    """
    Reconstruction loss
    To prevent posterior collapse of q(y|x) in GMVAE, there is no normalization performed w.r.t.
    number of frequency bins and time frames; which makes scale of reconstruction loss relatively large
    compared to KL terms.
    TODO:
        [] Allow optional normalization w.r.t. frequency and time axis
        [] Find a good normalization scheme w.r.t frequency and time axis
    """
    output = F.mse_loss(output, target, reduction='none')
    output = torch.sum(output)  # sum over all TF units
    if avg_batch:
        output = torch.mean(output, dim=0)
    # return F.mse_loss(output, target, reduction=reduce)  # careful about the scaling
    return output


def kld_gauss(q_mu, q_logvar, mu=None, logvar=None, avg_batch=True):
    """
    KL divergence between two diagonal Gaussians
    in standard VAEs, the prior p(z) is a standard Gaussian.
    :param q_mu: posterior mean
    :param q_logvar: posterior log-variance
    :param mu: prior mean
    :param logvar: prior log-variance
    """
    # set prior to a standard Gaussian
    if mu is None:
        mu = torch.zeros_like(q_mu)
    if logvar is None:
        logvar = torch.zeros_like(q_logvar)

    output = torch.sum(1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar),
                        dim=1)
    output *= -0.5
    if avg_batch:
        output = torch.mean(output, dim=0)
    return output


def kld_class(logLogit_qy_x, qy_x, n_component, avg_batch=True):
    h_qy_x = torch.sum(qy_x * torch.nn.functional.log_softmax(logLogit_qy_x, dim=1), dim=1)
    output = h_qy_x - np.log(1 / n_component)
    if avg_batch:
        output = torch.mean(output, dim=0)
    # return h_qy_x - np.log(1 / n_component)  # , h_qy_x
    return output


def kld_latent(qy_x, q_mu, q_logvar, mu_lookup, logvar_lookup, avg_batch=True):
    """
    Calculate the term of KLD in the ELBO of GMVAEs:
    sum_{y}{ q(y|x) * KLD[ q(z|x) | p(z|y) ] }
    :param qy_x: q(y|x)
    :param q_mu: approximated posterior mean
    :param q_logvar: approximated posterior log-variance
    :param mu_lookup: conditional prior mean
    :param logvar_lookup: conditional prior log-variance
    """
    batch_size, n_component = list(qy_x.size())
    kl_sumOver = torch.zeros(batch_size, n_component)
    for k_i in torch.arange(0, n_component):
        # KLD
        kl_sumOver[:, k_i] = kld_gauss(q_mu, q_logvar, mu_lookup(k_i), logvar_lookup(k_i), avg_batch=False)
        # weighted sum by q(y|x)
        kl_sumOver[:, k_i] *= qy_x[:, k_i]
    # sum over components
    output = torch.sum(kl_sumOver, dim=1)
    if avg_batch:
        output = torch.mean(output, dim=0)
    return output
