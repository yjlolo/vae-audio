import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class SpecVaeTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(SpecVaeTrainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _reshape(self, x):
        n_freqBand, n_contextWin = x.size(2), x.size(3)
        return x.view(-1, 1, n_freqBand, n_contextWin)

    def _forward_and_computeLoss(self, x, target):
        x_recon, mu, logvar, z = self.model(x)
        loss_recon, loss_kl = self.loss(mu, logvar, x_recon, target)
        loss = loss_recon + loss_kl
        return loss, loss_recon, loss_kl

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        # total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data_idx, label, data) in enumerate(self.data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            x = self._reshape(x)

            self.optimizer.zero_grad()
            loss, loss_recon, loss_kl = self._forward_and_computeLoss(x, x)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item()
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # TODO: visualize input/reconstructed spectrograms in TensorBoard
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'loss_recon': total_recon / len(self.data_loader),
            'loss_kl': total_kl / len(self.data_loader)
            # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_recon = 0
        total_val_kl = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data_idx, label, data) in enumerate(self.valid_data_loader):
                x = data.type('torch.FloatTensor').to(self.device)
                x = self._reshape(x)

                loss, loss_recon, loss_kl = self._forward_and_computeLoss(x, x)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_recon += loss_recon.item()
                total_val_kl += loss_kl.item()
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_loss_recon': total_val_recon / len(self.valid_data_loader),
            'val_loss_kl': total_val_kl / len(self.valid_data_loader)
            # 'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }


class GMVAETrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(GMVAETrainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _reshape(self, x):
        # assume dimensions to be [batch_size, n_freqBand, n_contextWin]
        return x.unsqueeze(2)

    def _forward_and_computeLoss(self, x, target):
        x_recon, q_mu, q_logvar, z, logLogit_qy_x, qy_x, y = self.model(x)
        neg_logpx_z, kld_latent, kld_class = self.loss(x_recon, target, logLogit_qy_x, qy_x, q_mu, q_logvar,
                                                        self.model.mu_lookup, self.model.logvar_lookup,
                                                        self.model.n_component)
        loss = neg_logpx_z + kld_latent + kld_class
        return loss, neg_logpx_z, kld_latent, kld_class

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kld_latent = 0
        total_kld_class = 0
        # total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data_idx, label, data) in enumerate(self.data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            # x = self._reshape(x)

            self.optimizer.zero_grad()
            loss, neg_logpx_z, kld_latent, kld_class = self._forward_and_computeLoss(x, x)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_recon += neg_logpx_z.item()
            total_kld_latent += kld_latent.item()
            total_kld_class += kld_class.item()
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # TODO: visualize input/reconstructed spectrograms in TensorBoard
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'loss_recon': total_recon / len(self.data_loader),
            'loss_kld_latent': total_kld_latent / len(self.data_loader),
            'loss_kld_class': total_kld_class / len(self.data_loader)
            # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kld_latent = 0
        total_kld_class = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data_idx, label, data) in enumerate(self.valid_data_loader):
                x = data.type('torch.FloatTensor').to(self.device)
                # x = self._reshape(x)

                loss, neg_logpx_z, kld_latent, kld_class = self._forward_and_computeLoss(x, x)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_loss += loss.item()
                total_recon += neg_logpx_z.item()
                total_kld_latent += kld_latent.item()
                total_kld_class += kld_class.item()
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_loss / len(self.valid_data_loader),
            'val_loss_recon': total_recon / len(self.valid_data_loader),
            'val_loss_kld_latent': total_kld_latent / len(self.valid_data_loader),
            'val_loss_kld_class': total_kld_class / len(self.valid_data_loader)
            # 'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
