

import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util

from ..wgan import WGAN
from ..dcgan import ShannonJensen_GAN
from ... import misc


def grad_penalty(disc, real, fake):  # for wgans
	# from "Improved Training of Wasserstein GANs" by Gulrajani et al. (1704.00028)

	B = real.size(0)
	eps = torch.rand(B, *[1 for _ in range(real.ndimension() - 1)], device=real.device)

	combo = eps * real.detach() + (1 - eps) * fake.detach()
	combo.requires_grad = True
	with torch.enable_grad():
		grad = autograd.grad(disc(combo).mean(), combo,
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	return (grad.contiguous().view(B, -1).norm(2, dim=1) - 1).pow(2).mean()


def grad_penalty_sj(disc, real, fake):  # for shannon jensen gans
	# from "Stabilizing Training of GANs through Regularization" by Roth et al. (1705.09367)

	B = real.size(0)

	fake, real = fake.clone().detach(), real.clone().detach()
	fake.requires_grad, real.requires_grad = True, True

	with torch.enable_grad():
		vfake, vreal = disc(fake), disc(real)
		gfake, greal = autograd.grad(vfake.mean() + vreal.mean(),
									 (fake, real),
									 create_graph=True, retain_graph=True, only_inputs=True)

	nfake = gfake.view(B, -1).pow(2).sum(-1, keepdim=True)
	nreal = greal.view(B, -1).pow(2).sum(-1, keepdim=True)

	return (vfake.sigmoid().pow(2) * nfake).mean() + ((-vreal).sigmoid().pow(2) * nreal).mean()



@fig.AutoModifier('grad-penalty')
class GradPenalty(WGAN):
	def __init__(self, config, **kwargs):

		gp_wt = config.pull('gp-wt', None)

		if gp_wt is None:
			print('WARNING: not using the grad-penalty')

		super().__init__(config, **kwargs)

		self.gp_wt = gp_wt

		self.gp_fn = grad_penalty_sj if isinstance(self, ShannonJensen_GAN) else grad_penalty

		self.stats.new('grad-penalty')

	def _grad_penalty(self, out):
		return self.gp_fn(self.discriminator, out.real, out.fake)

	def _disc_loss(self, out):
		loss = super()._disc_loss(out)

		if self.gp_wt is not None and self.gp_wt > 0:
			grad_penalty = self._grad_penalty(out)
			self.stats.update('grad-penalty', grad_penalty)
			loss += self.gp_wt * grad_penalty

		return loss


