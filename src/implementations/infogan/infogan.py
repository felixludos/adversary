


import torch
from torch import nn
from torch.nn import functional as F

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util

from ..wgan import WGAN
from ... import misc


@fig.AutoModifier('info')
class Info(WGAN): # Info-GANs - recover the original samples

	def __init__(self, config):

		super().__init__(config)

		self.rec_wt = config.pull('rec-wt', None)

		if self.rec_wt is not None and self.rec_wt > 0:
			self.rec_criterion = util.get_loss_type(config.pull('rec-criterion', 'mse'))

			if 'rec' in config:
				config.push('rec.din', self.generator.dout)
				config.push('rec.dout', self.generator.din)
			self.rec = config.pull('rec', None)

			self.stats.new('info-loss')

		else:
			self.rec = None

	def _gen_loss(self, out):

		loss = super()._gen_loss(out)

		if self.rec is None:
			return loss

		x = out.gen
		y = out.prior

		if self.train_me():
			self.optim.rec.zero_grad()

		rec = self.rec(x)
		out.rec = rec

		rec_loss = self.rec_criterion(rec, y)

		self.stats.update('info-loss', rec_loss)

		return loss + self.rec_wt * rec_loss

	def _gen_step(self, out):
		super()._gen_step(out)

		if self.train_me():
			self.optim.rec.step()





