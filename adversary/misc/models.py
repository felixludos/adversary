
import sys, os

import torch

from omnilearn import Model, Generative
from omnilearn import util

class GAN_Like(Generative, Model):
	
	def _process_batch(self, batch, out=None):

		if out is None:
			out = util.TensorDict()

		out.batch = batch

		if isinstance(batch, torch.Tensor):
			real = batch
		elif isinstance(batch, (tuple, list)):
			real = batch[0]
		elif isinstance(batch, dict):
			real = batch['x']
		else:
			raise NotImplementedError

		out.real = real

		return out


