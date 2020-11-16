
import sys, os

import torch

from foundation import Full_Model
from foundation import util

class GAN_Like(Full_Model):
	
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


