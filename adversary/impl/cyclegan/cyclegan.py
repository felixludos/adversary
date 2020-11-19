

import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util

from .models import weights_init_normal


@fig.Component('cycle-gan')
class CycleGAN(fd.Full_Model):
	
	def __init__(self, A):
	
		din, dout = A.pull('din', silent=True), A.pull('dout', silent=True)
		
		A.push('gen-AB.din', din, silent=True)
		A.push('gen-AB.dout', dout, silent=True)
		gen_AB = A.pull('gen-AB')

		A.push('disc-B.din', dout, silent=True)
		A.push('disc-B.dout', 1, silent=True)
		disc_B = A.pull('disc-B')
		

		A.push('gen-BA.din', dout, silent=True)
		A.push('gen-BA.dout', din, silent=True)
		gen_BA = A.pull('gen-BA')

		A.push('disc-A.din', din, silent=True)
		A.push('disc-A.dout', 1, silent=True)
		disc_A = A.pull('disc-A')
		
		if A.pull('init-normal', True):
			gen_AB.apply(weights_init_normal)
			gen_BA.apply(weights_init_normal)
			disc_A.apply(weights_init_normal)
			disc_B.apply(weights_init_normal)
			
		
		criterion_GAN = util.get_loss_type(A.pull('gan-criterion', 'mse'))
		
		cycle_wt = A.pull('cycle-wt', None)
		
		ident_wt = A.pull('ident-wt', None)
		if din != dout and ident_wt is not None and ident_wt > 0:
			ident_wt = None
			print('WARNING: cant used ident-loss since din and dout are different')
		
		buffer_A = A.pull('buffer-A', None)
		buffer_B = A.pull('buffer-B', None)
		
		super().__init__(din, dout)
		
		if cycle_wt is None:
			print('WARNING: not using cycle loss')
			criterion_cycle = None
		else:
			criterion_cycle = util.get_loss_type(A.pull('cycle-criterion', 'l1'))
			self.stats.new('loss-cycle')
		
		if ident_wt is None:
			print('WARNING: not using identity loss')
			criterion_identity = None
		else:
			criterion_identity = util.get_loss_type(A.pull('ident-criterion', 'l1'))
			self.stats.new('loss-ident')
		
		self.stats.new('loss-gan', 'loss-disc-A', 'loss-disc-B')
		

		self.gen_AB = gen_AB
		self.disc_B = disc_B
		self.gen_BA = gen_BA
		self.disc_A = disc_A
		
		self.buffer_A = buffer_A
		self.buffer_B = buffer_B
		
		self.ident_wt = ident_wt
		self.cycle_wt = cycle_wt
		
		self.criterion_GAN = criterion_GAN
		self.criterion_cycle = criterion_cycle
		self.criterion_identity = criterion_identity
		
	def _visualize(self, info, logger):
		
		N = 8
		
		# real = torch.cat([info.real_A[:N], info.real_B[:N]])
		# logger.add('images', 'real-img', util.image_size_limiter(real))

		real_A = info.real_A[:N]
		logger.add('images', 'real-img-A', util.image_size_limiter(real_A))

		real_B = info.real_B[:N]
		logger.add('images', 'real-img-B', util.image_size_limiter(real_B))
		
		gen_A = info.fake_A[:N * 2]
		logger.add('images', 'gen-img-A', util.image_size_limiter(gen_A))
		
		gen_B = info.fake_B[:N * 2]
		logger.add('images', 'gen-img-B', util.image_size_limiter(gen_B))
		
		rec_A = info.recov_A[:N * 2]
		logger.add('images', 'rec-img-A', util.image_size_limiter(rec_A))
		
		rec_B = info.recov_B[:N * 2]
		logger.add('images', 'rec-img-B', util.image_size_limiter(rec_B))
	
	def _process_batch(self, batch, out=None):
		
		if out is None:
			out = util.TensorDict()
		
		out.batch = batch
		
		if isinstance(batch, (tuple, list)):
			A, B = batch[0], batch[1]
		elif isinstance(batch, dict):
			A, B = batch['A'], batch['B']
		else:
			raise NotImplementedError
		
		out.real_A = A
		out.real_B = B
		
		return out
	
		
	def _step(self, batch, out=None):
		
		out = self._process_batch(batch, out)
		
		real_A = out.real_A
		out.valid = torch.ones(real_A.size(0), *self.disc_A.dout, device=real_A.device)
		out.invalid = torch.zeros(real_A.size(0), *self.disc_A.dout, device=real_A.device)
		
		if self.train_me():
			self.optim.gen_AB.zero_grad()
			self.optim.gen_BA.zero_grad()
		
		gen_loss = self._gen_step(out)

		if self.train_me():
			gen_loss.backward()
			self.optim.gen_AB.step()
			self.optim.gen_BA.step()
		
			self.optim.disc_A.zero_grad()
			self.optim.disc_B.zero_grad()
		
		disc_loss = self._disc_step(out)
		
		if self.train_me():
			disc_loss.backward()
			self.optim.disc_A.step()
			self.optim.disc_B.step()
		
		return out
		
	def _disc_loss(self, out, disc, real, fake, buffer=None):
		
		valid, invalid = out.valid, out.invalid
		
		# Real loss
		loss_real = self.criterion_GAN(disc(real), valid)
		
		# Fake loss (on batch of previously generated samples)
		fake = fake.detach()
		if buffer is not None:
			fake = buffer.push_and_pop(fake)
		loss_fake = self.criterion_GAN(disc(fake), invalid)
		
		# Total loss
		return (loss_real + loss_fake) / 2
		
	def _disc_step(self, out):
		
		real_A, real_B = out.real_A, out.real_B
		
		if 'fake_A' not in out:
			out.fake_A = self.gen_BA(real_B)
		loss_A = self._disc_loss(out, self.disc_A, real_A, out.fake_A, self.buffer_A)
		self.stats.update('loss-disc-A', loss_A)
		
		if 'fake_B' not in out:
			out.fake_B = self.gen_AB(real_A)
		loss_B = self._disc_loss(out, self.disc_B, real_B, out.fake_B, self.buffer_B)
		self.stats.update('loss-disc-B', loss_B)
		
		return loss_A + loss_B
		
		
	def _gen_step(self, out):
		
		loss = self._adversarial_loss(out)
		
		if self.ident_wt is not None and self.ident_wt > 0:
			loss += self.ident_wt * self._ident_loss(out)
		
		if self.cycle_wt is not None and self.cycle_wt > 0:
			loss += self.cycle_wt * self._cycle_loss(out)
	
		out.loss = loss
	
		return loss
	
	def _adversarial_loss(self, out):
		
		real_A, real_B = out.real_A, out.real_B
		valid = out.valid
		
		if 'fake_B' not in out:
			out.fake_B = self.gen_AB(real_A)
		fake_B = out.fake_B
		loss_GAN_AB = self.criterion_GAN(self.disc_B(fake_B), valid)
		
		if 'fake_A' not in out:
			out.fake_A = self.gen_BA(real_B)
		fake_A = out.fake_A
		loss_GAN_BA = self.criterion_GAN(self.disc_A(fake_A), valid)
		
		loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
		
		self.stats.update('loss-gan', loss_GAN)
		
		return loss_GAN
		
	def _ident_loss(self, out):
		
		real_A, real_B = out.real_A, out.real_B
		
		loss_id_A = self.criterion_identity(self.gen_BA(real_A), real_A)
		loss_id_B = self.criterion_identity(self.gen_AB(real_B), real_B)
		
		loss_identity = (loss_id_A + loss_id_B) / 2
		
		self.stats.update('loss-ident', loss_identity)
		
		return loss_identity

	def _cycle_loss(self, out):
		
		real_A, real_B = out.real_A, out.real_B
		
		if 'fake_B' not in out:
			out.fake_B = self.gen_AB(real_A)
		fake_B = out.fake_B
		
		if 'fake_A' not in out:
			out.fake_A = self.gen_BA(real_B)
		fake_A = out.fake_A
		
		recov_A = self.gen_BA(fake_B)
		loss_cycle_A = self.criterion_cycle(recov_A, real_A)
		recov_B = self.gen_AB(fake_A)
		loss_cycle_B = self.criterion_cycle(recov_B, real_B)
		
		out.recov_A = recov_A
		out.recov_B = recov_B
		
		loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

		self.stats.update('loss-cycle', loss_cycle)
		
		return loss_cycle



