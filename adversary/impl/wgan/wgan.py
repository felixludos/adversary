import sys, os

import numpy as np
import torch
from torch import nn

import omnifig as fig

import foundation as fd
from foundation import util

from ...misc import GAN_Like

@fig.Component('wgan-gen')
class Generator(fd.Optimizable):
	def __init__(self, A, **kwargs):
		
		latent_dim = A.pull('latent_dim', 100)
		dout = A.pull('dout')
		
		super().__init__(A, din=latent_dim, dout=dout, **kwargs)

		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_feat, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(latent_dim, 128, normalize=False),
			*block(128, 256),
			*block(256, 512),
			*block(512, 1024),
			nn.Linear(1024, int(np.prod(dout))),
			nn.Sigmoid()
		)

	def forward(self, z):
		img = self.model(z)
		img = img.view(img.shape[0], *self.dout)
		return img


@fig.Component('wgan-disc')
class Discriminator(fd.Optimizable):
	def __init__(self, A, din=None, dout=None, **kwargs):
		
		if din is None:
			din = A.pull('din')
		if dout is None:
			dout = A.push('dout', 1, overwrite=False)
	
		super().__init__(A, din=din, dout=dout, **kwargs)

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(din)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
		)

	def forward(self, img):
		img_flat = img.view(img.shape[0], -1)
		validity = self.model(img_flat)
		return validity



@fig.Component('wgan')
class WGAN(fd.Decodable, GAN_Like):
	def __init__(self, config, generator=None, discriminator=None, **other):
		
		if generator is None:
			generator = config.pull('generator')
		if discriminator is None:
			discriminator = config.pull('discriminator')
		
		viz_gen = config.pull('viz-gen', True)
		viz_disc = config.pull('viz-disc', True)
		viz_samples = config.pull('viz-samples', True)
		
		retain_graph = config.pull('retain-graph', False)
		
		super().__init__(config, din=generator.din, dout=generator.dout, **other)
		
		self.generator = generator
		self.discriminator = discriminator
		
		self.latent_dim = self.generator.din
		self._viz_settings = set()
		if viz_samples:
			self._viz_settings.add('samples')
		if viz_gen:
			self._viz_settings.add('gen')
		if viz_disc:
			self._viz_settings.add('disc')
		self.retain_graph = retain_graph
		
		self.register_stats('disc-real', 'disc-fake', 'gen-loss')
		
		self.register_attr('total_gen_steps', 0)
		self.register_attr('total_disc_steps', 0)
	
	def sample_prior(self, N=1):
		return torch.randn(N, self.latent_dim).to(self.device)
	
	def decode(self, q):
		return self.generator(q)
	
	def generate(self, N=1, prior=None):
		if prior is None:
			prior = self.sample_prior(N)
		return self.decode(prior)
	
	
	def _visualize(self, info, records):
		
		settings = self._viz_settings
		if 'gen' in settings and isinstance(self.generator, fd.Visualizable):
			self.generator.visualize(info, records)
		if 'disc' in settings and isinstance(self.discriminator, fd.Visualizable):
			self.discriminator.visualize(info, records)
		
		if 'samples' in settings:
			N = 16
			
			real = info.real[:N // 2]
			records.log('images', 'real-img', util.image_size_limiter(real))
			
			gen = info.gen[:N]
			records.log('images', 'gen-img', util.image_size_limiter(gen))
	
	def _step(self, batch, out=None):
		
		out = self._process_batch(batch, out)
		
		if self.train_me():
			self.optim.discriminator.zero_grad()
		
		self._disc_step(out)
		
		if self._take_gen_step():
			if self.train_me():
				self.optim.generator.zero_grad()
			self._gen_step(out)
		
		del out.batch
		
		return out
	
	def _take_gen_step(self):
		return True  # by default always take gen step
	
	def _disc_step(self, out):
		
		real = out.real
		
		if 'fake' not in out:
			out.fake = self.generate(real.size(0))
		fake = out.fake.detach()
		
		self.volatile.real = real
		self.volatile.fake = fake
		
		verdict_real = self.discriminator(real)
		verdict_fake = self.discriminator(fake)
		
		self.mete('disc-real', verdict_real.mean())
		self.mete('disc-fake', verdict_fake.mean())
		
		out.vreal = verdict_real
		out.vfake = verdict_fake
		
		disc_loss = self._disc_loss(out)
		out.disc_loss = disc_loss
		
		if self.train_me():
			# self.optim.discriminator.zero_grad()
			disc_loss.backward(retain_graph=self.retain_graph)
			self.optim.discriminator.step()
			self.total_disc_steps += 1
	
	def _disc_loss(self, out):
		vreal = out.vreal
		vfake = out.vfake
		
		diff = self._verdict_metric(vfake, vreal)
		out.loss = diff
		
		return diff  # discriminator should maximize the difference
	
	def _gen_step(self, out):
		
		if 'gen' not in out:
			if 'prior' not in out:
				out.prior = self.sample_prior(out.real.size(0))
			gen = self.generate(prior=out.prior)
			out.gen = gen
		
		gen = out.gen
		
		vgen = self.discriminator(gen)
		out.vgen = vgen
		
		gen_loss = self._gen_loss(out)
		out.gen_loss = gen_loss
		
		if self.train_me():
			# self.optim.generator.zero_grad()
			gen_loss.backward(retain_graph=self.retain_graph)
			self.optim.generator.step()
			self.total_gen_steps += 1
	
	def _gen_loss(self, out):
		
		gen_score = self._verdict_metric(out.vgen)
		out.gen_raw_loss = gen_score

		self.mete('gen-loss', gen_score)
		
		return gen_score
	
	def _verdict_metric(self, vfake, vreal=None):
		if vreal is None:
			return -vfake.mean()  # wasserstein metric
		return vfake.mean() - vreal.mean()



@fig.AutoModifier('clamp-disc')
class Clamped(WGAN):
	
	def __init__(self, A, **kwargs):
		
		clip_value = A.pull('clip-value', None)
		
		if clip_value is None:
			print('WARNING: not using the clamped-disc')
		
		super().__init__(A, **kwargs)
		
		self.clip_value = clip_value
		self.register_hparam('clip_value', clip_value)

	def _disc_step(self, out):
		super()._disc_step(out)
		
		for p in self.discriminator.parameters():
			p.data.clamp_(-self.clip_value, self.clip_value)


@fig.AutoModifier('skip-gen')
class Skip(WGAN):
	def __init__(self, A, **kwargs):

		disc_steps = A.pull('disc-steps', None)

		if disc_steps is None:
			print('WARNING: not using the skip-gen')

		super().__init__(A, **kwargs)

		self.disc_step_interval = disc_steps
		self.register_hparam('disc_step_interval', disc_steps)

	def _take_gen_step(self):
		return self.disc_step_interval is None \
			   or self.total_disc_steps % self.disc_step_interval == 0



