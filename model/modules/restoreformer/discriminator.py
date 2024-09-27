
import functools

import torch
from torch import nn

from super_res.module.modules.norms import ActNorm


class NLayerDiscriminator(nn.Module):
	""" Defines a PatchGAN discriminator as in Pix2Pix """

	def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
		super(NLayerDiscriminator, self).__init__()

		if not use_actnorm:
			norm_layer = nn.BatchNorm2d
		else:
			norm_layer = ActNorm

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func != nn.BatchNorm2d
		else:
			use_bias = norm_layer != nn.BatchNorm2d

		self.n_layers = n_layers
		kw = 4
		padw = 1
		self.head = nn.Sequential(
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		)

		nf_mult = 1
		nf_mult_prev = 1
		self.body = nn.ModuleList()

		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			self.body.append(nn.Sequential(
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			))

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		self.before_last = nn.Sequential(
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		)

		self.final = nn.Sequential(
			nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
		)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		""" Forward pass """
		features = []

		f = self.head(x)
		features.append(f)

		for i in range(self.n_layers - 1):
			f = self.body[i](f)
			features.append(f)

		beforelastF = self.before_last(f)
		final_logits = self.final(beforelastF)

		return features, final_logits
