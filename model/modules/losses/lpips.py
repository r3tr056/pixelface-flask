
import torch
from torch import nn

from torchvision import models
from collections import namedtuple

class LPIPS(nn.Module):

	""" Learned perceptual metric """

	def __init__(self, use_dropout=True, style_weight=0.0):
		super().__init__()

		# Scaling layer for normalizing the input images
		self.scaling_layer = ScalingLayer()
		# vgg16 features
		self.chns = [64, 128, 256, 512, 512]
		# VGG16 network with pretrained weights, applied to extract features at
		# different layers
		self.net = vgg16(pretrained=True, requires_grad=False)

		# Linear layers for comparing VGG features (learned linerar layers)
		# they are applied to features extracted from the VGG16 at different
		# layers
		self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
		self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
		self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
		self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
		self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

		self.load_from_pretrained()

		for param in self.parameters():
			param.requires_grad = False

		self.style_weight = style_weight

	def load_from_pretrained(self, name='vgg_lpips'):
		ckpt = get_ckpt_path(name, "experiments/pretrained_models/lpips")
		self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
		print("Loaded pretrained LPIPS loss from {}".format(ckpt))

	@classmethod
	def from_pretrained(cls, name="vgg_lpips"):
		if name is not "vgg_lpips":
			raise NotImplementedError()

		model = cls()
		ckpt = get_ckpt_path(name)
		model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
		return model

	def forward(self, input, traget):
		in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(traget))
		outs0, outs1 = self.net(in0_input), self.net(in1_input)
		feats0, feats1, diffs = {}, {}, {}
		lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
		style_loss = torch.tensor([0.0]).to(input.device)
		for kk in range(len(self.chns)):
			feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
			diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

			if self.style_weight > 0.:
				style_loss = style_loss + torch.mean((self._gram_mat(feats0[kk]) - self._gram_mat(feats1[kk])) ** 2)

		res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
		val = res[0]
		for l in range(1, len(self.chns)):
			val += res[l]

		return val, style_loss * self.style_weight

	def _gram_mat(self, x):
		""" Calculate gram matrix """
		n, c, h, w = x.size()
		features = x.view(n, c, w * h)
		features_t = features.transpose(1, 2)
		gram = features.bmm(features_t) / (c * h * w)
		return gram

class ScalingLayer(nn.Module):
	def __init__(self):
		super().__init__()
		self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
		self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

	def forward(self, inp):
		return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
	def __init__(self, chn_in, chn_out=1, use_dropout=False):
		super().__init__()
		layers = [nn.Dropout()] if (use_dropout) else []
		layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
		self.model = nn.Sequential(*layers)

class vgg16(nn.Module):
	def __init__(self, requires_grad:bool=False, pretrained:bool=True):
		super(vgg16, self).__init__()

		vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()

		self.N_slices = 5

		for x in range(4):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(4, 9):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(9, 16):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(16, 23):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(23, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.slice1(x)
		h_relu1_2 = h
		h = self.slice2(h)
		h_relu2_2 = h
		h = self.slice3(h)
		h_relu3_3 = h
		h = self.slice4(h)
		h_relu4_3 = h
		h = self.slice5(h)
		h_relu5_3 = h

		vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
		out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
		return out

	def normalize_tensor(x, eps:float=1e-10):
		norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
		return x / (norm_factor + eps)
		

	def spatial_average(x, keepdim: bool = True):
		return x.mean([2, 3], keepdim=keepdim)
