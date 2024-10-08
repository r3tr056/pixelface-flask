import collections
from typing import Callable, List, Tuple

import torch
from torch import nn

from super_res.model import utils
from super_res.model import modules

class SRResNetEncoder(nn.Module):
	""" Encoder' part of SRResNet network, processing images in LR Space

	It has been proposed in `Photo-Realistic Single Image Super-Resoultion
	Using Generative Adversial Network`

	Args:
		in_channels: Number of channels in the input image
		out_channels: Number of channels produced by the encoder
		num_basic_blocks: Depth of the encoder, number of basic blocks to use
		conv: Class constructor for the Convolutional Layer (nn.Conv2d)
		norm: Class constructor for the Normalization Layer (nn.BatchNorm2d)
		activation: Class constructor for the activation function to use after the BN (nn.RReLU)

	"""

	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 64,
		num_basic_blocks: int = 16,
		conv: Callable[..., nn.Module] = modules.Conv2d,
		norm: Callable[..., nn.Module] = nn.BatchNorm2d,
		activation: Callable[..., nn.Module] = nn.PReLU
	) -> None:
		super().__init__()

		num_features = out_channels
		blocks_list: List[nn.Module] = []

		first_conv = nn.Sequential(
			conv(in_channels, num_features), activation()
		)
		blocks_list.append(first_conv)

		# basic blocks - sequence of B residual blocks
		for _ in range(num_basic_blocks):
			basic_block = nn.Sequential(
				conv(num_features, num_features),
				norm(num_features, ),
				activation(),
				conv(num_features, num_features),
				norm(num_features)
			)
			blocks_list.append(modules.ResidualModule(basic_block))

		# last conv of the encoder
		last_conv = nn.Sequential(
			conv(num_features, out_channels), norm(out_channels),
		)
		blocks_list.append(last_conv)

		self.blocks = nn.ModuleList(blocks_list)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		""" Forward pass """
		input_ = output = self.blocks[0](x)
		for module in self.blocks[1:]:
			output = module(output)

		return input_ + output


class SRResNetDecoder(nn.Module):
	"""
	Decoder part of SRResNet, converting embeddings to output image
	
	Args:
		in_channels: Number of channels in the input embedding
		out_channels: Number of channels in the output image
		scale_factor: Ratio between the size of the high-resolution image (output)
		and its LR counterpart (input). Multiplier of the spatial size
		conv: Class constructor for the convolution layer (nn.Conv2d)
		activation: Class constructor for the activation function (nn.ReLU)
	"""

	def __init__(
		self,
		in_channels: int = 64,
		out_channels: int = 3,
		scale_factor: int = 2,
		conv: Callable[..., nn.Module] = modules.Conv2d,
		activation: Callable[..., nn.Module] = nn.PReLU,
	) -> None:
		super().__init__()

		# check params
		if utils.is_power_of_two(scale_factor):
			raise NotImplementedError(f"scale_factor should be power of 2, got {scale_factor}")

		blocks_list: List[Tuple[str, nn.Module]] = []

		# upsampling
		for i in range(scale_factor // 2):
			upsampling_block = modules.SubPixelConv(
				num_features=in_channels,
				conv=conv,
				activation=activation
			)
			blocks_list.append((f"upsampling_{i}", upsampling_block))

		# highres conv
		last_conv = conv(in_channels, out_channels)
		blocks_list.append(("conv", last_conv))

		self.blocks = nn.Sequential(collections.OrderedDict(blocks_list))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		""" Forward pass """
		output = self.blocks(x)

		return output