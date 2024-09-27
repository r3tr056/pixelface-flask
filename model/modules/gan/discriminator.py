

import collections
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from super_res import utils
from super_res.model.modules.residual_wrapper import ResidualWrapper

class StridedConvEncoder(nn.Module):
	""" Generalized Fully Convolution encoder 
	
	Args:
		layers : List of feature maps sized of each block
		layer_order: Ordered list of layers applied withing each block
		conv: Class constructor or partial object which when called should return convolution layer
		norm: Class construcor or partial object which when called should return normalization layer
		activation : Class constructor for activation
		residual: Class constructor for residual wrapper
	"""

	def __init__(
		self,
		layers: Iterable[int] = (3, 64, 128, 256, 512, 512),
		layer_order: Iterable[str] = ("conv", "norm", "activation"),
		conv: Callable[..., nn.Module] = modules.Conv2d,
		norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
		activation: Callable[..., nn.Module] = modules.LeakyReLU,
		residual: Optional[Callable[..., nn.Module]] = None,
	):
		super().__init__()

		name2fn: Dict[str, Callable[..., nn.Module]] = {
			"activation": activation,
			"conv": conv,
			"norm": norm,
		}

		self._layers = list(layers)
		net: List[Tuple[str, nn.Module]] = []

		first_conv = collections.OrderedDict([
			("conv_0", name2fn["conv"](self._layers[0], self._layers[1])),
			("act", name2fn["activation"]()),
		])

		net.append(("block_0", nn.Sequential(first_conv)))

		channels = utils.pairwise(self._layers[1:])
		for i, (in_ch, out_ch) in enumerate(channels, start=1):
			block_list: List[Tuple[str, nn.Module]] = []
			for name in layer_order:
				kwargs = {"stride": out_ch // in_ch} if name == "conv" else {}

				module = utils.create_layer(
					layer_name=name,
					layer=name2fn[name],
					in_channels=in_ch,
					out_channels=out_ch,
					**kwargs,
				)
				block_list.append((name, module))

			block = nn.Sequential(collections.OrderedDict(block_list))

			if residual is not None and in_ch == out_ch:
				block = residual(block)

			net.append((f"block_{i}", block))

		self.net = nn.Sequential(collections.OrderedDict(net))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		""" Forward pass """

		output = self.net(x)
		return output

	@property
	def in_channels(self) -> int:
		return self._layers[0]

	@property
	def out_channels(self) -> int:
		return self._layers[-1]
	
	
class LinerarHead(nn.Module):
	""" Stack of linear layers used for embeddings classification
	
	Args:
		in_channels: Size of each input sample
		out_channels: Size of each output channels
		latent_channels: Size of the latent space
		layer_order: Ordered list of layers applied within each block.
		For instance, if you don't want to use activation function
		just exclude it from this list
		linear: Class constructor or partial object which when called
		should return linear layer e.g., `nn.Linear`
		activation: Class constructor of a activation function layer
		norm: Class constructor of a normalization layer
		dropout: Class constructor of a dropout layer
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		latent_channels: Optional[Iterable[int]] = None,
		layer_order: Iterable[str] = ("linear", "activation"),
		linear: Callable[..., nn.Module] = nn.Linear,
		activation: Callable[..., nn.Module] = modules.LeakyReLU,
		norm: Optional[Callable[..., nn.Module]] = None,
		dropout: Optional[Callable[..., nn.Module]] = None,
	) -> None:
		super().__init__()

		name2fn: Dict[str, Callable[..., nn.Module]] = {
			"activation": activation,
			"dropout": dropout,
			"linear": linear,
			"norm": norm,
		}

		latent_channels = latent_channels or []
		channels = [in_channels, *latent_channels, out_channels]
		channels_pairs: List[Tuple[int, int]] = list(utils.pairwise(channels))

		net: List[nn.Module] = []
		for in_ch, out_ch in channels_pairs[:-1]:
			for name in layer_order:
				module = utils.create_layer(
					layer_name=name,
					layer=name2fn[name],
					in_channels=in_ch,
					out_channels=out_ch,
				)
				net.append(module)
		net.append(name2fn["linear"](*channels_pairs[-1]))

		self.net = nn.Sequential(*net)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		""" Forward pass

		Args:
			x - Batch of inputs, e.g. Images

		Return:
			Batch of logits
		"""
		output = self.net(x)
		return output

class VGGConv(nn.Module):
	"""
	A VGG like neural network for image classification
	Args:
		encoder: Image encoder module, usually used for the extraction
			of embeddings from input signals
		pool: Pooling Layer, used to reduce the embeddings from the encoder
		head: Classification head, usually consists of a FCNN
	"""
	def __init__(
		self,
		ender: nn.Module,
		pool: nn.Module,
		head: nn.Module
	) -> None:
		super().__init__()

		self.encoder = encoder
		self.pool = pool
		self.head = head

		utils.net_init_(self)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.pool(self.encoder(x))
		x = x.view(x.shape[0], -1)
		x = self.head(x)

		return x