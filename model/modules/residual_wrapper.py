import torch
from torch import nn


class ResidualWrapper(nn.Module):
	""" Residual Wrapper, adds identity connection
	It has been proposed in `Deep Residual Learning for Image Recognition`

	Args:
		module: PyTorch layer to wrap
		scale: Residual connections scaling factor
		required_grad: If set to `False` the layer will not learn the strength of the residual connection
	"""
	def __init__(self, module: nn.Module, scale: float=1.0, requires_grad: bool = False):
		self.module = module
		self.scale = nn.Parameter(
			torch.tensor(scale), requires_grad=requires_grad
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		""" Forward pass """
		return x + self.scale * self.module(x)

