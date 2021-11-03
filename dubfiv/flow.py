"""Module with glow function."""

from __future__ import annotations

import abc
import contextlib
import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import torch
from torch import nn
import torch.utils.data as torch_data
from torchvision import utils as torchvision_utils

from dubfiv import config
from dubfiv import utils


def normal_log_likelihood(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Returns the log likehood of `x` under Normal(mu, exp(log_std))."""

    return (-(mu - x)**2 / (2 * (2 * log_std).exp()) - log_std - np.log(2 * np.pi) / 2)


_log_2_np_pi = float(np.log(2 * np.pi) / 2)


Latent = Sequence[torch.Tensor]
ActivationJacobian = Tuple[torch.Tensor, Union[float, torch.Tensor]]


def standard_normal_log_likelihood(x: torch.Tensor) -> torch.Tensor:
    """Returns the log likehood of `x` under Normal(0, 1)."""
    return -x.pow(2) / 2 - _log_2_np_pi


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias,
                     padding_mode='zeros')


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = True
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class CouplingNNBlock1x1(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        n_out: Optional[int] = None,
        stride: int = 1
    ):
        super().__init__()
        n_out = n_out or planes
        self.conv1 = conv1x1(in_planes, planes, stride)
        self.relu1 = nn.ReLU()
        self.conv2 = conv1x1(planes, planes)
        self.relu2 = nn.ReLU()
        self.conv3 = conv1x1(planes, n_out)
        self.conv3.weight = nn.Parameter(0.01 * self.conv3.weight)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out


class CouplingNNBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        n_out: Optional[int] = None,
        stride: int = 1
    ):
        super().__init__()
        n_out = n_out or planes
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.actnorm1 = ActNorm(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv1x1(planes, planes)
        self.actnorm2 = ActNorm(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3(planes, n_out, bias=True)
        torch.nn.init.zeros_(self.conv3.weight)
        if self.conv3.bias is not None:
            torch.nn.init.zeros_(self.conv3.bias)
        # self.conv3.weight = nn.Parameter(0.01*self.conv3.weight)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out, _ = self.actnorm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out, _ = self.actnorm2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out


class FlowModule(nn.Module, metaclass=abc.ABCMeta):
    """A invertible flow module.

    The inverse of a FlowModule can be obtained by calling
    ``FlowModule.inverse()`` which returns an Inverse call.  Creating an extra
    nn.Module for the Inverse has the advantage that you can attach hooks also
    to the inverse computation.
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def inverse(self) -> nn.Module:
        """Returns a Inverse layer which will compute the inverse function.

        Example:
            m: FlowModule = get_my_module()
            (m.inverse()(m(x)) == x).allclose()
        """
        pass


T = TypeVar('T')


class SimpleFlowModule(FlowModule):
    """A invertible flow module.

    The inverse of a FlowModule can be obtained by calling
    ``FlowModule.inverse()`` which returns an Inverse call.  Creating an extra
    nn.Module for the Inverse has the advantage that you can attach hooks also
    to the inverse computation.
    """
    def __init__(self):
        super().__init__()

    _inverse: Callable[..., Any]

    def _inverse(self,  # type: ignore
                 *args: Any, **kwargs: Any) -> Any:
        """Computes the inverse function.

        Is called by the Inverse module.
        """
        raise NotImplementedError()

    def inverse(self) -> nn.Module:
        """Returns a Inverse layer which will compute the inverse function.

        Example:
            m: FlowModule = get_my_module()
            (m.inverse()(m(x)) == x).allclose()
        """
        return InverseWrapper(self)


class InverseWrapper(nn.Module):
    def __init__(self, module: SimpleFlowModule):
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module._inverse(*args, **kwargs)

    def inverse(self) -> nn.Module:
        return self.module


class AffineCoupling(SimpleFlowModule):
    def __init__(self, block: nn.Module, squash: str = 'sigmoid'):
        """Affine Coupling Layer.

        Args:
            block (nn.Module): layer to compute the scaling and offset components.
                See CouplingNNBlock and CouplingNNBlock1x1.
            squash (str): Squashing of the scaling parameters. Important to
                ensure numerical stability. Values: ``sigmoid``: sigmoid(s) + 0.5.
        """
        super().__init__()
        self._block = block

    def block(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._block(x)
        s, t = split(out)
        s = torch.sigmoid(s) + 0.5
        return s, t

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return split(x)

    def forward(self, x: torch.Tensor) -> ActivationJacobian:
        #                  x_a ---------|-------------> y_a ---->|
        #   x    [split]            [block(x_a)]              [concat] --> y
        #                  x_b --> [* s] -> [+ t] ----> y_b ---->|
        x_a, x_b = split(x)
        s, t = self.block(x_a)
        y_b = s * x_b + t
        y_a = x_a
        y = torch.cat([y_a, y_b], 1)
        jac = s.log().sum(1, keepdim=True)
        return y, jac

    def _inverse(self, y: torch.Tensor) -> ActivationJacobian:
        y_a, y_b = split(y)
        s, t = self.block(y_a)
        x_b = (y_b - t) / s
        x_a = y_a
        return torch.cat([x_a, x_b], 1), - s.log().sum(1, keepdim=True)


class AdditiveCoupling(AffineCoupling):
    def __init__(self, block: nn.Module):
        super().__init__(block)

    def block(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        add = self._block(x)
        scaling = torch.ones_like(add)
        return scaling, add


class Psi(SimpleFlowModule):
    # by Jörn Jacobsen
    # https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
    """Invertible pooling layer.

    Args:
        block_size (int): Block size to merge
    """

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def _inverse(self, x: torch.Tensor) -> ActivationJacobian:
        output = x.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(
            0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous(), torch.tensor(0, device=output.device)

    def forward(self, x: torch.Tensor) -> ActivationJacobian:
        output = x.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous(), torch.tensor(0, device=output.device)


class ActNorm(SimpleFlowModule):
    """ActNorm layer with ``n_channels``.

    Before training scales and offsets are estimated such that the output
    has mean = 0 and standard derivation = 1.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.register_parameter('s', nn.Parameter(torch.ones(1, n_channels, 1, 1)))
        self.register_parameter('b', nn.Parameter(torch.zeros(1, n_channels, 1, 1)))
        self.data_initialization = False

    def forward(self, x: torch.Tensor) -> ActivationJacobian:
        if self.data_initialization:
            b, c, h, w = x.shape
            x_c = x.permute(1, 0, 2, 3).contiguous().view(c, b * h * w)
            mean = x_c.mean(1).view(1, c, 1, 1)
            std = x_c.std(1).view(1, c, 1, 1)
            self.s = nn.Parameter(1 / std)
            self.b = nn.Parameter(-mean / std)
        b, c, h, w = x.shape
        jaccobian = h * w * self.s.abs().log().sum()
        return self.s * x + self.b, jaccobian

    def _inverse(self, y: torch.Tensor) -> ActivationJacobian:
        h, w = y.shape[2:]
        jaccobian = h * w * self.s.abs().log().sum()
        return (y - self.b) / self.s, - jaccobian


class data_init(torch.no_grad):
    """Context Manager to enable data dependent initialization for ActNorm Layers.

    Args:
        modules: will enable to data dependent initialization for these and all submodules.
    """
    def __init__(self, modules: Sequence[nn.Module]):
        super().__init__()
        self.modules = modules

    def __enter__(self):
        def enable_data_init(module: nn.Module):
            if isinstance(module, ActNorm):
                module.data_initialization = True

        super().__enter__()
        for module in self.modules:
            module.apply(enable_data_init)

    def __exit__(self, *args: Any):
        super().__exit__(*args)

        def disable_data_init(module: nn.Module):
            if isinstance(module, ActNorm):
                module.data_initialization = False

        for module in self.modules:
            module.apply(disable_data_init)


class Conv1x1Inv(SimpleFlowModule):
    """Invertible 1x1 Convolution using PLU decomposition.

    See Glow paper.

    Args:
        n_channels (int): number of channels
        init (str): initialization method. Options: orthonormal (random orthogonal weights),
            zero (fill weight with zeros).
    """
    def __init__(self,
                 n_channels: int,
                 init: str = 'orthonormal',
                 weight: Optional[np.ndarray] = None,
                 bias: Union[utils.AnyTensor, bool] = False):
        super().__init__()
        if weight is None:
            if init == 'orthonormal':
                torch_weight = torch.zeros(n_channels, n_channels)
                torch.nn.init.orthogonal_(torch_weight)
                weight = torch_weight.numpy().astype(np.float32)
            elif init == 'zero':
                weight = np.zeros((n_channels, n_channels))
                weight = weight.astype(np.float32)
            elif init == 'uniform':
                weight = np.random.uniform(size=(n_channels, n_channels))
                weight = weight.astype(np.float32)
            else:
                raise Exception()

        # see glow paper
        p, l_w_diag, u = scipy.linalg.lu(weight)
        l = l_w_diag - np.eye(len(l_w_diag))  # noqa: E741
        l_mask = (l != 0).astype(np.float32)
        u_mask = (u != 0).astype(np.float32)
        # remove ones from diag
        self.weight_init = weight
        self.register_buffer('p', torch.from_numpy(p).float())
        self.register_buffer('l_mask', torch.from_numpy(l_mask).float())
        self.register_buffer('u_mask', torch.from_numpy(u_mask).float())
        self.l = nn.Parameter(torch.from_numpy(l).float())
        self.u = nn.Parameter(torch.from_numpy(u).float())

        self.bias: Optional[nn.Parameter]

        if isinstance(bias, bool) and bias:
            self.bias = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        elif isinstance(bias, np.ndarray):
            self.bias = nn.Parameter(torch.from_numpy(bias.astype(np.float32))
                                     .view(1, n_channels, 1, 1))
        elif isinstance(bias, torch.Tensor):
            self.bias = nn.Parameter(bias.view(1, n_channels, 1, 1))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> ActivationJacobian:
        b, c, h, w = x.shape
        x = torch.nn.functional.conv2d(x, self.weight()[:, :, None, None])
        if self.bias is not None:
            x = x + self.bias
        jaccobian = h * w * self.logdet()
        return x, jaccobian

    @staticmethod
    def weight_clamping(module: Conv1x1Inv,
                        min: float = 1 / 10,
                        max: float = 10.):
        """Ensures that the eigenvalues are within (min, max).

        Use as ``model.apply(Conv1x1Inv.weihgt_clamping)``.
        """
        if type(module).__name__ == "Conv1x1Inv":
            with torch.no_grad():
                u = module.u.data
                c, c = u.shape
                diag_mask = torch.eye(c, device=u.device)
                u_diag = torch.diagonal(module.u)
                s_prod = torch.prod(u_diag.abs())

                s_clamp = s_prod.clamp(min, max)
                alpha = s_prod / s_clamp
                u_diag_norm = u_diag / (alpha**(1 / len(u_diag)))
                module.u.data = torch.diagflat(u_diag_norm) + (1 - diag_mask) * u

    def logdet(self) -> torch.Tensor:
        s = torch.diagonal(self.u)
        return s.abs().log().sum()

    def weight(self) -> torch.Tensor:
        l = self.l_mask * self.l + torch.eye(len(self.l), device=self.l.device, dtype=self.l.dtype)
        return torch.linalg.multi_dot([self.p, l, self.u_mask * self.u])

    def _inverse(self, x: torch.Tensor) -> ActivationJacobian:
        # could be speed up with tri magic
        w_inv = torch.inverse(self.weight())
        w_inv = w_inv.transpose(0, 1)[:, :, None, None]
        if self.bias is not None:
            x = x - self.bias
        b, c, h, w = x.shape
        jaccobian = -  h * w * self.logdet()
        return torch.nn.functional.conv_transpose2d(x, w_inv), jaccobian


class Logit(SimpleFlowModule):
    def __init__(
        self,
        min_value: float = - 1 / 256.,
        max_value: float = 1. + 1 / 256.,
        clamp: float = 8.,
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> ActivationJacobian:
        b = x.shape[0]

        scale = torch.tensor(self.max_value - self.min_value, device=x.device)

        y = (x - self.min_value) / scale
        # assert ((0. < y) & (y < 1.)).all()

        z = torch.log(y) - torch.log(1. - y)
        dzdy = - 1. / ((y - 1.) * y)
        dydx = 1. / scale
        jac = (dzdy.abs().log() + dydx.abs().log())
        # z = torch.clamp(z, -self.clamp, self.clamp)
        return z, jac.view(b, -1).sum(1)

    def _inverse(self, z: torch.Tensor) -> ActivationJacobian:
        b = z.shape[0]
        scale = torch.tensor(self.max_value - self.min_value, device=z.device)
        # z = torch.clamp(z, -self.clamp, self.clamp)
        y = torch.sigmoid(z)
        dydz = y * (1 - y)
        x = y * (self.max_value - self.min_value) + self.min_value
        dxdy = scale
        jac = dydz.abs().log() + dxdy.abs().log()
        return x, jac.view(b, -1).sum(1)


def split(
    x: torch.Tensor,
    channels_to_keep: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if channels_to_keep is None:
        c = x.shape[1] // 2
    else:
        assert channels_to_keep <= x.shape[1]
        c = channels_to_keep
    return x[:, :c].contiguous(), x[:, c:].contiguous()


SplittedActivation = Tuple[torch.Tensor, torch.Tensor]
SplitReturn = Tuple[
    SplittedActivation,
    Union[float, torch.Tensor]]


class Split2d(SimpleFlowModule):
    """Layer to split the feature map along the channel dimension.

    Useful for fading out dimensions.
    """

    def __init__(self, channels_to_keep: Optional[int] = None):
        super().__init__()
        self.channels_to_keep = channels_to_keep

    def forward(self, z: torch.Tensor) -> SplitReturn:
        return split(z, self.channels_to_keep), 0.

    @staticmethod
    def concat(zs: SplittedActivation) -> torch.Tensor:
        z1, z2 = zs
        if z2 is None:
            z2 = torch.randn_like(z1)
        return torch.cat([z1, z2], dim=1)

    def _inverse(self, zs: SplittedActivation) -> ActivationJacobian:
        return self.concat(zs), 0.


@dataclasses.dataclass
class FlowNLLLoss:
    n_pixels: int
    prior_mode: str = 'normal'
    quantization: float = 1. / 255

    def __call__(
        self,
        zs: Latent,
        jac: torch.Tensor,
        mixture_nll: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.prior_mode == 'normal':
            assert mixture_nll is None
            nll = self.flow_nll_loss(zs, jac)
            return self.normalize_nll(nll)
        elif self.prior_mode == 'gaussian_mixture':
            assert mixture_nll is not None
            nll_zs = self.flow_nll_loss(zs[:-1], jac)
            return self.normalize_nll(nll_zs + mixture_nll)
        else:
            raise ValueError(f'Unknown prior mode: {self.prior_mode}')

    def flow_nll_loss(
        self,
        zs: Latent,
        log_det_jac: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the flow nll loss per input dimension in bits.

        log p(z) + log det | J | - log(quanitzation)

        Args:
            zs (list of tensors): list of all outfaded tensors
            log_det_jac (tensor): Log determinante of the jacobian. If not a scalar,
                it will be summed up per batch sample.
            clamp (bool): Clamp
            mean (bool): Return nll mean over the batch.
        """
        def flatten(x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            return x.reshape(b, -1)

        b = zs[0].shape[0]
        nll_z = - sum([flatten(standard_normal_log_likelihood(z)).sum(1) for z in zs])

        if len(log_det_jac.shape) != 0:
            log_det_jac = log_det_jac.view(b, -1).sum(1)
        return (nll_z - log_det_jac)

    def normalize_nll(
        self,
        nll_z: torch.Tensor,
        mean: bool = True,
    ) -> torch.Tensor:
        nll_x = nll_z / self.n_pixels - np.log(self.quantization)
        nll_x = nll_x / np.log(2)
        if mean:
            nll_x = nll_x.mean()
        return nll_x


def flow_nll_loss(
        zs: Latent,
        log_det_jac: torch.Tensor,
        quantization: float = 1 / 255,
        mean: bool = True
) -> torch.Tensor:
    """Returns the flow nll loss per input dimension in bits.

    log p(z) + log det | J | - log(quanitzation)

    Args:
        zs (list of tensors): list of all outfaded tensors
        log_det_jac (tensor): Log determinante of the jacobian. If not a scalar,
            it will be summed up per batch sample.
        quantization (float): added noise for quanitization
        clamp (bool): Clamp
        mean (bool): Return nll mean over the batch.
    """
    def flatten(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return x.reshape(b, -1)

    per_pixel = 0
    for z in zs:
        b, c, h, w = z.shape
        per_pixel += c * h * w
    nll_z = - sum([flatten(standard_normal_log_likelihood(z)).sum(1) for z in zs])

    if len(log_det_jac.shape) != 0:
        log_det_jac = log_det_jac.view(b, -1).sum(1)
    nll_x = (nll_z - log_det_jac) / per_pixel - np.log(quantization)
    nll_x = nll_x / np.log(2)
    if mean:
        nll_x = nll_x.mean()
    return nll_x


class SequentialFlow(FlowModule):
    """Flow class containing a sequence of layers."""
    def __init__(self, layers: Sequence[FlowModule], quantization: float = 1. / 255):
        super().__init__()
        self.layers = nn.ModuleList(list(utils.flatten(layers)))
        self.quantization = quantization

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        zs, jac = self(x)
        return flow_nll_loss(zs, jac, self.quantization)

    def inverse(self) -> 'SequentialFlowInverse':
        return SequentialFlowInverse([l.inverse() for l in self.layers], self.quantization)

    def forward(self,
                x: torch.Tensor,
                start_idx: int = None,
                end_idx: int = None,
                ) -> tuple[Latent, torch.Tensor]:
        """Executes the flow on ``x`` from ``start_idx`` to ``end_idx``."""
        activations = self.get_activations(x, start_idx, end_idx)
        z, zs, jac = activations[end_idx or len(self.layers)]
        return list(zs) + [z], jac

    def get_activations(
        self,
        x: torch.Tensor,
        start_idx: int = None,
        end_idx: int = None,
        recorded_layers: list[int] = None,
    ) -> dict[int, tuple[torch.Tensor, Latent, torch.Tensor]]:
        """Executes the flow on ``x`` from ``start_idx`` to ``end_idx``."""
        b = x.shape[0]
        zs: list[torch.Tensor] = []
        sum_jacc = torch.zeros(b, device=x.device)
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.layers)

        recorded_acts: dict[int, tuple[torch.Tensor, Latent, torch.Tensor]] = {}

        for i, layer in enumerate(self.layers[start_idx:end_idx], start_idx):
            out = layer(x)
            if len(out) == 2:
                x, jacc = out
            if type(x) == tuple:
                x, z = x
                zs.append(z)
            if type(jacc) is torch.Tensor and len(jacc.shape) > 1:
                jacc = jacc.view(jacc.shape[0], -1).sum(1)
            sum_jacc = sum_jacc + jacc
            if recorded_layers is not None and i in recorded_layers:
                recorded_acts[i] = (x, tuple(zs), sum_jacc)
        recorded_acts[end_idx] = (x, tuple(zs), sum_jacc)
        return recorded_acts

    def get_shapes(self, x: torch.Tensor) -> tuple[
            dict[int, tuple[int, ...]],
            dict[int, tuple[int, ...]],
    ]:
        layer_idx = 0
        in_shapes = {}
        out_shapes = {}

        def get_shapes(
            module: torch.nn.Module,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
        ):
            nonlocal layer_idx

            inp = inputs[0]
            out = outputs[0]
            if isinstance(out, tuple):
                out = out[0]
            in_shapes[layer_idx] = tuple(inp.shape)
            out_shapes[layer_idx] = tuple(out.shape)
            layer_idx += 1

        with utils.one_time_hooks(self.layers, get_shapes), torch.no_grad():
            self(x)

        return in_shapes, out_shapes


class SequentialFlowInverse(FlowModule):
    """The inverse computation of a Flow."""
    def __init__(self, layers: Sequence[FlowModule], quantization: float = 1 / 255):
        super().__init__()
        self.layers = nn.ModuleList(list(utils.flatten(layers)))
        self.quantization = quantization

    def forward(self,
                zs: Latent,
                start_idx: int = None,
                end_idx: int = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Executes the flow backward on ``x`` from ``end_idx`` to ``start_idx``."""
        z_list = list(zs)
        del zs

        x = z_list.pop()
        sum_jacc = torch.zeros(x.shape[0], device=x.device)
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.layers)
        for layer in self.layers[start_idx:end_idx][::-1]:
            if type(layer.module).__name__ == "Split2d":
                z = z_list.pop()
                x, jacc = layer((x, z))
            else:
                x, jacc = layer(x)
            if type(x) == tuple:
                x, z = x
                z_list.append(z)
            if type(jacc) is torch.Tensor and len(jacc.shape) > 1:
                jacc = jacc.view(jacc.shape[0], -1).sum(1)
            sum_jacc = sum_jacc + jacc
        return x, sum_jacc

    def inverse(self) -> SequentialFlow:
        return SequentialFlow([l.inverse() for l in self.layers], self.quantization)


def glow_blocks(
    n_blocks: int,
    block_width: int,
    channels: int,
    coupling: str = 'affine',
    use_conv1x1: Any = True,
    loading: bool = False,
) -> Sequence[FlowModule]:
    """Returns a glow blocks.

    A block is a sequence of an actnorm, invertible 1x1 convolution and coupling layer.

    Args:
        n_blocks (int): Return this number of blocks.
        block_width (int): Size of hidden state in the block.
        channels (int): Number of channels.
        coupling (str): Coupling type (affine, no_coupling, additive).
        use_conv1x1: Use 1x1 convolution in the affine coupling nn blocks.
        loading: Skip orthonormal init.
    """
    layers: list[FlowModule] = []
    block: nn.Module

    if type(use_conv1x1) == bool:
        use_conv1x1 = [use_conv1x1] * n_blocks

    if loading:
        init = 'uniform'
    else:
        init = 'orthonormal'

    if coupling == "affine":
        for i in range(n_blocks):
            if use_conv1x1[i]:
                block = CouplingNNBlock1x1(channels // 2, block_width, channels)
            else:
                block = CouplingNNBlock(channels // 2, block_width, channels)
            layers.extend([
                ActNorm(channels),
                Conv1x1Inv(channels, init),
                AffineCoupling(block)])
    elif coupling == 'no_coupling':
        for i in range(n_blocks):
            layers.extend([
                ActNorm(channels),
                Conv1x1Inv(channels, init),
            ])
    elif coupling == 'additive':
        for i in range(n_blocks):
            if use_conv1x1[i]:
                block = CouplingNNBlock1x1(channels // 2, block_width, channels // 2)
            else:
                block = CouplingNNBlock(channels // 2, block_width, channels // 2)
            layers.extend([
                ActNorm(channels),
                Conv1x1Inv(channels, init),
                AdditiveCoupling(block)])
    else:
        raise Exception()
    return layers


def get_fade_out_and_pool(
    layer: config.FadeOutAndPool,
    channels: int,
) -> list[FlowModule]:
    flow_layers: list[FlowModule] = []

    pool_factor = utils.get(layer.pool, 1) ** 2
    if layer.channels_to_keep == 'half':
        flow_layers.append(Split2d(
            channels_to_keep=channels // 2,
        ))
    elif layer.channels_to_keep == 'no_fade_out':
        pass
    elif isinstance(layer.channels_to_keep, int):
        # Psi Layer will add a factor of 4
        flow_layers.append(Split2d(layer.channels_to_keep // pool_factor))
    else:
        raise ValueError(f'Unknown channels_to_keep: {layer.channels_to_keep}')

    if layer.pool is not None and not layer.pool == 1:
        flow_layers.append(Psi(layer.pool))
    return flow_layers


def get_new_channel_dim(
    layer: config.FadeOutAndPool,
    channels: int,
) -> int:
    if isinstance(layer.channels_to_keep, str):
        split_channels = {
            'half': channels // 2,
            'no_fade_out': channels,
        }[layer.channels_to_keep]
        pool_factor = utils.get(layer.pool, 1) ** 2
        return split_channels * pool_factor

    elif isinstance(layer.channels_to_keep, int):
        return layer.channels_to_keep


def get_model(
    cfg: config.FlowModel,
    in_channels: int = 3,
    loading: bool = False,
) -> SequentialFlow:
    flow_layers: list[FlowModule] = []

    channels = in_channels

    for layer in cfg.layers:
        if isinstance(layer, config.FlowBlocks):
            flow_layers.extend(glow_blocks(
                layer.n_blocks, layer.block_channels, channels,
                coupling=layer.coupling,
                use_conv1x1=layer.conv_1x1_kernel, loading=loading))
        elif isinstance(layer, config.FadeOutAndPool):
            flow_layers.extend(get_fade_out_and_pool(layer, channels))
            channels = get_new_channel_dim(layer, channels)
        elif isinstance(layer, config.Logit):
            flow_layers.append(Logit())

    assert cfg.prior == 'normal'
    flow_layers.extend([
        ActNorm(channels),
        Conv1x1Inv(channels),
    ])
    return SequentialFlow(flow_layers)


# utils

def concat_zs(list_of_zs: Sequence[Latent]) -> Latent:
    new_zs = []
    for i in range(len(list_of_zs[0])):
        new_zs.append(torch.cat([
            zs[i] for zs in list_of_zs
        ]))
    return new_zs


def zs_select_samples(zs: Latent, indices: list[int]) -> Latent:
    return [z[indices] for z in zs]

# TODO: get logits for classifier #6
# def get_logits(glow_net, probing_classifiers, imgs):
#     """Returns the classifiers output for the ``imgs``."""
#     predictions = {}
#
#     layer_idxs = [layer_idx for layer_idx, label_idx, cl in probing_classifiers]
#     with torch.no_grad():
#         for layer_idx, (zs, _) in glow_net.intermediate(imgs, layer_idxs):
#             for cl_layer_idx, label_idx, cl in probing_classifiers:
#                 if cl_layer_idx != layer_idx:
#                     continue
#                 z = zs[-1]
#                 b, c, h, w = z.shape
#                 logits = cl(z.view(b, c * h * w))
#                 predictions[layer_idx, label_idx, cl.name] = logits.to('cpu')
#
#     return predictions
#
#
# def get_logit_score(glow_net, zs, classifier_tuple):
#     layer_idx, label_idx, classifier = classifier_tuple
#     with torch.no_grad():
#         x, jac = glow_net.inverse()(zs, end_idx=layer_idx)
#         logits = list(get_logits(glow_net, [classifier_tuple], x).values())[0]
#         return logits


def zs_to_img(flow: SequentialFlow, zs: Latent, end_idx: int) -> torch.Tensor:
    with torch.no_grad():
        x, jac = flow.inverse()(zs, end_idx=end_idx)
        return x


def cross_product_zs(zs: Latent, z: list[torch.Tensor]) -> Latent:
    """Builds the cross product of a Latent with a list of modified activations."""

    zs = zs[:-1]
    zs_dist = []
    for i in range(len(zs[0])):
        for j in range(len(z)):
            zs_dist.append([zl[i:i + 1] for zl in zs] + [z[j][i:i + 1]])

    return concat_zs(zs_dist)


combine_zs = cross_product_zs


class InverseChecker:
    def __init__(self, flow: SequentialFlow, max_diff: float = 1e-6):
        self._flow = flow
        self._check_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._check = False
        self._fail = False
        self.max_diff = max_diff

    def check_inverse(
        self,
        module: FlowModule,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
    ):
        if not self._check:
            return
        x, = inputs
        out, jacc = outputs
        x_inv, jacc_inv = module.inverse()(out)
        x_diff = (x - x_inv).abs().mean()
        if x_diff > self.max_diff:
            print("{} {}: {:}".format(list(self._flow.layers).index(module),
                                      type(module).__name__, x_diff.item()))
            self._fail = True

    def __enter__(self):
        for layer in self._flow.layers:
            self._check_hooks.append(layer.register_forward_hook(
                self.check_inverse))

    def __exit__(self, type: Any, value: Any, traceback: Any):
        for hook in self._check_hooks:
            hook.remove()
        self._check_hooks = []
        if self._fail:
            raise ValueError("Some modules failed")


def flow_sample_image(
    flow: SequentialFlow,
    zs: Latent,
    plot: bool = True,
    filename: Optional[str] = None,
):
    """Samples images given the a list of ``zs` values."""
    flow.eval()
    with torch.no_grad():
        x_rand, _ = flow.inverse()(zs)
        x_rand = x_rand.detach().cpu()
        grid = torchvision_utils.make_grid(x_rand, nrow=10).numpy()
    if filename is not None:
        np_img = (255 * np.clip(grid.transpose(1, 2, 0), 0, 1)).astype(np.uint8)
        imageio.imsave(filename, np_img)
    flow.train()
    if plot:
        ax: mpl.axes.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # type: ignore
        ax.imshow(grid.transpose(1, 2, 0))
        plt.show()


@contextlib.contextmanager
def one_time_hooks(
    layers: Sequence[FlowModule],
    func: Callable[[nn.Module, Tuple[torch.Tensor], Tuple[torch.Tensor]], None],
):

    hooks = [layer.register_forward_hook(func) for layer in layers]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def get_output_shape(
    model: SequentialFlow,
    imgs_or_loader: Union[torch.Tensor, torch_data.DataLoader],
    do_print: bool = False,
) -> dict[int, tuple[int, ...]]:
    if isinstance(imgs_or_loader, torch_data.DataLoader):
        imgs, _, _ = next(iter(imgs_or_loader))
        imgs = imgs.to(list(model.parameters())[0].device)
    else:
        imgs = imgs_or_loader

    module_to_index: Sequence[FlowModule] = list(model.layers)
    index_to_shape: dict[int, tuple[int, ...]] = {}

    def print_output_hook(
        module: nn.Module,
        inputs: Tuple[torch.Tensor],
        outputs: Tuple[torch.Tensor],
    ):
        layer_idx = module_to_index.index(module)   # type: ignore
        out = outputs[0]
        if isinstance(out, tuple):
            out = out[0]
        if do_print:
            print(f"{layer_idx:03d}, {type(module).__name__:>20}, {str(tuple(out.shape)):>20}")
        index_to_shape[layer_idx] = tuple(out.shape)

    with one_time_hooks(list(model.layers), print_output_hook), torch.no_grad():
        model(imgs)

    return index_to_shape
