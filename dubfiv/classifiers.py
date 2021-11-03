"""Classifiers."""

from __future__ import annotations

import abc
from contextlib import contextmanager
import dataclasses
from numbers import Number
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dubfiv import callbacks
from dubfiv import config as config_mod
from dubfiv import flow
from dubfiv import pytorch_helper
from dubfiv import utils
from dubfiv.utils import AnyTensor


T = TypeVar('T', bound='Classifier')


class Classifier(nn.Module, callbacks.Callback, metaclass=abc.ABCMeta):

    optimizer: torch.optim.Optimizer

    def _setup_lr_scheduler(self, n_total_samples: int):
        self.lr_scheduler = callbacks.LRScheduler.from_config(
            self.optimizer,
            n_total_samples,
            self.config.lr_scheduler,
        )

    @property
    def label_name(self):
        return self.config.label_name

    @property
    def name(self):
        clsname = type(self).__name__
        cls_marker = clsname.replace("Classifier", "").lower()
        label_marker = self.config.label_name
        layer = self.config.after_layer
        return f"{layer}_{label_marker}_{cls_marker}"

    @property
    def activation_idx(self):
        return self.config.after_layer + 1

    @property
    @abc.abstractmethod
    def config(self) -> config_mod.Classifier:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(
        cls,
        config: Any,
        model: flow.SequentialFlow,
        n_total_samples: int,
    ) -> Classifier:
        if isinstance(config, config_mod.LinearClassifier):
            return LinearClassifier.from_config(config, model, n_total_samples)
        elif isinstance(config, config_mod.GaussianMixtureClassifier):
            return GaussianMixtureClassifier.from_config(config, model, n_total_samples)
        else:
            raise ValueError(f"Found no classifier for config: {config}")

    def on_batch_begin(self, step: int, train: bool):
        if train:
            self.optimizer.zero_grad()

    def on_batch_end(self, step: int, train: bool):
        self.lr_scheduler.on_batch_end(step, train)
        if train:
            self.optimizer.step()

    @abc.abstractmethod
    def loss(
        self,
        target: torch.Tensor,
    ) -> Union[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Callback for when a the model is done computing."""
        pass

    @abc.abstractmethod
    def record_metrics(self, enable: bool = True):
        pass

    @abc.abstractmethod
    def get_recorded_metrics(self) -> dict[str, float]:
        pass


@dataclasses.dataclass
class BinaryLogitAccumulator:
    _loss: list[np.ndarray] = dataclasses.field(
        default_factory=list, init=False, repr=False)
    _logits: list[np.ndarray] = dataclasses.field(
        default_factory=list, init=False, repr=False)
    _targets: list[np.ndarray] = dataclasses.field(
        default_factory=list, init=False, repr=False)

    def add(self,
            loss: AnyTensor,
            logits: AnyTensor,
            targets: AnyTensor):
        self._loss.append(utils.to_numpy(loss))
        self._logits.append(utils.to_numpy(logits))
        self._targets.append(utils.to_numpy(targets))

    @property
    def loss(self):
        return np.array(self._loss)

    @property
    def logits(self):
        return np.concatenate(self._logits)

    @property
    def targets(self):
        return np.concatenate(self._targets)

    def get_metrics(self) -> dict[str, float]:
        accuracy = ((self.logits > 0) == self.targets).mean()
        return {
            'accuracy': float(accuracy),
            'loss': float(np.mean(self.loss)),
        }


@dataclasses.dataclass
class GaussianMixtureAccumulator:
    binary_logit_acc: BinaryLogitAccumulator = dataclasses.field(
        default_factory=BinaryLogitAccumulator, init=False, repr=False)
    _nll: list[np.ndarray] = dataclasses.field(
        default_factory=list, init=False, repr=False)
    _supervised_loss: list[np.ndarray] = dataclasses.field(
        default_factory=list, init=False, repr=False)

    @property
    def loss(self):
        return self.binary_logit_acc.loss

    @property
    def logits(self):
        return self.binary_logit_acc.logits

    @property
    def targets(self):
        return self.binary_logit_acc.targets

    @property
    def nll(self):
        return np.array(self._nll)

    @property
    def supervised_loss(self):
        return np.array(self._supervised_loss)

    def add(self,
            loss: AnyTensor,
            logits: AnyTensor,
            targets: AnyTensor,
            nll: AnyTensor,
            supervised_loss: AnyTensor,
            ):
        self.binary_logit_acc.add(loss, logits, targets)
        self._nll.append(utils.to_numpy(nll))
        self._supervised_loss.append(utils.to_numpy(supervised_loss))

    def get_metrics(self) -> dict[str, float]:
        metrics = self.binary_logit_acc.get_metrics()
        metrics.update(dict(
            nll=float(np.mean(self.nll)),
            supervised_loss=float(np.mean(self.supervised_loss)),
        ))
        return metrics


class LinearClassifier(Classifier):
    """Linear classifier."""

    def __init__(
        self,
        config: config_mod.LinearClassifier,
        hook: pytorch_helper.RecordOutputHook,
        n_total_samples: int,
    ):
        super().__init__()
        self._config = config
        self.fc = nn.Linear(self.config.in_channels, 1)
        self.metrics: Optional[BinaryLogitAccumulator] = None
        self.hook = hook
        self.optimizer = torch.optim.SGD(self.parameters(), 0)
        self._setup_lr_scheduler(n_total_samples)
        self._loss = torch.tensor(0)

    @property
    def config(self) -> config_mod.LinearClassifier:
        return self._config

    @classmethod
    def from_config(
        cls,
        config: config_mod.LinearClassifier,
        model: flow.SequentialFlow,
        n_total_samples: int,
    ) -> LinearClassifier:
        layer_idx = config.after_layer
        hook = pytorch_helper.RecordOutputHook(model.layers[layer_idx])
        return LinearClassifier(config, hook, n_total_samples)

    @property
    def in_features(self):
        return self.fc.in_features

    @property
    def n_classes(self):
        return self.fc.out_features

    def forward(self, x):
        return torch.addmm(self.fc.bias, x, self.fc.weight.t().clone())

    def get_logit_from_record(self) -> torch.Tensor:
        hidden_activations, _ = self.hook.recorded_output

        if not self.config.backpropagate_loss:
            hidden_activations = hidden_activations.detach()

        device = list(self.parameters())[0].device
        hidden_activations = hidden_activations.to(device)

        b, c, h, w = hidden_activations.shape
        flat_output = hidden_activations.view(b, c * h * w).clone()
        logits = self(flat_output)
        return logits

    def loss(
        self,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = 'mean',
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = list(self.parameters())[0].device
        target = target.to(device)

        logits = self.get_logit_from_record()

        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(), weight=weight,
            size_average=size_average, reduce=reduce,
            reduction=reduction, pos_weight=pos_weight)
        if self.metrics is not None:
            self.metrics.add(bce, logits, target)

        self._loss = bce
        weighted_loss = self.config.loss_weight * bce
        return weighted_loss

    def record_metrics(self, enable: bool = True):
        if enable:
            self.metrics = BinaryLogitAccumulator()
        else:
            self.metrics = None

    def get_recorded_metrics(self) -> dict[str, float]:
        if self.metrics is None:
            raise ValueError("No recorded metrics!")

        return self.metrics.get_metrics()


def get_classifiers(
    classifiers: config_mod.Classifiers,
    model: flow.SequentialFlow,
    n_total_samples: int,
) -> List[Classifier]:
    return [
        Classifier.from_config(classifier_cfg, model, n_total_samples)
        for classifier_cfg in classifiers.classifiers
    ]


def load_classifier(state, device='cpu'):
    clz = globals()[state['class_name']]
    classifier = clz(**state['kwargs'])
    classifier.load_state_dict(state['state_dict'])
    classifier = classifier.to(device)
    return [state['layer_idx'], state['label_idx'], classifier]


@dataclasses.dataclass(init=False)
class FlowWithClassifier(nn.Module):
    model: flow.SequentialFlow
    classifier: Classifier
    start_layer: int

    def __init__(
        self,
        model: flow.SequentialFlow,
        classifier: Classifier,
        start_layer: int = 0,
    ):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.start_layer = start_layer

    def get_layer(self, layer_name: str):
        idx = int(layer_name)
        return self.model.layers[idx]

    @property
    def after_layer(self) -> int:
        return self.classifier.config.after_layer

    @property
    def activation_idx(self) -> int:
        return self.classifier.activation_idx

    def compute_acts_and_logits(
        self,
        x: torch.Tensor,
        start_layer: Optional[int] = None,
    ) -> tuple[flow.Latent, torch.Tensor, torch.Tensor]:
        zs, jac = self.model(
            x,
            start_idx=start_layer if start_layer is not None else self.start_layer,
            end_idx=self.activation_idx,
        )
        activation = zs[-1]
        logits = self.classifier(torch.flatten(activation, 1))
        return zs, jac, logits

    def forward(
        self,
        x: torch.Tensor,
        start_layer: Optional[int] = None,
    ) -> torch.Tensor:
        _, _, logits = self.compute_acts_and_logits(x, start_layer)
        return logits

    def get_cutted_model(self, cut_off_before: int) -> FlowWithClassifier:
        return FlowWithClassifier(
            self.model,
            self.classifier,
            start_layer=cut_off_before + 1,
        )


class GaussianMixtureClassifier(Classifier):
    def __init__(
        self,
        config: config_mod.GaussianMixtureClassifier,
        hook: pytorch_helper.RecordOutputHook,
        n_total_samples: int,
    ):
        super().__init__()
        self.gm = GaussianMixture(config.in_channels, config.n_mixtures)
        self._config = config
        assert self.config.n_mixtures == 2

        self.hook = hook
        self.metrics: Optional[GaussianMixtureAccumulator] = None
        self.optimizer = torch.optim.SGD(self.parameters(), 0)
        self._setup_lr_scheduler(n_total_samples)
        self._loss = torch.tensor(0)

    @property
    def config(self) -> config_mod.GaussianMixtureClassifier:
        return self._config

    @classmethod
    def from_config(
        cls,
        config: config_mod.GaussianMixtureClassifier,
        model: flow.SequentialFlow,
        n_total_samples: int,
    ) -> GaussianMixtureClassifier:
        layer_idx = config.after_layer
        hook = pytorch_helper.RecordOutputHook(model.layers[layer_idx])
        return GaussianMixtureClassifier(
            config,
            hook,
            n_total_samples)

    def forward(self, x) -> torch.Tensor:
        return self.gm(x)

    def run(self, x, target) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
    ]:
        # we use binary cross entropy
        assert self.config.n_mixtures == 2
        logits = self.gm(x)[:, :1]

        supervised_loss = F.binary_cross_entropy_with_logits(
            logits, target.float())

        nll_loss = self.gm.nll_loss(x, target.long()).sum(1).mean()
        loss = self.config.loss_weight * supervised_loss
        return loss, logits, target, nll_loss, supervised_loss

    def loss(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_activations, _ = self.hook.recorded_output

        if not self.config.backpropagate_loss:
            hidden_activations = hidden_activations.detach()

        device = list(self.parameters())[0].device
        hidden_activations = hidden_activations.to(device)
        target = target.to(device)

        b, c, h, w = hidden_activations.shape
        flat_output = hidden_activations.view(b, c * h * w).clone()

        loss, logits, target, nll_loss, supervised_loss = self.run(
            flat_output, target)

        if self.metrics is not None:
            self.metrics.add(
                loss, logits, target,
                nll_loss, supervised_loss)

        return supervised_loss, nll_loss

    def record_metrics(self, enable: bool = True):
        if enable:
            self.metrics = GaussianMixtureAccumulator()
        else:
            self.metrics = None

    def get_recorded_metrics(self) -> dict[str, float]:
        if self.metrics is None:
            raise ValueError("No recorded metrics!")
        return self.metrics.get_metrics()


class GaussianMixture(nn.Module):
    def __init__(
        self,
        channels_in: int,
        n_mixtures: int
    ):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.channels_in = channels_in
        self.weights = nn.Parameter(torch.ones(self.n_mixtures) / self.n_mixtures)
        self.mixture_stds = nn.Parameter(torch.ones(self.n_mixtures, channels_in))
        self.mixture_means = nn.Parameter(
            1. / channels_in * torch.randn(self.n_mixtures, channels_in))
        self.mixture_weights = nn.Parameter(
            torch.ones(self.n_mixtures) / self.n_mixtures)

    def nll_loss(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        means = self.mixture_means[label]
        stds = self.mixture_stds[label]
        x_norm = (x - means) / stds
        nll = - flow.standard_normal_log_likelihood(x_norm)
        jac = torch.log(1 / stds.abs()).sum(1)
        return nll.sum(1) - jac

    def log_likelihood(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        nll = self.nll_loss(x, label)
        return - nll

    def sample(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        mixture = torch.randint(high=self.n_mixtures, size=(size,))
        means = self.mixture_means[mixture]
        stds = self.mixture_stds[mixture]
        x = torch.randn(size, self.channels_in)
        return stds * x + means, mixture

    def forward(self, x) -> torch.Tensor:

        b, c = x.shape
        label = torch.ones(b, dtype=torch.long, device=x.device)
        nll_losses = torch.stack([
            self.log_likelihood(x, i * label)
            for i in range(self.n_mixtures)], dim=1)
        weights = 1 / self.n_mixtures * torch.ones(
            self.n_mixtures, device=x.device)
        log_weights = torch.log(weights)
        log_p_x = torch.logsumexp(log_weights[None, :] + nll_losses, 1)

        assert torch.isfinite(log_p_x).all()
        log_p_c = log_weights[None, :]
        log_p_x_c = nll_losses
        assert torch.isfinite(log_p_x_c).all()
        log_p_c_x = log_p_x_c + log_p_c - log_p_x[:, None]
        assert torch.isfinite(log_p_c_x).all()
        return log_p_c_x  # , log_p_x_c, log_p_c, log_p_x
