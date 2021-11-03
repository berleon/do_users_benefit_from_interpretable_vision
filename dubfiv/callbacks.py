"""Different callbacks."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from dubfiv import config as config_mod


class Callback:
    def on_model_init(self):
        pass

    def on_train_begin(self):
        pass

    def on_batch_begin(self, step: int, train: bool):
        pass

    def on_batch_end(self, step: int, train: bool):
        pass

    def on_epoch_end(self):
        pass

    def on_train_end(self):
        pass


@dataclasses.dataclass
class Callbacks(Callback):
    callbacks: list[Callback]

    def on_model_init(self):
        for cb in self.callbacks:
            cb.on_model_init()

    def on_train_begin(self):
        for cb in self.callbacks:
            cb.on_train_begin()

    def on_batch_begin(self, step: int, train: bool):
        for cb in self.callbacks:
            cb.on_batch_begin(step, train)

    def on_batch_end(self, step: int, train: bool):
        for cb in self.callbacks:
            cb.on_batch_end(step, train)

    def on_epoch_end(self):
        for cb in self.callbacks:
            cb.on_epoch_end()

    def on_train_end(self):
        for cb in self.callbacks:
            cb.on_train_end()


@dataclasses.dataclass
class LRScheduler(Callback):
    n_total_samples: int
    optimizer: torch.optim.Optimizer
    config: config_mod.LRScheduler

    @property
    def base_lr(self) -> float:
        return self.config.learning_rate

    def get_lr(self, step: int) -> float:
        raise NotImplementedError()

    @staticmethod
    def from_config(
        optimizer: torch.optim.Optimizer,
        n_total_samples: int,
        config: config_mod.LRScheduler,
    ) -> LRScheduler:
        if isinstance(config, config_mod.FadeInCosineLRScheduler):
            return FadeInCosineLRScheduler(n_total_samples, optimizer, config)
        elif config.name == 'cosine':
            return CosineLRScheduler(n_total_samples, optimizer, config)
        elif config.name == 'hill':
            return HillLRScheduler(n_total_samples, optimizer, config)
        else:
            raise ValueError(f'Unknown lr scheduler name: {config.name}')


@dataclasses.dataclass
class CosineLRScheduler(LRScheduler):
    def get_lr(self, step: int) -> float:
        def cos_to_zero_one(x: float) -> float:
            return float((np.cos(np.pi * x) + 1) / 2)
        progress = step / self.n_total_samples
        return self.base_lr * cos_to_zero_one(progress)

    def on_batch_end(self, step: int, train: bool):
        if not train:
            return
        cos_schedule = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group['lr'] = cos_schedule


@dataclasses.dataclass
class FadeInCosineLRScheduler(CosineLRScheduler):
    config: config_mod.FadeInCosineLRScheduler

    def get_lr(self, step: int) -> float:
        lr = super().get_lr(step)
        return min(1, step / self.config.fade_in_steps) * lr

    def on_batch_end(self, step: int, train: bool):
        if not train:
            return
        cos_schedule = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group['lr'] = cos_schedule


@dataclasses.dataclass
class HillScheduler():
    base_value: float
    n_total_samples: int

    def _hill(self, progress: float) -> float:
        # maps progress from [0; 1] -> [-pi, pi]
        progress_pi = 2 * np.pi * progress - np.pi
        return float((np.cos(progress_pi) + 1) / 2)

    def get(self, step: int) -> float:
        progress = step / self.n_total_samples
        return float(self.base_value * self._hill(progress))


@dataclasses.dataclass
class HillLRScheduler(CosineLRScheduler):

    hill_scheduler: HillScheduler = dataclasses.field(init=False)

    def __post_init__(self):
        self.hill_scheduler = HillScheduler(self.base_lr, self.n_total_samples)

    def get_lr(self, step: int) -> float:
        return self.hill_scheduler.get(step)
