#!/usr/bin/env python
"""Training script."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import logging
import os
import shutil
import socket
from typing import Any, Dict, Mapping, Optional, Sequence, Type, TypeVar, Union

import toml
import torch
from torch.utils.data import DataLoader

from dubfiv import data
from dubfiv import utils


C = TypeVar('C', bound='Config')


class Config:
    @classmethod
    def from_toml(cls: Type[C], filename: str) -> C:
        with open(filename) as f:
            return cls.from_dict(toml.load(f))

    @classmethod
    def from_dict(cls: Type[C], state: Mapping[str, Any]) -> C:
        return cls(**state)  # type: ignore

    def asdict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# LR Scheduling

@dataclasses.dataclass
class LRScheduler(Config):
    name: str
    learning_rate: float

    @staticmethod
    def from_config(config: Dict[str, Any]) -> LRScheduler:
        if config['name'] == 'fade_in_cosine':
            return FadeInCosineLRScheduler(
                config['name'],
                config['learning_rate'],
                config['fade_in_steps'],
            )
        else:
            return LRScheduler(config['name'], config['learning_rate'])


@dataclasses.dataclass
class FadeInCosineLRScheduler(LRScheduler):
    fade_in_steps: int


# Network Building
LAYER = TypeVar('LAYER', bound='FlowLayer')


class FlowLayer(Config):
    name: str

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> FlowLayer:
        layer_name = state['name']
        if layer_name == 'fade_out_and_pool':
            return FadeOutAndPool(**state)
        elif layer_name == 'flow_blocks':
            return FlowBlocks(**state)
        elif layer_name == 'logit':
            return Logit(**state)
        else:
            raise ValueError(f'Unknown layer type: {layer_name}')


@dataclasses.dataclass
class Logit(FlowLayer):
    name: str = 'logit'


@dataclasses.dataclass
class FlowBlocks(FlowLayer):
    name: str = 'flow_blocks'
    n_blocks: int = 8
    block_channels: int = 24
    coupling: str = 'affine'
    conv_1x1_kernel: Union[bool, list[bool]] = False


@dataclasses.dataclass
class FadeOutAndPool(FlowLayer):
    name: str = 'fade_out_and_pool'
    channels_to_keep: Union[int, str] = 'half'
    pool: Optional[int] = 2


@dataclasses.dataclass
class FlowModel(Config):
    layers: Sequence[FlowLayer]
    prior: str = 'normal'

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> FlowModel:
        return cls(
            [FlowLayer.from_dict(s) for s in state['layers']],
            prior=state['prior'],
        )


# Classifiers


CLF = TypeVar('CLF', bound='Classifier')


@dataclasses.dataclass
class Classifier(Config):
    # insert classifier after this layers
    after_layer: int
    label_name: str
    classifier: str
    lr_scheduler: LRScheduler = LRScheduler('cosine', learning_rate=5e-5)
    backpropagate_loss: bool = False
    loss_weight: float = 1.
    # loss_weight: float = 1.

    @classmethod
    def from_dict(cls: Type[CLF], state: Mapping[str, Any]) -> CLF:
        state_dict = dict(state)
        lr_scheduler = LRScheduler.from_config(state_dict.pop('lr_scheduler'))
        return cls(
            lr_scheduler=lr_scheduler,
            **state_dict,
        )


@dataclasses.dataclass
class LinearClassifier(Classifier):
    in_channels: int = 0
    classifier: str = 'linear'


@dataclasses.dataclass
class GaussianMixtureClassifier(Classifier):
    in_channels: int = 0
    n_mixtures: int = 0
    classifier: str = 'gaussian_mixture'

    def __post_init__(self):
        if self.n_mixtures < 2:
            raise ValueError(f'Need at least 2 mixtures. Got {self.n_mixtures}')


@dataclasses.dataclass
class Classifiers(Config):
    # insert classifier after these layers
    classifiers: Sequence[Classifier]

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Classifiers:
        classifiers: list[Classifier] = []
        for classifier_config in state['classifiers']:
            if classifier_config['classifier'] == 'linear':
                classifiers.append(LinearClassifier.from_dict(classifier_config))
            elif classifier_config['classifier'] == 'gaussian_mixture':
                classifiers.append(GaussianMixtureClassifier.from_dict(classifier_config))

        return Classifiers(classifiers)


TorchDevice = Union[torch.device, str]


# Training


@dataclasses.dataclass
class Train(Config):
    dataset: str = 'celeba'
    dataset_dir: Optional[str] = None
    image_size: int = 128
    output_dir: str = '../models/'
    label_noise: float = 0.
    # train on that many samples
    n_train_samples: int = 3000000
    batch_size: int = 40
    num_workers: int = 4
    # weight of the supervised loss. is devided by the number of input dimensions
    device: TorchDevice = 'cuda'
    lr_scheduler: LRScheduler = LRScheduler('cosine', learning_rate=5e-5)

    def get_dataset_dir(self) -> str:
        if self.dataset_dir is None:
            raise ValueError("Dataset directory not set!")
        return self.dataset_dir

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Train:
        state_dict = dict(state)
        lr_scheduler_cfg = state_dict.pop('lr_scheduler')
        lr_scheduler = LRScheduler.from_config(lr_scheduler_cfg)
        return cls(lr_scheduler=lr_scheduler, **state_dict)


def resolve_path(name: str, path_file: Optional[str] = None) -> str:
    paths = get_path_mappings(path_file)
    if name in paths:
        return paths[name]
    elif 'dataset_dirs' in paths:
        dataset_dirs = paths['dataset_dirs'].split(':')
        for dirname in dataset_dirs:
            if os.path.exists(os.path.join(dirname, name)):
                return os.path.join(dirname, name)

    raise KeyError(f'Could not resolve path with name: {name}')


def get_path_mappings(filename: Optional[str] = None) -> Mapping[str, str]:
    """A file like the ones in `config/paths/`."""

    # first check the enviroment variable
    if filename is None:
        filename = os.environ.get('INN4IML_PATHS')

    # test if we can find a host specific config
    if filename is None:
        this_dir = os.path.dirname(__file__)
        filename = f'{this_dir}/../config/paths/{socket.gethostname()}.toml'
        # other wise resort to default host mapping
        if not os.path.exists(filename):
            filename = f'{this_dir}/../config/paths/default.toml'
    return toml.load(filename)


@dataclasses.dataclass
class Experiment(Config):
    output_dir: str
    train: Train
    model: FlowModel
    classifiers: Classifiers
    resolve_paths_filename: Optional[str] = None
    pdb: bool = False
    wandb_mode: str = 'online'

    _unique_marker: Optional[str] = None
    _time_tag: Optional[str] = None

    def dirname(self) -> str:
        return self.get_dirname(self.unique_marker, self.time_tag)

    @staticmethod
    def get_dirname(
        unique_marker: str,
        time_tag: str,
    ) -> str:
        return f'{unique_marker}_{time_tag}'

    def create_time_tag(self):
        self._time_tag = Experiment.get_time_tag()

    @property
    def time_tag(self) -> str:
        if self._time_tag is None:
            return os.path.basename(self.output_dir).split('_')[-1]
        return self._time_tag

    @property
    def unique_marker(self) -> str:
        if self._unique_marker is not None:
            return self._unique_marker
        parts = os.path.basename(self.output_dir).split('_')[:-1]
        return '_'.join(parts)

    def makedirs(self):
        os.makedirs(self.output_dir_images, exist_ok=True)
        os.makedirs(self.output_dir_models, exist_ok=True)
        os.makedirs(self.output_dir_source, exist_ok=True)

    def load_datasets(
        self,
        dataset_kwargs: Dict[str, Any] = {},
    ) -> tuple[
            DataLoader, DataLoader, DataLoader,
            data.Dataset, data.Dataset, data.Dataset]:
        train = self.train
        logging.info(f"load dataset {train.dataset} from {train.dataset_dir}")
        return data.load_datasets(
            train.dataset,
            self.dataset_dir,
            train.batch_size,
            train.image_size,
            num_workers=train.num_workers,
            dataset_kwargs=dataset_kwargs
        )

    @property
    def dataset_dir(self) -> str:
        if self.train.dataset_dir is not None:
            return self.train.dataset_dir
        else:
            return resolve_path(self.train.dataset, self.resolve_paths_filename)

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None,
                  description: str = 'Train the model') -> Experiment:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--model', type=str, required=False,
                            help='path to model config file.')
        parser.add_argument('--train', type=str, required=False,
                            help='path to train config file.')
        parser.add_argument('--classifiers', type=str, required=False,
                            help='path to classifiers config file.')
        parser.add_argument('--experiment', type=str, required=False,
                            help='path to a experiment config file.')
        parser.add_argument('--resolve_paths', type=str, default=None, required=False,
                            help=('Paths file for the datasets. Can also be set by the '
                                  'INN4IML_PATHS enviroment variable.'))
        parser.add_argument('--output_base_dir', type=str, default=None, required=True,
                            help='Creates <output_base_dir>/<output_dir> to save state.')
        parser.add_argument('--dataset', type=str, default=None, required=False,
                            help='Overwrites the dataset given in the train file.')
        # TODO: add name
        parser.add_argument('--pdb', action='store_true',
                            help='Enter pdb on error.')
        parser.add_argument('--no_wandb', action='store_true',
                            help="Don't log to wandb.")
        parser.add_argument('--device', type=str, default=None, required=False,
                            help='torch device')
        args = parser.parse_args(args=argv)

        if args.experiment is None:
            assert args.train is not None
            assert args.model is not None
            assert args.classifiers is not None

            wandb_mode = {
                True: 'disabled',
                False: 'online',
            }[args.no_wandb]

            with utils.pdb_post_mortem():
                exp = Experiment.create_from_config_files(
                    args.output_base_dir,
                    args.train,
                    args.model,
                    args.classifiers,
                    args.resolve_paths,
                    args.device,
                    args.pdb,
                    wandb_mode,
                    dataset_overwrite=args.dataset,
                    copy_config_files=True,
                )
        else:
            exp = Experiment.from_toml(args.experiment)
            exp.create_time_tag()
            exp.pdb = args.pdb
            exp.resolve_paths_filename = args.resolve_paths

            if args.device:
                exp.train.device = args.device

            if args.output_base_dir:
                base_dir = args.output_base_dir
            else:
                base_dir = os.path.dirname(exp.output_dir)
            exp.output_dir = os.path.join(base_dir, exp.dirname())

        return exp

    @staticmethod
    def load_from_config_files(
        output_dir: str,
        train_filename: str,
        model_filename: str,
        classifiers_filename: str,
        resolve_paths_filename: Optional[str] = None,
        device: Optional[str] = None,
        pdb: bool = False,
        wandb_mode: str = 'online',
        dataset_overwrite: Optional[str] = None,
        unique_marker: Optional[str] = None,
        time_tag: Optional[str] = None,
    ) -> Experiment:
        train_cfg = Train.from_toml(train_filename)

        if device is not None:
            train_cfg.device = device

        if dataset_overwrite is not None:
            train_cfg.dataset = dataset_overwrite

        return Experiment(
            output_dir,
            train_cfg,
            FlowModel.from_toml(model_filename),
            Classifiers.from_toml(classifiers_filename),
            resolve_paths_filename,
            pdb,
            wandb_mode,
            _unique_marker=unique_marker,
            _time_tag=time_tag,
        )

    @staticmethod
    def get_time_tag() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat()

    @staticmethod
    def create_from_config_files(
        output_base_dir: str,
        train_filename: str,
        model_filename: str,
        classifiers_filename: str,
        resolve_paths_filename: Optional[str] = None,
        device: Optional[str] = None,
        pdb: bool = False,
        wandb_mode: str = 'online',
        dataset_overwrite: Optional[str] = None,
        copy_config_files: bool = False,
    ) -> Experiment:
        train_cfg = Train.from_toml(train_filename)
        dataset_name = utils.get(dataset_overwrite, train_cfg.dataset)

        unique_marker = dataset_name
        time_tag = Experiment.get_time_tag()

        output_dir = os.path.join(
            os.path.abspath(output_base_dir),
            Experiment.get_dirname(unique_marker, time_tag))

        os.makedirs(output_dir)

        shutil.copyfile(train_filename, os.path.join(output_dir, "train.toml"))
        shutil.copyfile(model_filename, os.path.join(output_dir, "model.toml"))
        shutil.copyfile(classifiers_filename,
                        os.path.join(output_dir, "classifiers.toml"))
        with open(os.path.join(output_dir, "args.toml"), 'w') as f:
            toml.dump(dict(
                dataset_overwrite=dataset_overwrite,
                train_filename=train_filename,
                model_filename=model_filename,
                classifiers_filename=classifiers_filename,
                resolve_paths_filename=resolve_paths_filename,
                device=device,
            ), f)

        return Experiment.load_from_config_files(
            output_dir,
            train_filename,
            model_filename,
            classifiers_filename,
            resolve_paths_filename,
            device,
            pdb,
            wandb_mode,
            dataset_overwrite,
            unique_marker=unique_marker,
            time_tag=time_tag,
        )

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Experiment:
        return Experiment(
            state['output_dir'],
            Train.from_dict(state['train']),
            FlowModel.from_dict(state['model']),
            Classifiers.from_dict(state['classifiers']),
            resolve_paths_filename=state.get('resolve_paths_filename'),
            pdb=state.get('pdb', False),
            wandb_mode=state.get('wandb_mode', 'online'),
            _unique_marker=state.get('_unique_marker'),
            _time_tag=state.get('_time_tag'),
        )

    @property
    def output_dir_images(self) -> str:
        return os.path.join(self.output_dir, "images")

    @property
    def output_dir_models(self) -> str:
        return os.path.join(self.output_dir, "models")

    @property
    def output_dir_source(self) -> str:
        return os.path.join(self.output_dir, "source")
