"""Generate config files for hyperparameter search."""

from __future__ import annotations

import copy
import os
from typing import Any, Optional, Sequence

import tap
import toml

from dubfiv import utils


class GenerateDatasetConfigArgs(tap.Tap):
    name: str
    config: str
    output_dir: str
    pdb: bool = False  # run pdb on error


def obj_color_slope(
    args: GenerateDatasetConfigArgs,
    loaded_config: dict[str, Any],
):
    for obj_color_slope in [1.5, 2.5, 3.5]:
        config = copy.deepcopy(loaded_config)
        name = f'obj_color_slope_{obj_color_slope}'
        for dataset in config['dataset']:
            dataset['sampler_config']['obj_color_slope'] = obj_color_slope
            dataset['output_dir'] = f'{dataset["output_dir"]}_{name}'
        yield name, config


def n_samples(
    args: GenerateDatasetConfigArgs,
    loaded_config: dict[str, Any],
):
    for factor in [0.5, 2, 4]:
        config = copy.deepcopy(loaded_config)
        name = f'n_samples_{factor}'
        for dataset in config['dataset']:
            for split in dataset['split']:
                split['n_samples'] = int(factor * split['n_samples'])
            dataset['output_dir'] = f'{dataset["output_dir"]}_{name}'
        yield name, config


def merge_dataset(
    args: GenerateDatasetConfigArgs,
    loaded_config: dict[str, Any],
):
    for i in range(4):
        config = copy.deepcopy(loaded_config)
        name = f'merge_{i}'
        for dataset in config['dataset']:
            dataset['output_dir'] = f'{dataset["output_dir"]}_{name}'
        yield name, config


def obj_color_uniform(
    args: GenerateDatasetConfigArgs,
    loaded_config: dict[str, Any],
):
    for obj_uniform in [0.0, 0.1, 0.3]:
        config = copy.deepcopy(loaded_config)
        name = f'obj_uniform_{obj_uniform}'
        for dataset in config['dataset']:
            dataset['sampler_config']['obj_color_uniform_prob'] = obj_uniform
            dataset['output_dir'] = f'{dataset["output_dir"]}_{name}'
        yield name, config


def cross_search_obj_color_uniform_spherical_uniform(
    args: GenerateDatasetConfigArgs,
    loaded_config: dict[str, Any],
):

    for obj_color_slope, obj_uniform in zip(
            [2.5, 2.5, 3.5],
            [0.1, 0.25, 0.25],
    ):
        for spherical_uniform in [0.25, 0.5, 0.75]:
            config = copy.deepcopy(loaded_config)
            name = (f'spherical_uniform_{spherical_uniform}_'
                    f'obj_color_slope_{obj_color_slope}_uniform_{obj_uniform}')
            for dataset in config['dataset']:
                dataset['sampler_config']['spherical_uniform_prob'] = spherical_uniform
                dataset['sampler_config']['obj_color_uniform_prob'] = obj_uniform
                dataset['sampler_config']['obj_color_slope'] = obj_color_slope
                dataset['output_dir'] = f'{dataset["output_dir"]}_{name}'
            yield name, config


def spherical_uniform(
    args: GenerateDatasetConfigArgs,
    loaded_config: dict[str, Any],
):
    for spherical_uniform in [0, 0.1, 0.25, 0.5, 0.75, 1]:
        config = copy.deepcopy(loaded_config)
        name = f'spherical_uniform_{spherical_uniform}'
        for dataset in config['dataset']:
            dataset['sampler_config']['spherical_uniform_prob'] = spherical_uniform
            dataset['output_dir'] = f'{dataset["output_dir"]}_{name}'
        yield name, config


def finer_search_obj_color_uniform_spherical_uniform(
    args: GenerateDatasetConfigArgs,
    loaded_config: dict[str, Any],
):

    for obj_uniform, spherical_uniform in [
            (0.25, 0.5),
            (0.25, 0.4),
            (0.25, 0.33),
            (0.20, 0.4),
            (0.15, 0.33),
    ]:
        config = copy.deepcopy(loaded_config)
        name = (f'finer_search_spherical_uniform_{spherical_uniform}_'
                f'uniform_{obj_uniform}')
        for dataset in config['dataset']:
            dataset['sampler_config']['spherical_uniform_prob'] = spherical_uniform
            dataset['sampler_config']['obj_color_uniform_prob'] = obj_uniform
            dataset['output_dir'] = f'{dataset["output_dir"]}_{name}'
        yield name, config


def main(argv: Optional[Sequence[str]] = None):
    args = GenerateDatasetConfigArgs().parse_args(argv)
    dataset_name, _ = os.path.splitext(os.path.basename(args.config))
    with open(args.config) as f:
        loaded_config = toml.load(f)

    with utils.pdb_post_mortem(enable=args.pdb):
        config_generator = globals()[args.name]
        outdir = os.path.join(args.output_dir, dataset_name, args.name)
        os.makedirs(outdir)
        for name, config in config_generator(args, loaded_config):
            outname = os.path.join(outdir, f'{name}.toml')
            with open(outname, 'w') as f:
                toml.dump(config, f)
