"""Module for ground truth evaluation."""

from __future__ import annotations

import dataclasses
import glob
import itertools
import json
import os
import shutil
from typing import Iterator, Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from torch import nn
import tqdm.auto
import two4two.cli_tool

from dubfiv import classifiers as classifiers_mod
from dubfiv import data
from dubfiv import figures
from dubfiv import flow as flow_mod
from dubfiv import interpolations
from dubfiv import train as train_mod
from dubfiv import train_supervised
from dubfiv import utils


@dataclasses.dataclass
class Two4TwoInterventionAnalysis:
    loaders: dict[str, data.DataLoader]
    datasets: dict[str, data.Two4Two]
    interventions: list[two4two.cli_tool.InterventionArgs]

    @staticmethod
    def from_path(
        dataset_dir: str,
        split: str = 'test',
        batch_size: int = 50,
    ) -> Two4TwoInterventionAnalysis:
        interventions_paths = glob.glob(f'{dataset_dir}/{split}**/intervention.json')
        interventions = []
        for path in interventions_paths:
            with open(path) as f:
                interventions.append(
                    two4two.cli_tool.InterventionArgs.from_dict(json.load(f)))

        loaders = {}
        datasets = {}
        for inter in interventions:
            loader, dataset = data.load_two4two_split(
                dataset_dir,
                inter.get_key(),
                batch_size)
            loaders[inter.get_key()] = loader
            datasets[inter.get_key()] = dataset

        return Two4TwoInterventionAnalysis(loaders, datasets, interventions)

    def __post_init__(self):
        n_originals = len([inter for inter in self.interventions
                           if inter.is_original()])

        assert n_originals == 1, n_originals
        self.align_datasets()

    def original_key(self) -> str:
        for inter in self.interventions:
            if inter.is_original():
                return inter.get_key()
        raise ValueError("Not original found")

    @property
    def modified_datasets(self) -> dict[str, data.Two4Two]:
        return {
            key: dataset
            for key, dataset in self.datasets.items()
            if key != self.original_key()
        }

    def splits(self) -> tuple[str, ...]:
        return tuple(self.loaders.keys())

    def get_modified_attributes(self, key: str) -> tuple[str, ...]:
        for inter in self.interventions:
            if inter.get_key() == key:
                return inter.modified_attributes
        raise ValueError(key)

    def align_datasets(self):
        original = self.datasets[self.original_key()]
        for dataset in self.modified_datasets.values():
            params_by_id = {param.original_id: param for param in dataset.params}
            assert len(params_by_id) == len(dataset.params)

            dataset.params = [params_by_id[p.id] for p in original.params]

            for idx, (param_orig, param_mod) in enumerate(
                    zip(original.params, dataset.params)):
                assert param_orig.id == param_mod.original_id, \
                    (idx, param_orig.id, param_mod.original_id)

    def iter_batches(self) -> Iterator[
            dict[str,
                 tuple[torch.Tensor, torch.Tensor]]]:
        for batches in zip(*self.loaders.values()):
            yield dict(zip(self.loaders.keys(), batches))

    def get_dataframe(self) -> pd.DataFrame:
        dfs = []
        for key, dataset in self.datasets.items():
            df = dataset.get_dataframe()
            df['split'] = key
            df['image_idx'] = np.arange(len(df))
            dfs.append(df)
        return pd.concat(dfs)

    def get_model_logits(
        self,
        flow_w_cls: classifiers_mod.FlowWithClassifier,
        device: torch.device,
    ) -> pd.DataFrame:
        dfs = []
        for split_name, dataset in self.datasets.items():
            df = dataset.get_dataframe()

            df['image_idx'] = np.arange(len(df))
            df['split'] = split_name

            logits = []
            labels_list = []
            activations = []
            imgs_list = []
            for imgs, labels in tqdm.auto.tqdm(self.loaders[split_name],
                                               desc=split_name):
                with torch.no_grad():
                    zs, jac, logit = flow_w_cls.compute_acts_and_logits(imgs.to(device))
                    logits.append(logit.cpu().numpy())
                    activations.append(torch.flatten(zs[-1], 1).cpu().numpy())
                labels_list.append(labels.numpy())
                imgs_list.append(imgs.numpy())

            df['logit'] = np.concatenate(logits)
            df['loader_labels'] = np.concatenate(labels_list)
            df['activations'] = list(np.concatenate(activations))
            dfs.append(df)
        return pd.concat(dfs)


@dataclasses.dataclass
class Two4TwoGTEvaluation:
    experiment: train_mod.Training
    classifier: classifiers_mod.LinearClassifier
    supervised_model: nn.Module
    supervised_options: train_supervised.CLIOptions
    criterion: train_supervised.Two4TwoCriterion
    loader: data.DataLoader
    dataset: data.Two4Two
    output_index: int = 0

    @property
    def device(self) -> torch.device:
        return self.experiment.dev

    @property
    def flow(self) -> flow_mod.SequentialFlow:
        return self.experiment.model

    def __post_init__(self):
        # sync dataset with supervised model
        self.dataset.set_return_attributes(self.criterion.get_output_names())
        self.criterion.read_index_positions(self.dataset)

        self.supervised_model.to(self.device)
        self.supervised_model.eval()
        # self.supervised_model.freeze()

    @staticmethod
    def from_ckpt(
        experiment: train_mod.Training,
        classifier: classifiers_mod.LinearClassifier,
        supervised_path: str,
        split_name: str,
        device: torch.device,
        dataset_path: Optional[str] = None,
        output_index: int = 0,
    ) -> Two4TwoGTEvaluation:
        super_cls, supervised_options = train_supervised.load_model_from_ckpt(
            supervised_path, device)

        loader, dataset = data.load_two4two_split(
            utils.get(dataset_path, experiment.cfg.dataset_dir),
            split_name,
            experiment.cfg.train.batch_size,
        )

        return Two4TwoGTEvaluation(
            experiment,
            classifier,
            super_cls,
            supervised_options,
            train_supervised.Two4TwoCriterion(),
            loader,
            dataset,
            output_index,
        )

    def output_to_dict(self, output: torch.Tensor) -> dict[str, np.ndarray]:
        return {
            outdim.name: output[:, outdim.indexes].detach().cpu().numpy()
            for outdim in self.criterion.output_dims}

    def score(self, images: torch.Tensor) -> dict[str, np.ndarray]:
        out = self.supervised_model(images.to(self.device))
        return self.output_to_dict(out)

    def predict_loader(
        self,
        loader: Optional[data.DataLoader] = None,
        progbar: bool = False,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:

        out_dicts = []
        label_dicts = []

        for imgs, labels in tqdm.auto.tqdm(utils.get(loader, self.loader),
                                           disable=not progbar):
            with torch.no_grad():
                out_dicts.append(self.score(imgs))
                label_dicts.append(self.output_to_dict(labels))

        return utils.concat_dicts(out_dicts), utils.concat_dicts(label_dicts)

    def get_interpolations(
        self,
        images: torch.Tensor,
        cf_values: np.ndarray,
    ) -> torch.Tensor:
        dev = self.experiment.dev

        images_bs = images.shape[0]

        with torch.no_grad():
            imgs_inter = interpolations.transform_zs(
                self.flow, images.to(dev), self.classifier.activation_idx,
                transform=lambda z: interpolations.get_weight_interpolations(
                    z, self.classifier.fc.weight[self.output_index],
                    self.classifier.fc.bias[self.output_index],
                    cf_values,
                    relative=False))

            b, c, h, w = imgs_inter.shape
            imgs_inter = imgs_inter.view(images_bs, b // images_bs, c, h, w)
        return imgs_inter

    def score_interpolations(
        self,
        logit_min: float,
        logit_max: float,
        n_interpolations: int = 5,
        loader: Optional[data.DataLoader] = None,
        progbar: bool = True,
        sample_mode: str = 'uniform',
        n_batches: Optional[int] = None,
    ) -> pd.DataFrame:
        abs_image_idx = 0
        dicts = []
        with torch.no_grad():
            pbar = tqdm.auto.tqdm(
                itertools.islice(utils.get(loader, self.loader), n_batches),
                disable=not progbar,
                total=n_batches,
            )
            for imgs, labels in pbar:
                bs = imgs.shape[0]
                label_dict = self.output_to_dict(labels)

                if sample_mode == 'uniform':
                    logit_values = np.random.uniform(
                        logit_min, logit_max,
                        size=(len(imgs), n_interpolations))
                elif sample_mode == 'linspace':
                    logit_values = np.linspace(logit_min, logit_max, n_interpolations)
                    logit_values = logit_values[np.newaxis].repeat(len(imgs), axis=0)
                else:
                    raise ValueError(sample_mode)

                imgs_inter = self.get_interpolations(
                    imgs, logit_values)

                score_dict = [
                    self.score(imgs_inter[:, i].to(self.device))
                    for i in range(n_interpolations)]

                for img_idx in range(bs):
                    for int_idx in range(n_interpolations):
                        data = {
                            'logit_value': logit_values[img_idx, int_idx],
                            'image_idx': abs_image_idx + img_idx,
                        }
                        data.update({
                            key: float(scores[img_idx])
                            for key, scores in score_dict[int_idx].items()
                        })
                        data.update({
                            key + '_gt': float(scores[img_idx])
                            for key, scores in label_dict.items()
                        })
                        dicts.append(data)
                abs_image_idx += bs
        return pd.DataFrame(dicts)

    def get_supervised_performance(self) -> pd.DataFrame:
        preds, labels = self.predict_loader(progbar=True)
        row = []
        for key, pred, label in zip(preds.keys(), preds.values(), labels.values()):
            if key != 'obj_name':
                row.append({
                    'label_name': key,
                    'metric': 'MAE',
                    'value': np.abs(pred - label).mean(),
                })
                row.append({
                    'label_name': key,
                    'metric': 'MSE',
                    'value': np.power(pred - label, 2).mean(),
                })
            else:
                row.append({
                    'label_name': key,
                    'metric': 'Accuracy',
                    'value': ((pred > 0) == label).mean(),
                })
        return pd.DataFrame(row)

    def run(
        self,
        logit_min: float,
        logit_max: float,
        output_dir: str,
        n_interpolations: int = 5,
        loader: Optional[data.DataLoader] = None,
        progbar: bool = True,
        sample_mode: str = 'uniform',
        n_batches: Optional[int] = None,
        clear_directory: bool = False,
    ) -> Two4TwoGTResult:
        def compute_medians(df: pd.DataFrame, attrs: list[str]) -> pd.DataFrame:
            df_median = df.join(
                df.groupby('image_idx').aggregate('median'),
                on='image_idx',
                rsuffix='_median',
            )
            for attr in attrs:
                df_median[f'{attr}_diff'] = df_median[attr] - df_median[f'{attr}_median']
            return df_median

        if clear_directory and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        df_supervised = self.get_supervised_performance()

        df = self.score_interpolations(
            logit_min, logit_max, n_interpolations, loader, progbar,
            sample_mode, n_batches,
        )
        attrs = [
            key for key in df.keys()
            if not key.endswith('_gt') and key != 'image_idx'
        ]
        df = compute_medians(df, attrs)

        result = Two4TwoGTResult(
            df_supervised,
            df,
            output_dir,
        )

        return result


@dataclasses.dataclass
class Two4TwoGTResult:
    superivsed_performance: pd.DataFrame
    df: pd.DataFrame
    output_dir: str

    @property
    def attributes(self) -> list[str]:
        return [
            key for key in self.df.keys()
            if (f'{key}_median' in self.df and f'{key}_gt' in self.df)
        ]

    def plot_trajectories(
        self,
        attributes: Optional[list[str]] = None,
        n_lines: int = 35,
        seed: int = 0,
        show: bool = False,
        save: bool = False,
        xlabel: str = 'Logit',
    ) -> pd.DataFrame:

        attrs = utils.get(attributes, self.attributes)

        np.random.seed(seed)
        mask = self.df.image_idx.isin(
            np.random.choice(self.df.image_idx.unique(), n_lines)
        )
        for attr_name in attrs:
            if attr_name == 'logit_value':
                continue
            fig, ax = plt.subplots(figsize=figures.get_figure_size(0.20))
            fig.set_dpi(200)

            def plot_line(sub_df: pd.DataFrame):
                sub_df.plot(
                    y=attr_name,
                    x='logit_value',
                    ax=ax,
                    legend=None,
                    linewidth=0.33,
                    c='tab:blue'
                )

            self.df[mask].groupby('image_idx').apply(plot_line)
            ax.set_ylabel(figures.two4two_nice_names[attr_name])
            ax.set_xlabel(xlabel)
            if save:
                path = os.path.join(self.output_dir, f'{attr_name}.pgf')
                figures.savefig_pgf(path, fig)
                print('save figure at', path)
            if show:
                plt.show()
            else:
                plt.close(fig)

    def fit_ols(
        self,
        attributes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        rows = []
        attrs = utils.get(attributes, self.attributes)
        for attr in attrs:
            attr_diff = f'{attr}_diff'
            ols = sm.OLS(
                self.df[attr_diff],
                sm.add_constant(self.df.logit_value)
            ).fit()
            rows.append({
                'Factor': figures.two4two_nice_names[attr],
                '$R^2$': ols.rsquared,
                'Coeff': ols.params['logit_value'],
                'Correlation (r)': np.corrcoef(self.df[attr], self.df.logit_value)[0, 1],
                'Delta': self.df[attr_diff].abs().mean(),
            })

        return pd.DataFrame(rows)
