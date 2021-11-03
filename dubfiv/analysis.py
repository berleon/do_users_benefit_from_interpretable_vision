"""user study."""

from __future__ import annotations

import abc
import copy
import dataclasses
import glob
import io
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Optional, Sequence

import attr
from cairosvg import svg2png
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
import skimage.color
import skimage.transform
import statsmodels.api as sm
import torch
from torch.utils.data import DataLoader
import tqdm.auto
import wandb

from dubfiv import classifiers as classifiers_mod
from dubfiv import data
from dubfiv import figures
from dubfiv import flow
from dubfiv import gt_eval as gt_eval_mod
from dubfiv import interpolations

from dubfiv import train as train_mod
from dubfiv import utils
from dubfiv.pytorch_helper import move_to


@dataclasses.dataclass
class ConditionData:
    zs: flow.Latent
    images: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor
    classifier: classifiers_mod.Classifier
    cf_bins: np.ndarray
    cf_values: np.ndarray

    def summary(self) -> str:
        return f"""#samples: {len(self.images)}
classifier name: {self.classifier.name}
classifier layer: {self.classifier.activation_idx}
cf_bins: {self.cf_bins}
cf_values: {self.cf_values}"""

    @staticmethod
    def from_dataset(
        model: flow.SequentialFlow,
        classifier: classifiers_mod.Classifier,
        dataset: data.Dataset,
        dataset_name: str,
        device: torch.device,
        batch_size: int = 50,
        seed: int = 0,
    ) -> ConditionData:
        if dataset_name == "celeba":
            n_images = 3000
        elif dataset_name.startswith("two4two"):
            n_images = 3000
        elif dataset_name.startswith("mice"):
            n_images = 500

        dataset_shuffled = data.DeterministicShuffle(dataset, seed)
        loader = DataLoader(dataset_shuffled, batch_size=batch_size)

        images, labels = data.collect_data_and_labels(
            loader, n_batches=n_images // loader.batch_size  # type: ignore
        )

        zs_list = []
        logit_list = []
        flow_w_cls = classifiers_mod.FlowWithClassifier(model, classifier)

        with torch.no_grad():
            for image_batch in tqdm.auto.tqdm(utils.batchify(images, 200)):
                zs, jac, logits = flow_w_cls.compute_acts_and_logits(
                    image_batch.to(device)
                )
                zs_list.append(move_to(zs, "cpu"))
                logit_list.append(logits.cpu())

        cf_values, cf_bins = get_counterfactual_bins(
            logits, logit_percentiles=get_default_percentiles(dataset_name)
        )
        return ConditionData(
            zs=flow.concat_zs(zs_list),
            images=images,
            labels=labels,
            logits=torch.cat(logit_list),
            classifier=classifier,
            cf_bins=cf_bins,
            cf_values=cf_values,
        )

    @staticmethod
    def load(classifier: classifiers_mod.Classifier, path: str) -> ConditionData:
        state = torch.load(path)
        cond_data = ConditionData(
            state["zs"],
            state["images"],
            state["labels"],
            state["logits"],
            copy.deepcopy(classifier),
            state["cf_bins"],
            state["cf_values"],
        )
        cond_data.classifier.load_state_dict(state["classifier"])
        return cond_data

    def state_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "labels": self.labels,
            "logits": self.logits,
            "zs": self.zs,
            "classifier": self.classifier.state_dict(),
            "cf_bins": self.cf_bins,
            "cf_values": self.cf_values,
        }

    def save(self, path: str):
        torch.save(self.state_dict(), path)


def get_default_percentiles(
    dataset_name: str,
) -> list[float]:
    if dataset_name == "celeba":
        logit_percentiles = [40.0, 85.0]
    elif dataset_name.startswith("two4two"):
        logit_percentiles = [33.33, 75.0]
    else:
        raise Exception()
    return logit_percentiles


def get_counterfactual_bins(
    logits: torch.Tensor,
    logit_percentiles: Sequence[float] = None,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the bins and values for the counterfactual interpolations.

    Returns a tuple of counterfacutal values and bins borders between them.
    """

    with torch.no_grad():
        abs_logits = logits.abs().cpu().numpy()
        abs_scales = np.percentile(abs_logits, logit_percentiles)

        logit_scales = np.concatenate([-abs_scales[::-1], np.array([0]), abs_scales])

        cf_bins = (logit_scales[:-1] + logit_scales[1:]) / 2
        cf_values = logit_scales.astype(np.float32)

    cf_bins_w_inf = np.array([-np.inf] + cf_bins.tolist() + [np.inf])
    return cf_values, cf_bins_w_inf


def load_runs_from_wandb(
    filter_by_name: Optional[Callable[[str], bool]] = None,
) -> pd.DataFrame:
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{utils.WANDB_ENTITY}/dubfiv")

    data = []

    for run in runs:
        if filter_by_name is not None and not filter_by_name(run.name):
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        tempfile.mkdtemp()
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_json = run.file("wandb-metadata.json").download(root=tmp_dir)
            meta = json.load(meta_json)
        data.append(
            {
                "name": run.name,
                "state": run.state,
                "config": config,
                "summary": summary,
                "meta": meta,
                "id": run.id,
                "project": run.project,
                "entity": run.entity,
                "url": run.url,
            }
        )

    return pd.DataFrame(data)


@dataclasses.dataclass
class Analysis:
    ex: train_mod.Training
    explained_classifier: classifiers_mod.LinearClassifier

    @staticmethod
    def from_checkpoint(path: str) -> Analysis:
        experiment = train_mod.Training.load(
            path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        assert len(experiment.classifiers) == 1
        return Analysis(experiment, experiment.classifiers[0])  # type: ignore

    def __post_init__(self):
        mpl.rcParams["pgf.texsystem"] = "pdflatex"
        self.flow_w_cls = classifiers_mod.FlowWithClassifier(
            self.ex.model,
            self.explained_classifier,
        )
        # self.task_loader = self.get_condition_loader(self.ex.val_set)
        # self.task = self.load_condition_data(self.task_loader)

        path = self.ex.cfg.output_dir
        self.output_dir = path
        self.model_tag = os.path.basename(self.output_dir)
        self.model_path = os.path.join(path, "models/models.torch")
        self.export_dir = os.path.join(path, "tmp-export")
        self.cache_dir = os.path.join(path, "tmp-cache")
        self.figure_dir = os.path.join(path, "figures")
        self.user_study_dir = os.path.join(path, "user_study")

        os.makedirs(self.export_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.user_study_dir, exist_ok=True)

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_set,
            self.test_set,
            self.val_set,
        ) = self.ex.cfg.load_datasets()
        self.dataset_name = self.ex.cfg.train.dataset

        self.flow = self.ex.model
        self.flow_w_cls = classifiers_mod.FlowWithClassifier(
            self.flow,
            self.explained_classifier,
        )
        self.classifiers: list[
            classifiers_mod.LinearClassifier
        ] = self.ex.classifiers  # type: ignore

        self.classifier_act_idxs = [cl.activation_idx for cl in self.classifiers]
        self.classifier_act_idx = self.explained_classifier.activation_idx

        self.task = self.get_task_data()
        self.treatment = self.get_treatment_data()

    def summary(self) -> str:
        return f"""
output_dir: {self.output_dir}
Layers: {len(self.ex.model.layers)}
Classifier after layer: {self.explained_classifier.config.after_layer}

Parameters: {self.ex.count_parameters()}
"""

    @property
    def device(self) -> torch.device:
        return self.ex.dev

    def get_condition_loader(
        self,
        dataset: data.Dataset,
        batch_size: int = 10,
        seed: int = 0,
    ) -> DataLoader:
        cond_set = data.DeterministicShuffle(dataset, seed=0)
        return DataLoader(cond_set, batch_size=batch_size)

    def _get_cond_data(
        self,
        name: str,
        dataset: data.Dataset,
        force: bool = False,
    ) -> ConditionData:
        cache_file = os.path.join(self.cache_dir, f"{name}_data.pickle")

        if not os.path.exists(cache_file) or force:
            cond_data = ConditionData.from_dataset(
                self.flow,
                self.explained_classifier,
                self.test_set,
                self.dataset_name,
                self.device,
            )
            cond_data.save(cache_file)
        else:
            cond_data = ConditionData.load(self.explained_classifier, cache_file)
        return cond_data

    def get_task_data(self, force: bool = False) -> ConditionData:
        return self._get_cond_data("task", self.test_set, force)

    def get_treatment_data(self, force: bool = False) -> ConditionData:
        return self._get_cond_data("treatment", self.val_set, force)

    def load_task_data(self, force: bool = False):
        self.task = self.get_task_data()

    def load_treatment_data(self, force: bool = False):
        self.treatment = self.get_treatment_data()

    def sync_figure_dir(
        self,
        name: str,
        dry: bool = False,
    ):
        self.sync_dir(os.path.join("figures", name), dry)

    def sync_dir(
        self,
        path: str,
        dry: bool = False,
    ):
        pass


@attr.s(auto_attribs=True)
class Figure:
    analysis: Analysis
    name: str
    force_clean: bool = True

    def __attrs_post_init__(self):
        if self.force_clean and os.path.exists(self.dirname):
            shutil.rmtree(self.dirname)
        print("create", self.dirname)
        os.makedirs(self.dirname, exist_ok=True)

    @property
    def dirname(self) -> str:
        return os.path.join(self.analysis.figure_dir, self.name)

    @property
    def rel_dirname(self) -> str:
        return os.path.relpath(self.dirname, self.analysis.output_dir)

    def savefig_pgf(
        self,
        figure_name: str,
        figure: mpl.figure.Figure,
        pdf: bool = True,
        sync: bool = True,
        **kwargs: Any,
    ):
        fname = os.path.join(self.dirname, figure_name + ".pgf")
        figures.savefig_pgf(fname, figure, pdf=pdf)
        if sync:
            self.sync()

    def sync(self):
        self.analysis.sync_dir(self.rel_dirname)


@attr.s(auto_attribs=True)
class UserStudy(Figure):
    cond_data: ConditionData
    n_images: int = 30
    force_clean: bool = True

    @property
    def class_name(self) -> str:
        return self.analysis.explained_classifier.config.label_name

    @property
    def dirname(self) -> str:
        return os.path.join(self.analysis.user_study_dir, self.name)


@dataclasses.dataclass
class WeightInterpolationResult:
    cf_logits: list[float]
    fig_path: str
    info: list[interpolations.SampleGridRow]
    logits: list[list[float]]
    logit_diff_mean: float
    logit_diff_max: float

    def save(self, output_dir: str) -> str:
        base, _ = os.path.splitext(self.fig_path)
        fname = os.path.join(output_dir, base + ".json")
        with open(fname, "w") as f:
            json.dump(dataclasses.asdict(self), f)
        return fname


@attr.s(auto_attribs=True)
class WeightInterpolation(UserStudy):
    name: str = "weight_interpolation"
    force_clean: bool = True

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.n_bins = len(self.cond_data.cf_values)
        self.class_idx = self.cond_data.classifier
        self.device = self.analysis.device

    def __call__(
        self,
        sync: bool = True,
        show: bool = False,
        cf_values: Optional[np.ndarray] = None,
        marker: Optional[str] = None,
    ) -> WeightInterpolationResult:
        classifier = self.analysis.explained_classifier
        flow_w_cls = self.analysis.flow_w_cls
        if cf_values is not None and marker is None:
            raise ValueError("cf_values changed but no marker given")

        cf_values_ = utils.get(cf_values, self.cond_data.cf_values)
        del cf_values
        # cf_values = cf_values_
        cf_bins = self.cond_data.cf_bins

        n_rows = self.n_images // self.n_bins
        with torch.no_grad():
            np.random.seed(0)
            imgs_real_grid, info = interpolations.sample_real_grid(
                self.cond_data,
                n_images=n_rows,
                bins=self.cond_data.cf_bins,
                selected_bins=[[0], [1], [2], [3], [4]],
                permuted_columns=False,
            )

        n, b, c, h, w = imgs_real_grid.shape
        imgs = imgs_real_grid.view(n * b, c, h, w)

        # grid_idx = np.arange(5)[None].repeat(n_rows, axis=0)
        # grid_idx = grid_idx.reshape(-1)

        with torch.no_grad():
            imgs_inter = interpolations.transform_zs(
                self.analysis.flow,
                imgs.to(self.device),
                classifier.activation_idx,
                transform=lambda z: interpolations.get_weight_interpolations(
                    z,
                    classifier.fc.weight,
                    classifier.fc.bias,
                    cf_values_,
                    relative=False,
                ),
            )

            logit_grid = flow_w_cls(imgs_inter.to(self.device))[:, 0]

            b, c, h, w = imgs_inter.shape
            imgs_inter = imgs_inter.view(self.n_images, len(cf_values_), c, h, w)
            logit_grid = logit_grid.view(self.n_images, len(cf_values_))

            logit = flow_w_cls(imgs.to(self.device))[:, 0]
            bin_idxs = np.digitize(logit.cpu().numpy(), cf_bins[1:-1])
            bin_idxs = torch.from_numpy(bin_idxs).to(self.device)

            # double check logits
            n, b, c, h, w = imgs_inter.shape
            logits = flow_w_cls(imgs_inter.view(n * b, c, h, w))
            logits = logits.view(n, b)
            cf_torch = torch.from_numpy(cf_values_[None]).to(self.device)
            logit_diff_mean = (logits - cf_torch).abs().mean().item()
            logit_diff_max = (logits - cf_torch).abs().max().item()

        sizes = figures.calculate_sizes(
            figures.get_figure_size(0.33)[0],
            len(imgs),
            6,
            label_size=0.0,
            pad_ratio_h=0.06,
            pad_ratio_w=0.0,
        )
        fig, axes = figures.plot_grid(
            imgs_inter.permute(0, 1, 2, 3, 4)[:, :],
            sizes=sizes,
            borderwidth=0.33,
            fontsize=10,
        )
        for row_idx, (axes_row, bin_idx, logit_row) in enumerate(
            zip(axes, bin_idxs, logit_grid)
        ):
            for ax, l in zip(axes_row, logit_row):
                [i.set_linewidth(0.1) for i in ax.spines.values()]

        fig.set_dpi(600)
        if marker is None:
            fig_fname = os.path.join(self.dirname, f"{self.class_name}.png")
        else:
            fig_fname = os.path.join(self.dirname, f"{marker}_{self.class_name}.png")
        if sync:
            fig.savefig(fig_fname, bbox_inches="tight")
            self.sync()
        if show:
            plt.show()
        else:
            plt.close(fig)

        return WeightInterpolationResult(
            cf_values_.tolist(),
            fig_path=os.path.relpath(fig_fname, self.analysis.output_dir),
            info=info,
            logits=logits.cpu().detach().numpy().tolist(),
            logit_diff_mean=logit_diff_mean,
            logit_diff_max=logit_diff_max,
        )


@dataclasses.dataclass
class BaselineResult:
    cf_bins: list[float]
    fig_path: str
    info: list[interpolations.SampleGridRow]

    def save(self, output_dir: str) -> str:
        base, _ = os.path.splitext(self.fig_path)
        fname = os.path.join(output_dir, base + ".json")
        with open(fname, "w") as f:
            json.dump(dataclasses.asdict(self), f)
        return fname


@attr.s(auto_attribs=True)
class Baseline(UserStudy):
    name: str = "baseline"

    def __call__(
        self,
        show: bool = False,
        sync: bool = True,
        cf_bins: Optional[np.ndarray] = None,
        marker: Optional[str] = None,
    ) -> BaselineResult:
        if cf_bins is not None and marker is None:
            raise ValueError("cf_bins changed but no marker given")
        cf_bins_ = utils.get(cf_bins, self.cond_data.cf_bins)
        del cf_bins

        with torch.no_grad():
            imgs_real_grid, info = interpolations.sample_real_grid(
                self.cond_data,
                n_images=30,
                bins=cf_bins_,
                selected_bins=[[0], [1], [2], [3], [4]],
                permuted_columns=False,
                seed=0,
            )

        sizes = figures.calculate_sizes(
            figures.get_figure_size(0.33)[0],
            len(imgs_real_grid),
            6,
            label_size=0.0,
            pad_ratio_h=0.06,
            pad_ratio_w=0.0,
        )
        fig, axes = figures.plot_grid(
            imgs_real_grid.permute(0, 1, 2, 3, 4)[:, :],
            sizes=sizes,
            borderwidth=0.33,
            fontsize=10,
        )
        for ax_row in axes:
            for ax in ax_row:
                [i.set_linewidth(0.1) for i in ax.spines.values()]

        fig.set_dpi(600)

        if marker is None:
            fig_fname = os.path.join(self.dirname, "baseline.png")
        else:
            fig_fname = os.path.join(self.dirname, f"baseline_{marker}.png")

        if sync:
            fig.savefig(fig_fname, bbox_inches="tight")
            print("saved at", os.path.join(self.dirname, "baseline.png"))
            self.sync()

        if show:
            plt.show()
        else:
            plt.close(fig)

        return BaselineResult(
            cf_bins=cf_bins_.tolist(),
            fig_path=os.path.relpath(fig_fname, self.analysis.output_dir),
            info=info,
        )
