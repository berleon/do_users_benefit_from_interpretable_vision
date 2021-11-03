"""module for prototypes implementation."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import os
import pickle
import shutil
from typing import Iterator, Mapping, Optional, OrderedDict, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.segmentation
import sklearn.decomposition as sklearn_pca
from sklearn.preprocessing import MinMaxScaler
import spotlight.evaluation
import spotlight.factorization.explicit
import spotlight.interactions
import torch
from torch import nn
import torch.nn.functional as F
import tqdm.auto
import tqdm.auto as tqdm_auto
import wandb


from dubfiv import analysis as analysis_mod
from dubfiv import classifiers as classifiers_mod
from dubfiv import config as config_mod
from dubfiv import figures
from dubfiv import flow as flow_mod
from dubfiv import train as train_mod
from dubfiv import utils


# Util Functions


def get_all_layers_with_resolution(
    model: flow_mod.SequentialFlow,
    resolution: int,
    start_resolution: int = 128,
    layer_type: Optional[Type[flow_mod.FlowModule]] = None,
) -> list[tuple[int, flow_mod.FlowModule]]:
    layers = []
    current_resolution = start_resolution
    for idx, layer in enumerate(model.layers):
        if current_resolution == resolution:
            layers.append((idx, layer))
        if isinstance(layer, flow_mod.Psi):
            current_resolution = current_resolution // layer.block_size
    if layer_type is not None:
        return [(idx, layer) for idx, layer in layers if isinstance(layer, layer_type)]
    else:
        return layers


def get_activations(
    analysis: analysis_mod.Analysis,
    act_idxs: list[int],
    batch_size: int = 1000,
) -> Iterator[tuple[int, np.ndarray]]:
    cache: dict[int, list[np.ndarray]] = {act_idx: [] for act_idx in act_idxs}
    end_idx = max(act_idxs)
    with torch.no_grad():
        pbar = tqdm.auto.tqdm(analysis.val_loader)
        for batch_idx, (imgs, labels) in enumerate(pbar):
            labels = labels.to(analysis.device)
            for act_idx, (z, zs, jac) in analysis.flow.get_activations(
                imgs.to(analysis.device),
                end_idx=end_idx,
                recorded_layers=act_idxs,
            ).items():
                b, c, h, w = z.shape
                z_flat = z.permute(0, 2, 3, 1).reshape(b * h * w, c)
                cache[act_idx].append(z_flat.detach().cpu().numpy())
            bs, ch = imgs.shape[:2]

            for act_idx, cached_act in cache.items():
                n_samples = sum([len(act) for act in cached_act])
                if n_samples >= batch_size:
                    yield act_idx, np.concatenate(cached_act)
                    cache[act_idx] = []


def get_mf(
    analysis: analysis_mod.Analysis,
    act_idxs: list[int],
    n_components: int,
    n_samples: int = 500,
    verbose: bool = False,
) -> dict[int, nn.Linear]:

    assert len(act_idxs) == 1

    def activation_to_dataset(act: np.ndarray) -> spotlight.interactions.Interactions:
        n_samples = act.shape[0]
        n_channels = act.shape[1]

        channel_ids = []
        sample_ids = []
        act_list = []

        for ch_idx in range(n_channels):
            for sample_idx in range(n_samples):
                channel_ids.append(ch_idx)
                sample_ids.append(sample_idx)
                act_list.append(act[sample_idx, ch_idx])
        return spotlight.interactions.Interactions(
            user_ids=np.array(sample_ids),
            item_ids=np.array(channel_ids),
            ratings=np.array(act_list),
        )

    with torch.no_grad():
        act_gen = iter(
            get_activations(
                analysis,
                act_idxs,
                batch_size=n_samples,
            )
        )
        idx, act_train = next(act_gen)
        idx, act_test = next(act_gen)
        act_train = act_train[:n_samples]
        act_test = act_test[:n_samples]
        train = activation_to_dataset(act_train)
        test = activation_to_dataset(act_test)

    torch.set_grad_enabled(True)

    mf = spotlight.factorization.explicit.ExplicitFactorizationModel(
        embedding_dim=n_components
    )
    mf.fit(train)
    print(spotlight.evaluation.rmse_score(mf, test))
    return {act_idxs[0]: mf}


def compute_mf_prototypes(
    analysis: analysis_mod.Analysis,
    config: PrototypesConfig,
) -> Prototypes:
    assert config.n_components is not None
    # n_components = config.n_components

    def mf_to_linear(
        mf: spotlight.factorization.explicit.ExplicitFactorizationModel,
    ) -> nn.Linear:
        assert config.n_components is not None
        w = mf._net.item_embeddings.weight
        linear = nn.Linear(mf._num_items, config.n_components)
        linear.weight.data = w.t().clone()
        # omit biases: the bias have num_items size and not n_components
        # b = mf._net.item_biases.weight
        # linear.bias.data = b.clone()
        return linear

    act_idx = config.recorded_activation(analysis.flow)
    mfs = get_mf(
        analysis,
        act_idxs=[act_idx],
        n_components=config.n_components,
        n_samples=config.fit_on_n_samples,
        verbose=True,
    )
    mf_linear = mf_to_linear(mfs[act_idx])
    return Prototypes(
        mf_linear,
        config.n_components,
    )


def get_pca_linear_params(
    pca: sklearn_pca.PCA,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pca_components = torch.from_numpy(pca.components_).float()
    pca_components[-1] = 0
    pca_components = pca_components
    pca_mean = torch.from_numpy(pca.mean_).float()
    weight = pca_components.t()
    bias = -pca_mean @ pca_components.t()
    if device is not None:
        return weight.to(device), bias.to(device)
    else:
        return weight, bias


def get_pca_linear_layer(
    pca: sklearn_pca.PCA,
    device: torch.device = None,
) -> nn.Linear:
    pca_weight, pca_bias = get_pca_linear_params(pca, device)
    linear = nn.Linear(len(pca_weight), len(pca_weight[0]))
    linear.weight.data = pca_weight.t()
    linear.bias.data = pca_bias
    linear.to(device)
    return linear


def get_pcas(
    analysis: analysis_mod.Analysis, act_idxs: list[int], n_batches: int = 200
) -> dict[int, sklearn_pca.PCA]:
    pcas = {}
    for act_idx in act_idxs:
        pcas[act_idx] = sklearn_pca.IncrementalPCA()

    cache: dict[int, sklearn_pca.PCA] = {key: [] for key in pcas.keys()}
    end_idx = max(act_idxs)
    with torch.no_grad():
        pbar = tqdm.auto.tqdm(analysis.val_loader)
        for batch_idx, (imgs, labels) in enumerate(pbar):
            labels = labels.to(analysis.device)
            for act_idx, (z, zs, jac) in analysis.flow.get_activations(
                imgs.to(analysis.device),
                end_idx=end_idx,
                recorded_layers=act_idxs,
            ).items():
                b, c, h, w = z.shape
                act_channels = z.shape[1]
                pca_idx: int = act_idx
                z_flat = z.permute(0, 2, 3, 1).reshape(b * h * w, c)
                cache[pca_idx].append(z_flat.detach().cpu().numpy())

            bs, ch = imgs.shape[:2]
            if len(cache[pca_idx]) * bs > 2 * act_channels:
                for pca_idx, cached_act in cache.items():
                    z_sub = np.concatenate(cached_act)
                    pcas[pca_idx].fit(z_sub)
                    cache[pca_idx] = []

            if batch_idx > n_batches:
                break
    return pcas


@dataclasses.dataclass(frozen=True)
class PrototypesConfig:
    layer_resolution: int = 4
    conv1x1_location: float = 0
    n_components: Optional[int] = None
    fit_on_n_samples: int = 5000
    out_features: int = 1
    method: str = "pca"  # either `pca` or `mf`

    def get_prototype_layer_idx(self, model: flow_mod.SequentialFlow) -> int:
        layers = get_all_layers_with_resolution(
            model,
            self.layer_resolution,
            layer_type=flow_mod.Conv1x1Inv,
        )
        layer_idx = int((len(layers) - 1) * self.conv1x1_location)
        layer_idx, _ = layers[layer_idx]
        return layer_idx

    def recorded_activation(self, model: flow_mod.SequentialFlow) -> int:
        return self.get_prototype_layer_idx(model) + 1

    def hash(self) -> str:
        m = hashlib.sha256()
        m.update(str(self).encode("utf-8"))
        return m.hexdigest()

    def clone(
        self,
        layer_resolution: Optional[int] = None,
        conv1x1_location: Optional[float] = None,
        n_components: Optional[int] = None,
        fit_on_n_samples: Optional[int] = None,
        out_features: Optional[int] = None,
        method: Optional[str] = None,
    ) -> PrototypesConfig:
        return PrototypesConfig(
            utils.ifnone(layer_resolution, self.layer_resolution),
            utils.ifnone(conv1x1_location, self.conv1x1_location),
            utils.ifnone(n_components, self.n_components),
            utils.ifnone(fit_on_n_samples, self.fit_on_n_samples),
            utils.ifnone(out_features, self.out_features),
            utils.ifnone(method, self.method),
        )


def compute_pca_prototypes(
    analysis: analysis_mod.Analysis,
    config: PrototypesConfig,
) -> Prototypes:
    recorded_act = config.recorded_activation(analysis.flow)
    batch_size = analysis.train_loader.batch_size
    assert isinstance(batch_size, int)
    pcas = get_pcas(
        analysis,
        act_idxs=[recorded_act],
        n_batches=config.fit_on_n_samples // batch_size,
    )
    pca = pcas[recorded_act]
    return Prototypes(
        get_pca_linear_layer(pca),
        config.n_components,
        config.out_features,
    )


def compute_prototypes(
    analysis: analysis_mod.Analysis,
    config: PrototypesConfig,
) -> Prototypes:
    if config.method == "mf":
        return compute_mf_prototypes(analysis, config)
    elif config.method == "pca":
        return compute_pca_prototypes(analysis, config)
    else:
        raise ValueError()


class Prototypes(nn.Module):
    def __init__(
        self,
        prototype_layer: nn.Linear,
        n_components: Optional[int] = None,
        out_features: int = 1,
    ):
        super().__init__()
        self.out_features = out_features
        if n_components is None:
            self.n_components = prototype_layer.out_features
        else:
            self.n_components = n_components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Linear(self.n_components, self.out_features)

        # a little hack so that prototype_layer does not show up in self.parameters()
        self._prototype_layer = [prototype_layer]

        self.metrics: Optional[classifiers_mod.BinaryLogitAccumulator] = None

    def state_dict(  # type: ignore
        self,
        destination: nn.Module.T_destination = None,  # type: ignore
        prefix: str = "",
        keep_vars: bool = False,
    ) -> collections.OrderedDict[str, torch.Tensor]:
        state = dict(super().state_dict(destination, prefix, keep_vars))  # type: ignore
        state.update(
            self.prototype_layer.state_dict(
                destination=destination, prefix="prototypes", keep_vars=keep_vars
            )
        )  # type: ignore

        state["n_components"] = self.n_components
        state["out_features"] = self.out_features
        state["prototype_layer_in"] = self.prototype_layer.in_features
        state["prototype_layer_out"] = self.prototype_layer.out_features
        return state  # type: ignore

    @property
    def prototype_layer(self) -> nn.Linear:
        return self._prototype_layer[0]

    def to(self, device: torch.device):  # type: ignore
        super().to(device)
        self.prototype_layer.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        embedding = self.prototype_layer(x.permute(0, 2, 3, 1).reshape(b * h * w, c))
        embedding_ch = embedding.shape[1]
        embedding = embedding.reshape(b, h, w, embedding_ch).permute(0, 3, 1, 2)
        # maybe use less dimensions
        embedding = embedding.contiguous()[:, : self.n_components]
        avg_pooled = self.avg_pool(embedding)
        # remove last dimensions with (1, 1)
        avg_pooled = avg_pooled[:, :, 0, 0]
        logits = self.layer(avg_pooled)
        return logits

    def loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = list(self.parameters())[0].device
        target = target.to(device)

        logits = self(x)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )
        if self.metrics is not None:
            self.metrics.add(bce, logits, target)

        self._loss = bce
        return bce

    def record_metrics(self, enable: bool = True):
        if enable:
            self.metrics = classifiers_mod.BinaryLogitAccumulator()
        else:
            self.metrics = None

    def get_recorded_metrics(self) -> dict[str, float]:
        if self.metrics is None:
            raise ValueError("No recorded metrics!")

        return self.metrics.get_metrics()

    def load_state_dict(self, state: OrderedDict[str, torch.Tensor]):  # type: ignore
        prototype_state = collections.OrderedDict(
            {
                k[len("prototypes") :]: v
                for k, v in state.items()
                if k.startswith("prototypes")
            }
        )
        self.prototype_layer.load_state_dict(prototype_state)

        super().load_state_dict(
            collections.OrderedDict(
                {
                    k[len("layer.") :]: v
                    # k: v
                    for k, v in state.items()
                    if k.startswith("layer")
                }
            ),
            strict=False,
        )


@dataclasses.dataclass
class PrototypeScorer:
    config: PrototypesConfig
    prototypes: Prototypes
    analysis: analysis_mod.Analysis

    def __post_init__(self):
        self.flow = self.analysis.flow
        self.recorded_activation: int = self.config.recorded_activation(self.flow)

    def score(
        self,
        images: torch.Tensor,
        n_components: Optional[int] = None,
    ) -> PrototypeScore:
        flow = self.analysis.flow
        # cl_activation_idx = self.analysis.explained_classifier.activation_idx
        device = self.analysis.device
        proto_layer = self.prototypes.prototype_layer
        proto_layer.to(device)
        acts = flow.get_activations(
            images.to(device),
            end_idx=self.recorded_activation,
        )

        z_proto, _, _ = acts[self.recorded_activation]

        # shape: [out, in]
        max_n_concepts = proto_layer.weight.shape[0]

        similarity = torch.einsum("bchw, kc -> bkhw", z_proto, proto_layer.weight)

        b, c, h, w = z_proto.shape
        concept_importances = []
        for idx_component in range(utils.ifnone(n_components, max_n_concepts)):
            prototype = proto_layer.weight[idx_component, :, None, None]
            prototype_repeated = prototype.repeat(b, 1, h, w)
            logits, concept_importance = torch.autograd.functional.jvp(
                lambda z: self.analysis.flow_w_cls(
                    z, start_layer=self.recorded_activation
                ),
                inputs=z_proto,
                v=prototype_repeated,
            )
            concept_importances.append(concept_importance)

        return PrototypeScore(
            similarity.cpu().numpy(),
            torch.cat(concept_importances, dim=1).cpu().numpy(),
            logits.cpu().numpy(),
            images.cpu().numpy(),
        )

    def collect_scores_from_loader(
        self,
        n_samples: int = 300,
        n_components: Optional[int] = None,
        split: str = "val",
    ) -> PrototypeScore:
        scores = []
        n_seen = 0
        if n_components is None:
            assert self.config.n_components is not None
            n_comp = self.config.n_components
        else:
            n_comp = n_components
        for imgs, labels in self.analysis.val_loader:
            with torch.no_grad():
                scores.append(self.score(imgs, n_components=n_comp))
            n_seen += len(imgs)
            if n_seen >= n_samples:
                break
        return PrototypeScore.cat(scores)


@dataclasses.dataclass
class TrainPrototypes:
    config: PrototypesConfig
    prototypes: Prototypes
    analysis: analysis_mod.Analysis

    wandb_mode: str = "online"
    _is_wandb_init: bool = False
    log: train_mod.LogHistory = dataclasses.field(default_factory=train_mod.LogHistory)

    def __post_init__(self):
        self.flow: flow_mod.SequentialFlow = self.analysis.flow
        self.recorded_activation: int = self.config.recorded_activation(self.flow)
        self.prototypes.to(self.analysis.device)

    def wandb_init(self):
        exp_config = self.analysis.ex.cfg
        config_dict = utils.strict_union(
            dict(run_name=exp_config.dirname()), dataclasses.asdict(self.config)
        )
        self._wandb = wandb.init(  # type: ignore
            project="dubfiv_prototypes",
            entity=utils.WANDB_ENTITY,
            name=exp_config.unique_marker,
            dir=config_mod.resolve_path("wandb"),
            mode=self.wandb_mode,
            config=config_dict,
            reinit=True,
        )

    @property
    def device(self) -> torch.device:
        return self.analysis.device

    @staticmethod
    def get_lr_factor(step: int, n_total_samples: int) -> float:
        def cos_to_zero_one(x: float) -> float:
            return float((np.cos(np.pi * x) + 1) / 2)

        progress = step / n_total_samples
        return cos_to_zero_one(progress)

    def fit(
        self,
        n_epochs: int = 1,
        n_samples: int = 10_000_000 * 10_000_000,
        lr_scheduling: bool = True,
        use_wandb: bool = True,
    ):
        torch.set_grad_enabled(True)

        if use_wandb and not self._is_wandb_init:
            self.wandb_init()
            self._is_wandb_init = True

        train_loader = self.analysis.train_loader

        lr = 0.1
        opt = torch.optim.SGD(self.prototypes.parameters(), lr=lr, momentum=0.8)

        assert train_loader.batch_size
        n_total_batches = min(
            n_epochs * len(train_loader), n_samples // train_loader.batch_size
        )
        batch_idx = 0
        n_samples_seen = 0
        for _ in range(n_epochs):
            pbar = tqdm_auto.tqdm(train_loader)
            for imgs, labels in pbar:
                opt.zero_grad()
                if lr_scheduling:
                    reduced_lr = lr * self.get_lr_factor(batch_idx, n_total_batches)
                    opt.param_groups[0]["lr"] = reduced_lr
                    self.log.log("lr", reduced_lr)
                with torch.no_grad():
                    zs, jac = self.flow(
                        imgs.to(self.device), end_idx=self.recorded_activation
                    )

                loss = self.prototypes.loss(zs[-1].clone(), labels.to(self.device))
                loss.backward()
                opt.step()

                batch_idx += 1
                n_samples_seen += len(imgs)
                # logging
                self.log.log("train.loss", loss)
                self.log.increase_step(len(imgs))
                pbar.set_postfix(loss=loss.item(), lr=opt.param_groups[0]["lr"])
                if n_samples_seen > n_samples:
                    break
            self.log.mark_epoch_end()
            metrics = self.evaluate(split="val")
            self.log.logs("val", metrics)

            if n_samples_seen > n_samples:
                break

    def evaluate(self, split: str = "test") -> dict[str, float]:
        if split == "test":
            loader = self.analysis.test_loader
        elif split == "val":
            loader = self.analysis.val_loader
        else:
            raise ValueError()

        pbar = tqdm_auto.tqdm(loader)
        for imgs, labels in pbar:
            self.prototypes.record_metrics()
            with torch.no_grad():
                zs, jac = self.flow(
                    imgs.to(self.device), end_idx=self.recorded_activation
                )
                self.prototypes.loss(zs[-1], labels.to(self.device))
        metrics = self.prototypes.get_recorded_metrics()
        return metrics


@dataclasses.dataclass
class SplittedContribution:
    component: int
    direction: str
    similarity: np.ndarray
    concept_importance: np.ndarray
    contribution: np.ndarray
    image_idx: np.ndarray

    @property
    def is_positive(self) -> bool:
        return self.direction == "positive"

    @property
    def is_negative(self) -> bool:
        return not self.is_positive


@dataclasses.dataclass
class PrototypeScore:
    similarity: np.ndarray
    concept_importance: np.ndarray
    logits: np.ndarray
    images: np.ndarray

    @staticmethod
    def cat(scores: list[PrototypeScore]) -> PrototypeScore:
        return PrototypeScore(
            np.concatenate([s.similarity for s in scores]),
            np.concatenate([s.concept_importance for s in scores]),
            np.concatenate([s.logits for s in scores]),
            np.concatenate([s.images for s in scores]),
        )

    def get_contribution(self, image_idx: int, concept_idx: int) -> torch.Tensor:
        similarity = self.similarity[image_idx, concept_idx].sum()
        importance = self.concept_importance[image_idx, concept_idx]
        return similarity * importance

    @property
    def n_components(self) -> int:
        return min(self.similarity.shape[1], self.concept_importance.shape[1])

    @staticmethod
    def _reduce(
        value: np.ndarray,
        reduce: str,
        axis: tuple[int, ...],
    ) -> np.ndarray:
        if reduce == "sum":
            return np.array(value.sum(axis=axis))
        elif reduce == "max":
            return np.array(value.max(axis=axis))
        else:
            raise ValueError()

    def get_all_contribution(
        self,
        n_components: Optional[int] = None,
        reduce: str = "sum",
    ) -> np.ndarray:
        if n_components is None:
            n_components = self.n_components
        similarity = self._reduce(
            self.similarity[:, :n_components], reduce, axis=(2, 3)
        )

        concept_importance = self.concept_importance[:, :n_components]
        contribution = similarity * concept_importance
        return contribution

    def get_splitted_contribution(
        self,
        n_components: Optional[int] = None,
        reduce: str = "sum",
    ) -> list[SplittedContribution]:
        if n_components is None:
            n_components = self.n_components
        concept_importance = self.concept_importance[:, :n_components]
        contribution = self.get_all_contribution(n_components, reduce)
        split_value = np.percentile(contribution, 50)
        pos = contribution > split_value
        neg = contribution < split_value
        splitted_contribs = []
        for i in range(n_components):
            for direction, neg_pos in [("positive", pos), ("negative", neg)]:
                n_samples = np.sum(neg_pos)
                if n_samples < 5:
                    continue
                splitted_contribs.append(
                    SplittedContribution(
                        component=i,
                        direction=direction,
                        similarity=self.similarity[neg_pos[:, i], i],
                        concept_importance=concept_importance[neg_pos[:, i], i],
                        contribution=contribution[neg_pos[:, i], i],
                        image_idx=neg_pos[:, i].nonzero(),
                    )
                )

        return splitted_contribs

    def get_important_concepts(
        self,
        image_idx: int,
        n_components: Optional[int] = None,
    ) -> np.ndarray:
        if n_components is None:
            n_components = self.n_components
        max_similarity = self.similarity[image_idx, :n_components].sum(axis=(1, 2))
        concept_importance = self.concept_importance[image_idx, :n_components]
        contribution = max_similarity * concept_importance
        return np.argsort(contribution)

    def get_examples_for_concept(
        self,
        concept_idx: int,
        n_examples: int,
        lower_percentile: float = 95.0,
        upper_percentile: float = 100,
        exclude: list[int] = [],
        reduce: str = "sum",
    ) -> np.ndarray:
        similarity = self._reduce(self.similarity[:, concept_idx], reduce, axis=(1, 2))
        vmin, vmax = np.percentile(similarity, q=[lower_percentile, upper_percentile])

        mask = np.logical_and(vmin < similarity, similarity < vmax).astype(np.float32)
        mask[exclude] = 0.0
        mask /= mask.sum()
        return np.random.choice(len(self), n_examples, p=mask, replace=False)

    def get_pos_examples_for_concept(
        self,
        concept_idx: int,
        n_examples: int,
        percentile: float = 5.0,
        exclude: list[int] = [],
        reduce: str = "sum",
    ) -> np.ndarray:
        return self.get_examples_for_concept(
            concept_idx,
            n_examples,
            lower_percentile=100 - percentile,
            upper_percentile=100.0,
            exclude=exclude,
            reduce=reduce,
        )

    def get_neg_examples_for_concept(
        self,
        concept_idx: int,
        n_examples: int,
        percentile: float = 5.0,
        exclude: list[int] = [],
        reduce: str = "sum",
    ) -> np.ndarray:
        return self.get_examples_for_concept(
            concept_idx,
            n_examples,
            lower_percentile=0.0,
            upper_percentile=percentile,
            exclude=exclude,
            reduce=reduce,
        )

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.images.shape[-2], self.images.shape[-1]

    def get_image_for_plot(self, image_idx: int) -> np.ndarray:
        return np.clip(self.images[image_idx].transpose(1, 2, 0), 0, 1)

    def get_correlation(
        self,
        reduce: str = "max",
    ) -> pd.DataFrame:
        data_corr = []

        for i in range(self.n_components):
            contributions = self.get_all_contribution(reduce=reduce)
            corr = float(
                np.corrcoef(contributions[:, i].flatten(), self.logits.flatten())[1, 0]
            )
            data_corr.append(
                {
                    "Component": i,
                    "Correlation": corr,
                }
            )
        df_corr = pd.DataFrame(data_corr)
        return df_corr

    def explain_scores(
        self,
        image_idx: int,
        n_concepts: int = 3,
        n_examples: int = 4,
        percentile: float = 5.0,
        debug_info: bool = False,
    ) -> tuple[mpl.figure.Figure, list[list[mpl.axes.Axes]]]:
        pos_concepts = self.get_important_concepts(image_idx)[-n_concepts // 2 :]
        neg_concepts = self.get_important_concepts(image_idx)[: n_concepts // 2]
        concepts = np.concatenate([neg_concepts, pos_concepts])

        fig, axes = plt.subplots(
            ncols=n_examples + 2,
            nrows=n_concepts,
            squeeze=False,
            figsize=(n_examples + 2, n_concepts),
        )
        axes_proto = axes[:, 1:-1]
        axes_scores = axes[:, -1]

        axes[0, 0].set_title("{:.2f}".format(float(self.logits[image_idx])))
        for ax, concept_idx in zip(axes[:, 0], concepts):
            ax.set_ylabel(concept_idx, rotation=90)
            ax.imshow(self.get_image_for_plot(image_idx))
            add_alpha_mask(
                ax, self.similarity[image_idx, concept_idx], self.image_shape
            )

        for ax_row, concept_idx in zip(axes_proto, concepts):
            if concept_idx in pos_concepts:
                examples_idx = self.get_pos_examples_for_concept(
                    concept_idx, n_examples, exclude=[image_idx], percentile=percentile
                )
            else:
                examples_idx = self.get_neg_examples_for_concept(
                    concept_idx, n_examples, exclude=[image_idx], percentile=percentile
                )

            for ax, example_idx in zip(ax_row, examples_idx):
                ax.imshow(self.get_image_for_plot(example_idx))

                add_alpha_mask(
                    ax, self.similarity[example_idx, concept_idx], self.image_shape
                )

                for spine in ax.spines.values():
                    spine.set_edgecolor(
                        "blue" if concept_idx in pos_concepts else "red"
                    )

            if debug_info:
                contribution = self.get_contribution(image_idx, concept_idx)
                ax_row[0].set_title(concept_idx)
                ax_row[1].set_title("{:.2f}".format(float(contribution)))

        for ax, concept_idx in zip(axes_scores, concepts):
            contribution = self.get_contribution(image_idx, concept_idx)
            ax.text(
                0.5,
                0.5,
                "{:.2f}".format(float(contribution)),
                horizontalalignment="center",
                verticalalignment="center",
            )

        for ax in axes.flatten():
            ax.set_yticks([])
            ax.set_xticks([])

        return fig, axes


def get_alpha_mask(
    similarity: np.ndarray,  # [h, w]
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()

    scaled_sim = scaler.fit_transform(similarity.flatten().reshape(-1, 1))
    scaled_sim = scaled_sim.reshape(*similarity.shape)

    mask = torch.nn.functional.interpolate(
        torch.from_numpy(scaled_sim)[None, None],
        size=image_shape,
        mode="bicubic",
        align_corners=True,
    ).numpy()[0, 0]
    binary_mask = mask < 0.5
    boundaries = skimage.segmentation.find_boundaries(binary_mask + 1)
    return binary_mask.astype(np.float32), boundaries.astype(np.float32)


def add_alpha_mask(
    ax: mpl.axes.Axes,
    similarity: np.ndarray,
    image_shape: tuple[int, int],
):
    alpha_mask, boundaries = get_alpha_mask(similarity, image_shape)
    ax.imshow(alpha_mask, cmap="binary", alpha=0.3 * alpha_mask)
    ax.imshow(0.8 * boundaries, cmap="Reds", alpha=boundaries, vmin=0, vmax=1)


def visualize_prototypes(
    # train_proto: TrainPrototypes,
    scores: PrototypeScore,
    n_prototypes: int,
    n_examples: int = 5,
) -> tuple[mpl.figure.Figure, list[list[mpl.axes.Axes]]]:
    fig, axes = plt.subplots(
        ncols=2 * n_examples,
        nrows=n_prototypes,
        squeeze=False,
        figsize=(2 * n_examples, n_prototypes),
    )
    # fig.set_dpi(150)

    for idx_proto in range(n_prototypes):
        b, c, h, w = scores.similarity.shape

        max_per_sample = np.max(
            scores.similarity[:, idx_proto].reshape(b, h * w), axis=1
        )
        max_idx = np.argsort(max_per_sample)
        scaler = MinMaxScaler()
        scaler.fit(scores.similarity[:, idx_proto].flatten().reshape(-1, 1))
        neg_indicies = max_idx[:n_examples]
        pos_indicies = max_idx[-n_examples:]
        indicies = np.concatenate([neg_indicies, pos_indicies])

        axes_row = axes[idx_proto]
        axes_row[0].set_ylabel(idx_proto)

        for ax, idx in zip(axes_row.flatten(), indicies):
            ax.imshow(np.clip(scores.images[idx].transpose(1, 2, 0), 0, 1))

            similarity = scores.similarity[idx, idx_proto]
            # if idx in neg_indicies:
            #     similarity = - similarity

            add_alpha_mask(ax, similarity, scores.image_shape)

            for spine in ax.spines.values():
                spine.set_edgecolor("blue" if idx in pos_indicies else "red")

            ax.set_yticks([])
            ax.set_xticks([])
    return fig, axes


def grid_visualization_pos_neg(
    # train_proto: TrainPrototypes,
    scores: PrototypeScore,
    prototypes: list[int],
    n_examples: int = 5,
    percentile: float = 5.0,
    reduce: str = "sum",
    swap_neg: bool = False,
    contributions: Optional[np.ndarray] = None,
) -> tuple[mpl.figure.Figure, list[list[mpl.axes.Axes]]]:
    n_prototypes = len(prototypes)
    fig, axes = plt.subplots(
        ncols=2 * n_examples + 1,
        nrows=n_prototypes,
        squeeze=False,
        figsize=(2 * n_examples + 1, n_prototypes),
    )
    # fig.set_dpi(150)

    # omit one column in the center
    axes_grid = np.concatenate(
        [axes[:, :n_examples], axes[:, n_examples + 1 :]], axis=1
    )

    for ax in axes[:, n_examples]:
        ax.axis("off")
        ax.set_yticks([])
        ax.set_xticks([])

    for idx, (idx_proto, axes_row) in enumerate(zip(prototypes, axes_grid)):
        b, c, h, w = scores.similarity.shape

        neg_indicies = scores.get_neg_examples_for_concept(
            idx_proto, n_examples, percentile, reduce=reduce
        )
        pos_indicies = scores.get_pos_examples_for_concept(
            idx_proto, n_examples, percentile, reduce=reduce
        )

        if contributions is not None and contributions[idx] < 0:
            # ensures it is algined with Peaky / Stretchy
            indicies = np.concatenate([pos_indicies, neg_indicies])
        else:
            indicies = np.concatenate([neg_indicies, pos_indicies])

        axes_row[0].set_ylabel(idx_proto)

        for ax, idx in zip(axes_row.flatten(), indicies):
            ax.imshow(np.clip(scores.images[idx].transpose(1, 2, 0), 0, 1))

            similarity = scores.similarity[idx, idx_proto]
            if idx in neg_indicies and swap_neg:
                similarity = -similarity

            add_alpha_mask(ax, similarity, scores.image_shape)

            ax.set_yticks([])
            ax.set_xticks([])
    return fig, axes


def grid_visualization_single(
    # train_proto: TrainPrototypes,
    scores: PrototypeScore,
    n_prototypes: Optional[int] = None,
    n_examples: int = 5,
    percentile: float = 5.0,
    reduce: str = "sum",
    swap_neg: bool = False,
    show_info: bool = False,
) -> tuple[mpl.figure.Figure, list[list[mpl.axes.Axes]]]:

    all_contribution = scores.get_all_contribution(scores.n_components, reduce)
    contribs = scores.get_splitted_contribution(reduce=reduce)

    contribs = sorted(
        contribs,
        key=lambda c: np.abs(
            np.corrcoef(
                all_contribution[:, c.component],
                scores.logits[:, 0],
            )[0, 1]
        ),
        reverse=True,
    )
    contribs = contribs[:n_prototypes]

    n_rows = len(contribs)
    fig, axes = plt.subplots(
        ncols=n_examples,
        nrows=n_rows,
        squeeze=False,
        figsize=(n_examples, 1.1 * n_rows),
    )

    for idx, (contrib, axes_row) in enumerate(zip(contribs, axes)):
        idx_proto = contrib.component
        if contrib.direction == "negative":
            indicies = scores.get_neg_examples_for_concept(
                idx_proto, n_examples, percentile, reduce=reduce
            )
        else:
            indicies = scores.get_pos_examples_for_concept(
                idx_proto, n_examples, percentile, reduce=reduce
            )

        if show_info:
            axes_row[0].set_ylabel(f"{idx_proto}{'p' if contrib.is_positive else 'n'}")

        for ax, idx in zip(axes_row.flatten(), indicies):
            ax.imshow(np.clip(scores.images[idx].transpose(1, 2, 0), 0, 1))

            similarity = scores.similarity[idx, idx_proto]
            if contrib.direction == "negative" and swap_neg:
                similarity = -similarity

            add_alpha_mask(ax, similarity, scores.image_shape)

            ax.set_yticks([])
            ax.set_xticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.33)

    fig.subplots_adjust(wspace=0.0, hspace=0.1)
    return fig, axes


@dataclasses.dataclass(frozen=True)
class VisualConfig:
    n_examples_per_row: int = 5
    prototypes_percentile: float = 10
    reduce_method: str = "sum"
    swap_neg: bool = False

    def hash(self) -> str:
        m = hashlib.sha256()
        m.update(str(self).encode("utf-8"))
        return m.hexdigest()


@dataclasses.dataclass
class ExportPrototypes:
    analysis: analysis_mod.Analysis
    proto: Prototypes
    proto_config: PrototypesConfig = PrototypesConfig(
        layer_resolution=4,
        conv1x1_location=-2,
        n_components=5,
        fit_on_n_samples=5000,
    )
    visual_config: VisualConfig = VisualConfig()
    marker: str = ""
    overwrite_force: bool = False

    def __post_init__(self):
        if self.marker == "":
            self.marker = utils.combine_hash(
                [self.proto_config.hash(), self.visual_config.hash()]
            )[:8]

    @property
    def flow(self) -> flow_mod.SequentialFlow:
        return self.analysis.flow

    @property
    def recorded_activation(self) -> int:
        return self.proto_config.recorded_activation(self.flow)

    @staticmethod
    def estimate(
        analysis: analysis_mod.Analysis,
        proto_config: PrototypesConfig,
        visual_config: VisualConfig = VisualConfig(),
    ) -> ExportPrototypes:
        print("Activations:", proto_config.recorded_activation(analysis.flow))
        return ExportPrototypes(
            analysis,
            compute_prototypes(analysis, proto_config),
            proto_config,
            visual_config,
        )

    @property
    def name(self) -> str:
        return (
            f"prototypes_{self.proto_config.method}_"
            f"layer{self.recorded_activation}_"
            f"components{self.proto_config.n_components}_" + self.marker
        )

    @property
    def dirname(self) -> str:
        return os.path.join(self.analysis.user_study_dir, "prototypes", self.name)

    @property
    def device(self) -> torch.device:
        return self.analysis.device

    def export_prototype(
        self,
        accuracies: bool = False,
    ):
        if os.path.exists(self.dirname) and self.overwrite_force:
            shutil.rmtree(self.dirname)

        os.makedirs(self.dirname)

        if accuracies:
            self.export_linear_model_accuracy()

        proto_scorer = PrototypeScorer(self.proto_config, self.proto, self.analysis)
        scores = proto_scorer.collect_scores_from_loader()

        self.export_figure(scores)
        self.export_correlation(scores)

        with open(os.path.join(self.dirname, "scores.pickle"), "wb") as f:
            pickle.dump(scores, f)
        with open(os.path.join(self.dirname, "prototypes.pickle"), "wb") as f:
            torch.save(self.proto.state_dict(), f)

    def export_figure(
        self,
        scores: PrototypeScore,
        save: bool = True,
        show_info: bool = False,
    ) -> tuple[mpl.figure.Figure, list[list[mpl.axes.Axes]]]:

        fig, axes = grid_visualization_single(
            scores,
            self.proto_config.n_components,
            self.visual_config.n_examples_per_row,
            self.visual_config.prototypes_percentile,
            self.visual_config.reduce_method,
            self.visual_config.swap_neg,
            show_info=show_info,
        )
        fig.set_dpi(300)
        if save:
            config_name = os.path.join(self.dirname, "proto_config.json")
            with open(config_name, "w") as f:
                json.dump(dataclasses.asdict(self.proto_config), f, indent=2)

            config_name = os.path.join(self.dirname, "visual_config.json")
            with open(config_name, "w") as f:
                json.dump(dataclasses.asdict(self.visual_config), f, indent=2)
            fig_name = os.path.join(self.dirname, "proto.png")
            print(f"Saving figure to: {fig_name}")
            fig.savefig(fig_name, bbox_inches="tight", pad_inches=0.01)
            plt.close(fig)
            return fig, axes
        else:
            return fig, axes

    def export_linear_model_accuracy(self):
        val_accuracy, test_accuracy = self.get_linear_model_accuracy()
        defs: Mapping[str, str] = {
            "prototypes_linear_model_test_accuracy": f"{100 * test_accuracy:.2f}",
            "prototypes_linear_model_val_accuracy": f"{100 * val_accuracy:.2f}",
        }
        figures.export_latex_defs(
            os.path.join(self.dirname, "linear_model_accuracy.tex"), **defs
        )
        fname = os.path.join(self.dirname, "linear_model_accuracy.json")
        with open(fname, "w") as f:
            json.dump(
                dict(validation_accuracy=val_accuracy, test_accuracy=test_accuracy), f
            )

    def export_correlation(self, scores: PrototypeScore):
        df_corr = scores.get_correlation(self.visual_config.reduce_method)

        corr_table = df_corr.to_latex(index=False)
        figures.export_latex_defs(
            os.path.join(self.dirname, "concept_logit_corr.tex"),
            prototypes_concept_logit=corr_table,
        )
        with open(os.path.join(self.dirname, "concept_logit_corr.pickle"), "wb") as f:
            pickle.dump(df_corr, f)
        print("-" * 80)
        print(corr_table)
        print("-" * 80)

    def get_linear_model_accuracy(self) -> tuple[float, float]:
        train_proto = TrainPrototypes(
            self.proto_config,
            self.proto,
            self.analysis,
        )
        train_proto.fit()
        test_metric = train_proto.evaluate("test")
        val_accuracy = min(train_proto.log.data["val.accuracy"].values())
        return val_accuracy, test_metric["accuracy"]
