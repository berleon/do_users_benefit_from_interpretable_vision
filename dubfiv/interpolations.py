"""Module for the interpolation of classifier weights."""

from __future__ import annotations

import copy
import dataclasses
import io
import os
import shutil
import tempfile
from typing import Callable, Union

from cairosvg import svg2png
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import skimage.color
import skimage.transform
import torch

from dubfiv import analysis
from dubfiv import classifiers as classifiers_mod
from dubfiv import figures
from dubfiv import flow
from dubfiv.pytorch_helper import get_linear_layer


def add_vector(
    z: torch.Tensor, vector: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    b, c = z.shape
    w_norm = torch.norm(vector)
    z = z + (scale / w_norm) * vector.view(c).unsqueeze(0).repeat(b, 1)
    return z


def get_cosine_similarity(z: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector_norm = torch.norm(vector)[None, None]
    z_norm = torch.norm(z, dim=1)[:, None]
    return z @ vector / (vector_norm * z_norm)


def set_dot_to(
    x: torch.Tensor, vector: torch.Tensor, value: Union[torch.Tensor, float] = 0
) -> torch.Tensor:
    norm = torch.norm(vector)
    vector_norm = vector / norm
    similarity = x @ vector_norm
    o = x - similarity[:, None] * vector_norm[None].repeat(len(similarity), 1)

    if isinstance(value, torch.Tensor):
        if len(value.shape) == 1:
            value = value[:, None]

    o = o + value * vector_norm[None].repeat(len(similarity), 1) / norm
    return o


def get_weight_prototypes(weight: torch.Tensor, scale: float = None) -> torch.Tensor:
    b, c = weight.shape
    weight_norm = torch.norm(weight, dim=1)[:, None]
    if scale is None:
        scale = np.sqrt(c)
    weight_scaled = scale * weight / weight_norm
    return weight_scaled


def set_weight_bias_to(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    value: Union[float, torch.Tensor],
) -> torch.Tensor:
    # x @ w + b = v
    # / torch.norm(weight)
    return set_dot_to(x, weight, value - bias)


def remove_loc(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)


def recover_loc(x: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
    b, c, h, w = shape
    return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def get_classifier_interpolations(
    z: torch.Tensor,
    classifier: classifiers_mod.LinearClassifier,
    class_idx: int,
    values: list[float],
    relative: bool = False,
) -> list[torch.Tensor]:
    weight = classifier.fc.weight[class_idx]
    bias = classifier.fc.bias[class_idx]

    bs, ch, h, w = z.shape
    z_flat = z.view(bs, ch * h * w)

    z_off = []
    for scale in values:
        if relative:
            logits = classifier(z_flat)[:, class_idx]
            logit_offset = scale + logits
        else:
            logit_offset = scale
        z_inter = set_weight_bias_to(z_flat, weight, bias, logit_offset)
        z_off.append(z_inter.view(bs, ch, h, w))
    return z_off


def get_weight_interpolations(
    z: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    values: Union[list[float], np.ndarray],
    relative: bool = False,
) -> list[torch.Tensor]:
    bs, ch, h, w = z.shape
    z_flat = z.view(bs, ch * h * w)

    if len(weight) == 1 and len(weight.shape) == 2:
        weight = weight[0]

    if isinstance(values, list):
        val_arr = np.array(values)
    else:
        val_arr = values
    if val_arr.ndim == 1:
        val_arr = val_arr[None, :].repeat(len(z), axis=0)

    assert val_arr.shape[0] == bs

    z_grid = []
    for i in range(val_arr.shape[1]):
        z_column = []
        for j in range(bs):
            z_item = z_flat[j : j + 1]
            scale = float(val_arr[j, i])

            if relative:
                logits = get_linear_layer(weight, bias, z.device)(z_item)[:, 0]
                logit_offset = scale + logits
            else:
                logit_offset = scale
            z_inter = set_weight_bias_to(z_item, weight, bias, logit_offset)
            z_column.append(z_inter.view(1, ch, h, w))
        z_grid.append(torch.cat(z_column))
    return z_grid


def transform_zs(
    model: flow.SequentialFlow,
    imgs: torch.Tensor,
    layer_idx: int,
    transform: Callable[[torch.Tensor], list[torch.Tensor]],
) -> torch.Tensor:
    zs, _ = model(imgs, end_idx=layer_idx)
    z_weight = transform(zs[-1])
    zs_com = flow.cross_product_zs(zs, z_weight)
    imgs_inter, _ = model.inverse()(zs_com, end_idx=layer_idx)
    return imgs_inter


class Modification:
    def __call__(self):
        pass


def sample_from_bins(
    logits: np.ndarray,
    bins: np.ndarray,
    selected_bins: Union[str, list[list[int]]] = "all",
    logit_mask: bool = None,
) -> np.ndarray:
    n_bins = len(bins) - 1
    indicies = []
    if selected_bins == "all":
        selected_bins = [[i] for i in range(n_bins)]
    for bins_for_col in selected_bins:
        mask = np.zeros_like(logits, dtype=bool)
        for i in range(n_bins):
            if selected_bins != "all" and i not in bins_for_col:
                continue

            mask = np.logical_or(
                mask, np.logical_and(bins[i] <= logits, logits <= bins[i + 1])
            )
            if logit_mask is not None:
                mask = np.logical_and(mask, logit_mask)
        indicies.append(np.random.choice(len(mask), 1, p=mask / mask.sum()))

    return np.concatenate(indicies)


@dataclasses.dataclass
class SampleGridRow:
    task_id: str
    image_idx: list[int]
    logits: list[float]
    labels: list[float]


def sample_real_grid(
    task_data: analysis.ConditionData,
    n_images: int = 30,
    bins: Union[np.ndarray, int] = 6,
    selected_bins: Union[str, list[list[int]]] = "all",
    permuted_columns: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, list[SampleGridRow],]:
    np.random.seed(seed)

    classifier = copy.deepcopy(task_data.classifier)
    with torch.no_grad():
        zs_original = task_data.zs
        imgs_original = task_data.images
        labels_original = task_data.labels

        z_orig = zs_original[-1]
        b, c, h, w = z_orig.shape
        classifier.to(z_orig.device)

        logits = classifier(z_orig.view(b, c * h * w))[:, 0].numpy()

        if type(bins) == int:
            bins = np.percentile(logits, np.linspace(0, 100, bins + 1))

        imgs = []

        logit_mask = np.ones(logits.shape)
        info = []
        for task_id in range(0, n_images):
            image_idx = torch.from_numpy(
                sample_from_bins(logits, bins, selected_bins, logit_mask)
            ).long()  # type: ignore

            logit_mask[image_idx] = 0
            imgs_row = imgs_original[image_idx]
            labels_row = labels_original[image_idx]
            sel_logits = logits[image_idx]
            if permuted_columns:
                shuffle_idxs = torch.from_numpy(
                    np.random.choice(len(imgs_row), len(imgs_row), replace=False)
                )
            else:
                shuffle_idxs = torch.arange(len(imgs_row))

            imgs_row = imgs_row[shuffle_idxs]
            labels_row = labels_row[shuffle_idxs]
            sel_logits = [float(sel_logits[idx]) for idx in shuffle_idxs]
            info.append(
                SampleGridRow(
                    task_id=f"{task_id:05d}",
                    image_idx=[int(image_idx[i]) for i in shuffle_idxs],
                    logits=sel_logits,
                    labels=labels_row[:, 0].numpy().tolist(),
                )
            )
            imgs.append(imgs_row)

        return torch.stack(imgs), info
