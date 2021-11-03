"""util functions."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import pdb
import sys
import time
import traceback
from typing import Any, Iterator, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
import torch.utils.data as torch_data
import torchvision.utils


AnyTensor = Union[np.ndarray, torch.Tensor]


def to_numpy(x: AnyTensor) -> np.ndarray:
    """Returns a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x


def numpy_grid(
    x: AnyTensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
) -> np.ndarray:
    """Wrapper around ``torchvision.utils.make_grid``."""
    if isinstance(x, np.ndarray):
        x_torch = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        x_torch = x
    grid = torchvision.utils.make_grid(x_torch, nrow, padding, normalize,
                                       value_range, scale_each, pad_value)
    return to_numpy(grid).transpose(1, 2, 0)


T = TypeVar('T')


def get(maybe_none: Optional[T], default: T) -> T:
    if maybe_none is not None:
        return maybe_none
    else:
        return default


ifnone = get


def combine_hash(hashes: list[str]) -> str:
    m = hashlib.sha256()
    for h in hashes:
        m.update(h.encode('utf-8'))
    return m.hexdigest()


def strict_union(*dicts: dict) -> dict:
    merged = {}
    for dictionary in dicts:
        for k, v in dictionary.items():
            if k in merged:
                raise ValueError(f"key {k} already exists.")
            merged[k] = copy.deepcopy(v)
    return merged


def flatten(x: Union[T, Sequence[T], Sequence[Any]]) -> list[T]:
    def flatten(x: Union[T, Sequence[T], Sequence[Any]]) -> Iterator[T]:
        if isinstance(x, (list, tuple)):
            for gen in [flatten(xi) for xi in x]:
                for xj in gen:
                    yield xj
        else:
            yield x  # type: ignore
    return list(flatten(x))


def concat_dicts(
    dicts: list[dict[str, np.ndarray]],
    axis: int = 0,
) -> dict[str, np.ndarray]:
    keys = dicts[0].keys()
    return {
        key: np.concatenate([d[key] for d in dicts], axis=0)
        for key in keys
    }


def take_n_samples(
    loader: torch_data.DataLoader,
    n_samples: int
) -> Iterator[tuple[Any, ...]]:
    if n_samples < 1:
        raise ValueError("n_samples are too small: {n_samples}")
    n_seen_samples = 0
    for batch in loader:
        imgs = batch[0]
        n_seen_samples += len(imgs)
        last_batch = n_seen_samples > n_samples

        if last_batch:
            diff = n_seen_samples - n_samples
            yield tuple(elem[:-diff] for elem in batch)
            break
        else:
            yield batch

        if n_seen_samples == n_samples:
            break


TENSOR = TypeVar('TENSOR', bound=AnyTensor)


def batchify(tensor: TENSOR, batch_size: int) -> Iterator[TENSOR]:
    """Yields tensors of batch_size."""
    n_seen_samples = 0
    while True:
        batch: TENSOR = tensor[n_seen_samples:n_seen_samples + batch_size]  # type: ignore
        yield batch
        n_seen_samples += len(batch)

        if n_seen_samples == len(tensor):
            break


@contextlib.contextmanager
def pdb_post_mortem(enable: bool = True):
    if enable:
        try:
            yield
        except Exception:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        yield


@contextlib.contextmanager
def timeit(message: str):
    start = time.time()
    yield
    print(f"[{message}] {time.time() - start}")


@contextlib.contextmanager
def one_time_hooks(layers: Union[Sequence[torch.nn.Module],
                                 torch.nn.ModuleList],
                   hook: Any):
    hooks = [layer.register_forward_hook(hook) for layer in layers]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


WANDB_ENTITY = "XXXXXXX"
