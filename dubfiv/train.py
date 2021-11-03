#!/usr/bin/env python
"""Training script."""

from __future__ import annotations

import dataclasses
from dataclasses import field
import logging
import os
import pprint
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import warnings

import imageio
import numpy as np
import PIL.Image
import toml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import utils as torchvision_utils
import tqdm.auto as tqdm_auto
import wandb

from dubfiv import callbacks as callbacks_mod
from dubfiv import classifiers as classifiers_mod
from dubfiv import config
from dubfiv import data
from dubfiv import flow
from dubfiv import utils
from dubfiv.flow import (
    data_init,
    InverseChecker,
)

torch.backends.cudnn.benchmark = True

RunWandB = wandb.sdk.wandb_run.Run


@dataclasses.dataclass
class LogHistory(callbacks_mod.Callback):
    """Logs the history of scalars.

    Attrs:
        n_samples: number of seen samples for the training.
        epoch_ends: samples when epoch ends.
        data: stored data of keys to stored history
    """

    n_samples: int = 0
    n_batches: int = 0
    epoch_ends: List[int] = field(default_factory=list)
    data: Dict[str, Dict[int, float]] = field(default_factory=dict)
    _last_index: Dict[str, int] = field(init=False, default_factory=dict)
    use_wandb: bool = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_batches": self.n_batches,
            "data": self.data,
            "epoch_ends": self.epoch_ends,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.n_samples = state["n_samples"]
        self.n_batches = state["n_batches"]
        self.data = state["data"]
        self.epoch_ends = state["epoch_ends"]
        self._last_index.clear()

    @property
    def epoch(self) -> int:
        """Number of full loops over the dataset."""
        return len(self.epoch_ends)

    def log(self, key: str, value: Union[torch.Tensor, float, np.ndarray]):
        """Logs ``value`` for ``key``."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, np.ndarray):
            value = float(value)

        if key not in self.data:
            self.data[key] = {}

        if self.n_samples in self.data[key]:
            raise ValueError("Already logged a value for this iteration.")
        self.data[key][self.n_samples] = value
        self._last_index[key] = self.n_samples
        if self.use_wandb:
            try:
                wandb.log({key: value}, step=self.n_samples)
            except OSError as e:
                warnings.warn(e.strerror)

    def logs(self, key: str, metrics: Dict[str, float]):
        for sub_key, value in metrics.items():
            self.log(f"{key}.{sub_key}", value)

    def mark_epoch_end(self):
        self.epoch_ends.append(self.n_samples)

    def log_image(self, key: str, filename: str):
        img = PIL.Image.open(filename)
        if self.use_wandb:
            try:
                wandb.log({key: wandb.Image(img)}, step=self.n_samples)
            except OSError as e:
                warnings.warn(e.strerror)

    def on_batch_end(self, step: int, train: bool):
        """Increments sample counter by ``batch_size``."""
        if train:
            self.n_batches += 1

    def increase_step(self, batch_size: int):
        self.n_samples += batch_size

    def get_tqdm_postfix(
        self, keys: Optional[Sequence[str]] = None
    ) -> Dict[str, float]:
        if keys is None:
            keys = list(self.data.keys())

        postfix = {}
        for key in keys:
            postfix[key] = self.data[key][self._last_index[key]]
        return postfix


@dataclasses.dataclass
class Training:
    experiment: config.Experiment
    model: flow.SequentialFlow
    classifiers: List[classifiers_mod.Classifier]

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    train_set: data.Dataset
    val_set: data.Dataset
    test_set: data.Dataset

    dev: torch.device
    optimizer: torch.optim.Optimizer

    lr_scheduler: callbacks_mod.LRScheduler
    image_sampler: ImageSampler
    flow_loss: flow.FlowNLLLoss
    history: LogHistory = field(default_factory=LogHistory)
    callbacks: callbacks_mod.Callback = field(
        default_factory=lambda: callbacks_mod.Callbacks([])
    )

    _wandb: Optional[RunWandB] = None

    def __post_init__(self):
        self.to(self.dev)
        self.experiment.makedirs()

    def to(self, device: torch.device):
        self.model.to(device)
        for classifier in self.classifiers:
            classifier.to(device)
        self.image_sampler.to(device)
        self.dev = device

    @staticmethod
    def from_experiment(
        experiment: config.Experiment,
        device: Optional[config.TorchDevice] = None,
        loading: bool = False,
    ) -> Training:
        (
            train_loader,
            val_loader,
            test_loader,
            train_set,
            test_set,
            val_set,
        ) = experiment.load_datasets()

        model = flow.get_model(experiment.model, loading=loading)
        classifiers = classifiers_mod.get_classifiers(
            experiment.classifiers, model, experiment.train.n_train_samples
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0
        )  # set later by lr_scheduler

        torch_device = torch.device(  # type: ignore
            utils.get(device, experiment.train.device)
        )

        lr_scheduler = callbacks_mod.LRScheduler.from_config(
            optimizer, experiment.train.n_train_samples, experiment.train.lr_scheduler
        )

        history = LogHistory()

        examplar_input, _ = next(iter(train_loader))
        examplar_input = examplar_input.to(torch_device)
        n_pixels = int(np.prod(examplar_input.shape[1:]))

        if any(
            isinstance(cl, classifiers_mod.GaussianMixtureClassifier)
            for cl in classifiers
        ):
            prior = "gaussian_mixture"
        else:
            prior = "normal"

        flow_loss = flow.FlowNLLLoss(n_pixels, prior)
        image_sampler = ImageSampler(
            history, model, experiment.output_dir_images, examplar_input
        )

        cbks = [
            lr_scheduler,
            history,
            image_sampler,
        ]
        for cl in classifiers:
            cbks.append(cl)
        cbs = callbacks_mod.Callbacks(cbks)

        return Training(
            experiment,
            model,
            classifiers,
            train_loader,
            val_loader,
            test_loader,
            train_set,
            val_set,
            test_set,
            torch_device,
            optimizer,
            lr_scheduler,
            image_sampler,
            flow_loss,
            history,
            cbs,
        )

    @property
    def cfg(self) -> config.Experiment:
        return self.experiment

    def setup_logging(self):
        # setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.cfg.output_dir, "train.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info(f"output directory: {self.cfg.output_dir}")
        logging.info("Experiment " + repr(self.experiment))

    def init(self):
        """Runs all initialization methods."""
        self.weight_init()
        self.wandb_init()

    def summary(self) -> dict:
        return dict(
            n_parameters=self.count_parameters(),
            n_layers=len(self.model.layers),
            n_train=len(self.train_set),
            n_validation=len(self.val_set),
            n_test=len(self.test_set),
        )

    def dataset_info(self) -> dict[str, Any]:
        if isinstance(self.train_set, data.Two4Two):
            split_args = self.train_set.split_args()
            return dict(
                sampler=split_args.sampler,
                sampler_config=split_args.sampler_config,
                n_samples=split_args.n_samples,
                unbiased=split_args.unbiased,
            )
        return {}

    def slurm_info(self) -> dict[str, Any]:
        slurm_keys = [
            "SLURM_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_ARRAY_JOB_ID",
        ]
        return {
            slurm_key: os.environ[slurm_key]
            for slurm_key in slurm_keys
            if slurm_key in os.environ
        }

    def wandb_init(self, mode: Optional[str] = None):
        config_dict = utils.strict_union(
            dataclasses.asdict(self.cfg),
            self.summary(),
            self.dataset_info(),
            self.slurm_info(),
        )
        self._wandb = wandb.init(  # type: ignore
            project="dubfiv",
            name=self.cfg.unique_marker,
            dir=config.resolve_path("wandb", self.cfg.resolve_paths_filename),
            mode=utils.get(mode, self.experiment.wandb_mode),
            config=config_dict,
        )
        self.wandb.save(
            f"{self.cfg.output_dir}/*.toml", base_path=self.cfg.output_dir, policy="now"
        )

    @property
    def wandb(self) -> RunWandB:
        if self._wandb is None:
            raise ValueError("Forgot to call `wandb_init`?")
        return self._wandb

    def weight_init(self, init_batch_size: int = 500):
        def collect_init_data() -> torch.Tensor:
            img_buf: List[torch.Tensor] = []
            while True:
                for imgs, _ in self.train_loader:
                    img_buf.append(imgs)
                    if sum([ps.shape[0] for ps in img_buf]) >= init_batch_size:
                        return torch.cat(img_buf)

        # img_buf: List[torch.Tensor] = []
        # for i, (imgs, _) in enumerate(self.train_loader):
        #     img_buf.append(imgs)
        #     if sum([ps.shape[0] for ps in img_buf]) >= init_batch_size:
        #         data_for_init = torch.cat(img_buf)
        #         break

        logging.info("data dependent initialization")
        data_for_init = collect_init_data()

        with data_init(list(self.model.layers)):
            self.model(data_for_init.to(self.dev))

    def check_inverse(self):
        imgs, labels = next(iter(self.train_loader))
        with torch.no_grad(), InverseChecker(self.model):
            zs, jac = self.model(imgs.to(self.dev))

    def count_parameters(self) -> int:
        def count_parameters(module: nn.Module) -> int:
            return sum([int(np.prod(np.array(p.shape))) for p in module.parameters()])

        n_parameters = count_parameters(self.model)
        return n_parameters

    @staticmethod
    def load(
        output_dir: str,
        dataset_paths: Union[None, str, Dict[str, str]] = None,
        resolve_paths_filename: Optional[str] = None,
        device: Optional[str] = None,
        wandb_mode: Optional[str] = None,
        dataset_overwrite: Optional[str] = None,
    ) -> Training:
        state = torch.load(
            os.path.join(output_dir, "models", "models.torch"), map_location="cpu"
        )

        # args_file = os.path.join(output_dir, 'args.toml')
        # if os.path.exists(args_file):
        #     with open(os.path.join(output_dir, 'args.toml')) as f:
        #         saved_args = toml.load(f)
        # else:
        #     saved_args = {}

        cfg_fname = os.path.join(output_dir, "models", "experiment.toml")
        cfg = config.Experiment.from_toml(
            cfg_fname,
        )

        device_ = utils.get(device, cfg.train.device)
        del device

        cfg.output_dir = output_dir
        if wandb_mode is not None:
            cfg.wandb_mode = wandb_mode
        if dataset_overwrite is not None:
            cfg.train.dataset = dataset_overwrite
        if resolve_paths_filename is not None:
            cfg.resolve_paths_filename = resolve_paths_filename
        if resolve_paths_filename is not None:
            cfg.resolve_paths_filename = resolve_paths_filename

        train = Training.from_experiment(cfg, device="cpu", loading=True)
        try:
            train.model.load_state_dict(state["model_state"])
            train.optimizer.load_state_dict(state["optimizer"])
            train.history.load_state_dict(state["history"])
            train.image_sampler.load_state_dict(state["image_sampler"])
            for cl, cl_state in zip(train.classifiers, state["classifiers"]):
                cl.load_state_dict(cl_state)
            train.to(torch.device(device_))
        except RuntimeError:
            raise
        return train

    def save(self):
        torch.save(self.history, os.path.join(self.cfg.output_dir, "history_log.torch"))

        torch.save(
            {
                "output_dir": self.cfg.output_dir,
                "image_sampler": self.image_sampler.state_dict(),
                "history": self.history.state_dict(),
                "model_state": self.model.state_dict(),
                "classifiers": [cl.state_dict() for cl in self.classifiers],
                "optimizer": self.optimizer.state_dict(),
                "experiment": self.experiment,
            },
            os.path.join(self.cfg.output_dir_models, "models.torch"),
        )

        cfg_fname = os.path.join(self.cfg.output_dir_models, "experiment.toml")
        with open(cfg_fname, "w") as f:
            toml.dump(self.cfg.asdict(), f)

    @property
    def step(self) -> int:
        """Current training step. measured in seen samples."""
        return self.history.n_samples

    def train_loop(self):
        self.model.train()
        logging.info("Saving model to: " + str(self.cfg.output_dir))

        train_cfg = self.experiment.train
        label_mapping = self.train_set.get_label_mapping()

        while True:
            tqdm_loader = tqdm_auto.tqdm(self.train_loader, ascii=True, mininterval=10)
            self.model.train()
            for imgs, labels in tqdm_loader:
                self.history.increase_step(len(imgs))
                self.callbacks.on_batch_begin(self.step, train=True)
                imgs = imgs.to(self.dev)
                labels = labels.float().to(self.dev)
                self.optimizer.zero_grad()
                bs, ch, h, w = imgs.shape

                zs, logdet = self.model(imgs)

                losses = []
                mixture_nll = None
                for cl in self.classifiers:
                    cl_label = self.get_label(cl.label_name, label_mapping, labels)
                    if self.cfg.train.label_noise > 0:
                        mask = torch.rand_like(cl_label) < self.cfg.train.label_noise
                        cl_label[mask] = torch.randint_like(cl_label, 2)[mask]
                    if isinstance(cl, classifiers_mod.GaussianMixtureClassifier):
                        cl_loss, mixture_nll = cl.loss(cl_label)
                    else:
                        cl_loss = cl.loss(cl_label)  # type: ignore
                    self.history.log(f"train.{cl.name}.loss", cl_loss.item())
                    self.history.log(
                        f"train.{cl.name}.lr", cl.lr_scheduler.get_lr(self.step)
                    )
                    losses.append(cl_loss)

                nll_loss = self.flow_loss(zs, logdet, mixture_nll)
                losses.append(nll_loss)
                loss = torch.stack(losses).sum()

                if loss.item() > 100:
                    raise ValueError("loss too high: ", loss.item())

                self.history.log("lr", self.lr_scheduler.get_lr(self.step))
                self.history.log("loss", loss)
                self.history.log("train.nll", nll_loss.mean())
                logdet_per_dim = logdet.mean().item() / (ch * h * w * np.log(2))
                self.history.log("log_det_jac", logdet_per_dim)
                postfix = self.history.get_tqdm_postfix()
                tqdm_loader.set_postfix(postfix, refresh=False)

                # optimize
                loss.backward()
                self.optimizer.step()
                self.callbacks.on_batch_end(self.step, train=True)

                if self.step >= train_cfg.n_train_samples:
                    self.evaluate("validation", progbar=False)
                    self.evaluate("test", progbar=False)
                    self.callbacks.on_train_end()
                    self.save()
                    return

            self.evaluate("validation")
            self.save()
            self.callbacks.on_epoch_end()

    def save_train_images(self, n_rows: int = 8):
        imgs = [self.train_set[i][0] for i in range(n_rows ** 2)]
        grid = torchvision_utils.make_grid(torch.stack(imgs), nrow=n_rows).numpy()
        filename = os.path.join(self.cfg.output_dir_images, "train.png")
        np_grid = (255 * np.clip(grid.transpose(1, 2, 0), 0, 1)).astype(np.uint8)
        imageio.imsave(filename, np_grid)
        self.history.log_image("train_images", filename)

    def get_dataloader(self, split_name: str) -> DataLoader:
        return {
            "train": self.train_loader,
            "validation": self.val_loader,
            "test": self.test_loader,
        }[split_name]

    def get_dataset(self, split_name: str) -> data.Dataset:
        return {
            "train": self.train_set,
            "validation": self.val_set,
            "test": self.test_set,
        }[split_name]

    def get_label(
        self,
        label_name: str,
        label_mapping: Mapping[str, int],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        idx = label_mapping[label_name]
        return torch.index_select(labels, 1, torch.tensor([idx], device=labels.device))

    def evaluate(
        self,
        split: str = "validation",
        dataloader: Optional[DataLoader] = None,
        classifiers: Optional[Sequence[classifiers_mod.Classifier]] = None,
        progbar: bool = True,
        log: bool = True,
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Tests the glow_net and classifiers on the data loader ``loader``."""
        self.model.eval()

        # ensure that dataloader is not None
        loader = dataloader if dataloader is not None else self.get_dataloader(split)

        dataset: data.Dataset = loader.dataset  # type: ignore
        label_mapping = utils.get(label_mapping, dataset.get_label_mapping())

        if classifiers is None:
            classifiers = self.classifiers

        if progbar:
            loader = tqdm_auto.tqdm(loader, ascii=True)
        nll_x_bits: List[np.ndarray] = []

        for cl in classifiers:
            cl.record_metrics(True)

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.dev)
                labels = labels.to(self.dev)

                zs, logdet = self.model(imgs.to(self.dev))

                mixture_nll = None
                for cl in classifiers:
                    cl_label = self.get_label(cl.label_name, label_mapping, labels)
                    if isinstance(cl, classifiers_mod.GaussianMixtureClassifier):
                        _, mixture_nll = cl.loss(cl_label)
                    else:
                        cl.loss(cl_label)

                nll_loss = self.flow_loss(zs, logdet, mixture_nll)
                nll_x_bits.append(utils.to_numpy(nll_loss))

        nll = float(np.mean(np.stack(nll_x_bits)))
        classifier_metrics = [cl.get_recorded_metrics() for cl in classifiers]

        if log:
            self.history.log(f"{split}.nll", nll)

            for cl, cl_metrics in zip(classifiers, classifier_metrics):
                self.history.logs(f"{split}.{cl.name}", cl_metrics)
        self.model.train()
        return {
            "nll": nll,
            "classifiers": classifier_metrics,
        }

    def setup(self, wandb_mode: Optional[str] = None):
        self.setup_logging()
        self.weight_init()
        self.wandb_init(utils.get(wandb_mode, self.cfg.wandb_mode))

    def train(self):
        self.save_train_images()
        self.check_inverse()
        logging.info(
            "parameters: {:0.2f} million".format(self.count_parameters() / 1e6)
        )
        self.evaluate()
        self.train_loop()


@dataclasses.dataclass
class ImageSampler(callbacks_mod.Callback):
    history: LogHistory
    model: flow.SequentialFlow
    output_dir: str
    examplar_input: torch.Tensor
    _rand_zs: Optional[Sequence[torch.Tensor]] = None

    def to(self, device: torch.device):
        self.examplar_input = self.examplar_input.to(device)
        if self._rand_zs is not None:
            self._rand_zs = [z.to(device) for z in self._rand_zs]

    def load_state_dict(self, state: dict[str, Any]):
        self._rand_zs = state["_rand_zs"]
        if "examplar_input" in state:
            self.examplar_input = state["examplar_input"]

    def state_dict(self) -> dict[str, Any]:
        return {
            "_rand_zs": self._rand_zs,
            "examplar_input": self.examplar_input,
        }

    def sample_rand_zs(self, n_samples: int = 100) -> flow.Latent:
        def random_zs(zs: flow.Latent, bs: int) -> flow.Latent:
            rand_zs = []
            for z in zs:
                _, c, h, w = z.shape
                rand_zs.append(torch.randn((bs, c, h, w), device=z.device))
            return rand_zs

        with torch.no_grad():
            zs, jac = self.model(self.examplar_input)
        return random_zs(zs, 100)

    @property
    def rand_zs(self) -> Sequence[torch.Tensor]:
        if self._rand_zs is not None:
            return self._rand_zs

        self._rand_zs = self.sample_rand_zs()
        return self._rand_zs

    def sample_image(
        self,
        zs: Optional[flow.Latent] = None,
        plot: bool = False,
        filename: Optional[str] = None,
    ):
        flow.flow_sample_image(
            self.model,
            utils.get(zs, self.rand_zs),
            plot=plot,
            filename=filename,
        )
        if filename is not None:
            self.history.log_image("samples", filename)

    def get_filename(self) -> str:
        return os.path.join(
            self.output_dir,
            f"{self.history.n_samples:08d}_{self.history.epoch:05d}.png",
        )

    def on_batch_end(self, step: int, train: bool):
        # plot for the first 3000 batches
        if self.history.n_batches < 3000 and self.history.n_batches % 100 == 0:
            self.sample_image(filename=self.get_filename())
        if self.history.n_batches % 1000 == 0:
            self.sample_image(filename=self.get_filename())

    def on_epoch_end(self):
        self.sample_image(filename=self.get_filename())

    def on_train_end(self):
        self.sample_image(filename=self.get_filename())


def main():
    exp = config.Experiment.from_args()

    pprint.pprint(exp.asdict(), indent=2)

    with utils.pdb_post_mortem(enable=exp.pdb):
        training = Training.from_experiment(exp, exp.train.device)
        training.setup()
        training.train()


if __name__ == "__main__":
    main()
