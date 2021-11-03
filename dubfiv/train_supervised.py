"""Script to train a resnet on Two4Two."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import field
from datetime import datetime
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
from tap import Tap
import torch
from torchvision import models
import wandb

from dubfiv import config
from dubfiv import data
from dubfiv import utils


@dataclasses.dataclass
class PerformanceMetrics:
    losses: dict[str, float]
    accuracies: dict[str, float]
    split: str
    step: int

    def keys(self) -> list[str]:
        return sorted(set(self.losses.keys()).union(self.accuracies.keys()))

    def print_metrics(self):
        for key in self.keys():
            if key in self.losses:
                print(f'{self.split}.loss.{key}: {self.losses[key]:.4f}')
            if key in self.accuracies:
                print(f'{self.split}.accuracy.{key}: {self.accuracies[key]:.4f}')

    def log_metrics(self):
        wandb.log({f'{self.split}.loss.{key}': val
                   for key, val in self.losses.items()},
                  step=self.step)
        wandb.log({f'{self.split}.accuracy.{key}': val
                   for key, val in self.accuracies.items()},
                  step=self.step)


def test_model(
    opt: CLIOptions,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Two4TwoCriterion,
    split: str,
    step: int,
) -> PerformanceMetrics:
    total_step = len(dataloader)

    print('Num steps', total_step)

    model.eval()
    criterion.reset_metrics()

    for inputs, labels in dataloader:
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)

        with torch.no_grad():
            outputs = model(inputs)
            criterion.get_loss(outputs, labels)

    stats = criterion.get_metrics(split, step)
    stats.print_metrics()
    stats.log_metrics()
    return stats


def train_model(
    opt: CLIOptions,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Two4TwoCriterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    validation_dataloader: torch.utils.data.DataLoader = None,
) -> tuple[
    int,
    # train losses and accuracies
    list[PerformanceMetrics],
    # validation losses and accuracies
    list[PerformanceMetrics],
]:
    def save():
        torch.save({
            'epoch': epoch,
            'cli_arguments': opt.as_dict(),
            'model_state_dict': model.state_dict(),
            'num_classes': criterion.get_num_output_dim(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': loss
        }, os.path.join(opt.log_path, 'model_{}.ckpt'.format(epoch)))

    since = time.time()
    train_metrics = []
    val_metrics = []

    print('Num steps', len(dataloader))

    global_step = 0

    if validation_dataloader is not None:
        val_metrics.append(
            test_model(opt, model, validation_dataloader, criterion,
                       'val', global_step)
        )

    for epoch in range(opt.num_epochs):
        print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
        print('-' * 10)

        model.train()
        criterion.reset_metrics()

        for inputs, labels in dataloader:
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion.get_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            wandb.log({'train.loss': loss.item()}, step=global_step)
            global_step += len(inputs)

        epoch_metrics = criterion.get_metrics('train', global_step)
        epoch_metrics.print_metrics()
        epoch_metrics.log_metrics()
        train_metrics.append(epoch_metrics)

        if epoch % opt.ckpt_freq == 0:
            save()
            if validation_dataloader is not None:
                val_metrics.append(
                    test_model(opt, model, validation_dataloader, criterion,
                               'val', global_step)
                )

        scheduler.step()
    save()
    time_elapsed = time.time() - since

    print('Time elapsed:', time_elapsed)

    return global_step, train_metrics, val_metrics


def patch_model(
    opt: CLIOptions,
    model: torch.nn.Module,
    num_classes: int,
):
    if opt.model_name.startswith('resnet'):
        num_ftrs: int = model.fc.in_features  # type: ignore
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif (hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential)):
        fc = model.classifier[-1]
        num_ftrs: int = fc.in_features  # type: ignore
        model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)  # type: ignore
    else:
        raise ValueError(f'Do not know how to patch the model: {opt.model_name}')


def initialize_model(
    opt: CLIOptions,
    num_classes: int
) -> torch.nn.Module:

    model = None
    use_pretrained = False
    if opt.pretrained == 'pretrained':
        use_pretrained = True

    model_cls = getattr(models, opt.model_name)

    model = model_cls(pretrained=use_pretrained, **opt.get_model_kwargs())

    patch_model(opt, model, num_classes)

    return model


def load_model_from_ckpt(
    path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, CLIOptions]:
    state = torch.load(path, map_location=device)
    options = CLIOptions(**state['cli_arguments'])
    num_classes = state['num_classes']
    model = initialize_model(options, num_classes)
    model.load_state_dict(state['model_state_dict'])
    return model, options


def reload_model(
    opt: CLIOptions,
    model: torch.nn.Module
) -> torch.nn.Module:
    print('LOADING', os.path.join(opt.model_path,
                                  "model_{}.ckpt".format(opt.model_num)))

    model.load_state_dict(
        torch.load(
            os.path.join(opt.model_path,
                         "model_{}.ckpt".format(opt.model_num)),
            map_location=opt.get_device().type))

    return model


@dataclasses.dataclass
class OutputDim:
    name: str
    loss: torch.nn._Loss  # type: ignore
    indexes: list[int] = field(default_factory=list)
    accumulated_loss: list[float] = field(default_factory=list)
    accumulated_pred: list[np.ndarray] = field(default_factory=list)
    accumulated_targets: list[np.ndarray] = field(default_factory=list)

    def reset_accumulated(self, reset: bool = True):
        if reset:
            self.accumulated_loss = []
            self.accumulated_pred = []
            self.accumulated_targets = []


@dataclasses.dataclass
class Two4TwoCriterion:
    output_dims: list[OutputDim] = field(default_factory=lambda: [
        OutputDim('obj_name', torch.nn.BCEWithLogitsLoss()),
        OutputDim('spherical', torch.nn.MSELoss()),
        OutputDim('bending', torch.nn.MSELoss()),
        OutputDim('obj_rotation_roll', torch.nn.MSELoss()),
        OutputDim('obj_rotation_pitch', torch.nn.MSELoss()),
        OutputDim('obj_rotation_yaw', torch.nn.MSELoss()),
        OutputDim('position_x', torch.nn.MSELoss()),
        OutputDim('position_y', torch.nn.MSELoss()),
        OutputDim('arm_position', torch.nn.MSELoss()),
        OutputDim('obj_color', torch.nn.MSELoss()),
        OutputDim('bg_color', torch.nn.MSELoss()),
    ])

    metrics: dict[str, float] = field(default_factory=dict)

    def get_output_names(self) -> list[str]:
        return [out.name for out in self.output_dims]

    def get_num_output_dim(self) -> int:
        indexes = set()
        for output_dim in self.output_dims:
            indexes.update(output_dim.indexes)
        return max(indexes) + 1

    def read_index_positions(self, dataset: data.Dataset):
        label_mapping = dataset.get_label_mapping()
        for output_dim in self.output_dims:
            output_dim.indexes = [label_mapping[output_dim.name]]

    def get_loss(
            self,
            outputs: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0., device=outputs.device)
        for outdim in self.output_dims:
            idx = outdim.indexes
            pred = outputs[:, idx]
            targets = labels[:, idx]
            loss_outdim = outdim.loss(pred, targets)
            loss += loss_outdim
            outdim.accumulated_loss.append(loss_outdim.item())
            outdim.accumulated_pred.append(utils.to_numpy(pred))
            outdim.accumulated_targets.append(utils.to_numpy(targets))

        return loss

    def get_metrics(self, split: str, step: int) -> PerformanceMetrics:
        losses = {}
        accuracies = {}
        for outdim in self.output_dims:
            losses[outdim.name] = float(np.stack(outdim.accumulated_loss).mean())

            if isinstance(outdim.loss, torch.nn.BCEWithLogitsLoss):
                pred = np.concatenate(outdim.accumulated_pred)
                targets = np.concatenate(outdim.accumulated_targets)
                accuracies[outdim.name] = float(((pred > 0) == targets).mean())

        return PerformanceMetrics(losses, accuracies, split, step)

    def reset_metrics(self):
        for outdim in self.output_dims:
            outdim.reset_accumulated(reset=True)


def run_for_seed(opt: CLIOptions):
    # set random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    opt.create_log_path(run_id=datetime.utcnow().isoformat())

    run = wandb.init(
        project='dubfiv_supervised',
        reinit=True,
        mode=opt.wandb_mode,
        dir=opt.wandb_dir,
        config=opt.as_dict(),
    )

    with open(os.path.join(opt.log_path, 'opt.json'), 'w') as f:
        json.dump(opt.as_dict(), f, indent=2)

    dataset_dir = (opt.data_input_dir if opt.data_input_dir is not None
                   else config.resolve_path(opt.dataset))

    critertion = Two4TwoCriterion()

    (train_loader, val_loader, test_loader,
     train_set, test_set, val_set) = data.load_datasets(
         opt.dataset, dataset_dir,
         batch_size=opt.batch_size,
         image_size=128,
         num_workers=opt.num_workers,
         dataset_kwargs=dict(
             return_attributes=critertion.get_output_names()
         ))

    critertion.read_index_positions(train_set)

    model = initialize_model(opt, critertion.get_num_output_dim())

    model = model.to(opt.device)

    # params_to_update = model.parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    step_size = opt.num_epochs // 3
    scheduler = torch.optim.lr_scheduler.StepLR(  # type: ignore
        optimizer, step_size=step_size, gamma=0.1, verbose=True,
    )

    # TRAIN
    global_step, train_metrics, val_metrics = train_model(
        opt, model, train_loader, critertion,
        optimizer, scheduler, val_loader)

    final_test_metric = test_model(opt, model, test_loader,
                                   critertion, 'test', global_step)

    out = {
        'train_metrics': [dataclasses.asdict(metric) for metric in train_metrics],
        'val_metrics': [dataclasses.asdict(metric) for metric in val_metrics],
        'final_test_metrics': dataclasses.asdict(final_test_metric),
    }

    with open(os.path.join(opt.log_path, 'results.json'), 'w') as f:
        json.dump(out, f)

    run.finish()  # type: ignore


class CLIOptions(Tap):
    batch_size: int = 128    # Batch size per GPU
    learning_rate: float = 1e-3  # Learning rate
    num_epochs: int = 50    # Number of epochs to train
    in_channels: int = 3    # Number of input channels
    num_workers: int = 4    # Number of data loader threads
    resume: bool = False    # Resume training
    seed: int = 0           # Random seed for reproducibility
    ckpt_freq: int = 10     # Checkpoint frequency
    model_name: str = 'resnet18'    # Model name resnet18/34/50
    model_kwargs: str = '{}'    # Additional model kwargs
    pretrained: bool = False    # scratch or pretrained
    model_path: str = './'      # Load model path. Use when resuming
    model_num: int = 100        # Load model number. Use when resuming
    wandb_mode: str = 'online'  # Weight and bias mode. Eihter offline online or disabled.
    wandb_dir: str = 'wandb'    # wand cache dir.
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"  # Torch device
    data_output_dir: str    # Output model dir
    save_dir: str   # Training name for save folder
    dataset: str = 'two4two'    # Dataset name.
    data_input_dir: Optional[str] = None  # Data input dir. If not given, will try to resolve it.
    log_path: str = ""

    def process_args(self):
        self._model_kwargs_dict: dict[str, Any] = ast.literal_eval(self.model_kwargs)
        if not isinstance(self._model_kwargs_dict, dict):
            raise ValueError(f'--model_kwargs must be a dict literal. Got {self.model_kwargs}')

    def get_model_kwargs(self) -> dict[str, Any]:
        return self._model_kwargs_dict

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    def create_log_path(self, run_id: str):
        self.log_path = os.path.join(self.data_output_dir, run_id)
        os.makedirs(self.log_path, exist_ok=True)

    def load_datasets(self) -> data.DATALOADERS_AND_DATASETS:
        dataset_dir = (self.data_input_dir if self.data_input_dir is not None
                       else config.resolve_path(self.dataset))

        critertion = Two4TwoCriterion()
        return data.load_datasets(
            self.dataset, dataset_dir,
            batch_size=self.batch_size,
            image_size=128,
            num_workers=self.num_workers,
            dataset_kwargs=dict(
                return_attributes=critertion.get_output_names()
            ))

    def as_dict(self) -> Dict[str, Any]:
        state = super().as_dict()
        return {
            k: v for k, v in state.items()
            if k not in ['get_model_kwargs', 'get_device',
                         'create_log_path', 'load_datasets']
        }


def main():
    opt = CLIOptions().parse_args()
    run_for_seed(opt)


if __name__ == '__main__':
    main()
