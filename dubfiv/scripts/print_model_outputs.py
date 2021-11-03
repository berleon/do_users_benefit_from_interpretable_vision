"""Script to print output dimension."""

import argparse
import contextlib
from typing import Any, Sequence, Union

import torch

from dubfiv import config
from dubfiv import train as train_mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='train config')
    parser.add_argument('--model', type=str, help='model config')
    parser.add_argument('--classifier', type=str, help='classifier config')
    parser.add_argument('--dataset', type=str, default=None,
                        help='overwrite dataset')

    args = parser.parse_args()
    print_output_dimension(
        args.train,
        args.model,
        args.classifier,
        dataset=args.dataset,
    )


def print_output_dimension(
    train: str,
    model: str,
    classifier: str,
    dataset: str = None,
    output_base_dir: str = '/tmp/',
):
    print('output_base_dir', output_base_dir)
    print('train', train)
    print('model', model)
    print('classifier', classifier)
    print('dataset', dataset)

    exp = config.Experiment.create_from_config_files(
        output_base_dir,
        train,
        model,
        classifier,
        dataset_overwrite=dataset,
    )

    training = train_mod.Training.from_experiment(exp, device='cpu')
    layer_idx = 0

    def print_output_shape(
        module: torch.nn.Module,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
    ):
        nonlocal layer_idx

        inp = inputs[0]
        out = outputs[0]
        if isinstance(out, tuple):
            out = out[0]
        print(f"{layer_idx:03d}, {type(module).__name__:>20}, "
              f"{str(tuple(inp.shape)):>20}, {str(tuple(out.shape)):>20}")
        layer_idx += 1

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

    flow = training.model
    imgs, _ = next(iter(training.train_loader))
    imgs = imgs.to(training.dev)
    with one_time_hooks(flow.layers, print_output_shape), torch.no_grad():
        flow(imgs[:1])
