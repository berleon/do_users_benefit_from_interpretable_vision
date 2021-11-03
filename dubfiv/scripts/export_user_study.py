"""script to export user study figures."""

from __future__ import annotations

import os
import pdb
from typing import Iterator, Optional

import tap

from dubfiv import analysis as analysis_mod
from dubfiv import prototypes as prototypes_mod


ProtoAndVisualGen = Iterator[
    tuple[prototypes_mod.PrototypesConfig, list[prototypes_mod.VisualConfig]]
]


def get_prototype_configs() -> ProtoAndVisualGen:
    proto_config_base = prototypes_mod.PrototypesConfig()
    for resolution in [8, 4]:
        for n_components in [6, 8, 10]:
            for method in ["pca", "mf"]:
                proto_config = proto_config_base.clone(
                    layer_resolution=resolution,
                    method=method,
                    n_components=n_components,
                )
                yield proto_config, list(get_prototype_visual_config(proto_config))


def get_prototype_config_conv1x1() -> ProtoAndVisualGen:
    proto_config_base = prototypes_mod.PrototypesConfig()
    for resolution in [8]:
        for n_components in [8]:
            for conv1x1_location in [0, 0.5, 0.95]:
                for method in ["mf"]:
                    proto_config = proto_config_base.clone(
                        layer_resolution=resolution,
                        method=method,
                        n_components=n_components,
                        conv1x1_location=conv1x1_location,
                    )
                    yield proto_config, [prototypes_mod.VisualConfig()]


def get_prototype_config_n_components() -> ProtoAndVisualGen:
    proto_config_base = prototypes_mod.PrototypesConfig()
    for resolution in [8]:
        for n_components in [10, 6, 15]:
            for method in ["mf"]:
                proto_config = proto_config_base.clone(
                    layer_resolution=resolution,
                    method=method,
                    n_components=n_components,
                )
                yield proto_config, [prototypes_mod.VisualConfig()]


def get_prototype_config_depth() -> ProtoAndVisualGen:
    proto_config_base = prototypes_mod.PrototypesConfig()
    for resolution in [8, 4, 2, 1]:
        for n_components in [6]:
            for method in ["mf"]:
                proto_config = proto_config_base.clone(
                    layer_resolution=resolution,
                    method=method,
                    n_components=n_components,
                )
                yield proto_config, [prototypes_mod.VisualConfig()]


def get_prototype_visual_config(
    proto_config: prototypes_mod.PrototypesConfig,
) -> Iterator[prototypes_mod.VisualConfig]:
    if proto_config.layer_resolution in [8, 4] and proto_config.n_components == 6:
        for swap_neg in [True, False]:
            for reduce_method in ["sum", "max"]:
                for prototypes_percentile in [
                    5.0,
                    10.0,
                ]:  # 25., 50.]:
                    yield prototypes_mod.VisualConfig(
                        swap_neg=swap_neg,
                        reduce_method=reduce_method,
                        prototypes_percentile=prototypes_percentile,
                    )
    else:
        yield prototypes_mod.VisualConfig()


def run(
    model_dir: str,
    condition: str,
    prototype_accuracy: bool = False,
    prototype_config: str = "default",
    sync: bool = True,
):
    def selected(condition: str, cond_name: str) -> bool:
        return condition == "all" or condition == cond_name

    analysis = analysis_mod.Analysis.from_checkpoint(model_dir)

    print(analysis.summary())

    print("treatment")
    print(analysis.treatment.summary())
    print()
    print("task")
    print(analysis.task.summary())

    if selected(condition, "inn"):
        weight_interpolation = analysis_mod.WeightInterpolation(
            analysis,
            analysis.task,
        )
        weight_interpolation(sync=sync)

        for perc in [80, 90, 95, 99]:
            cf_values, _ = analysis_mod.get_counterfactual_bins(
                analysis.task.logits, [50, perc]
            )
            weight_result = weight_interpolation(
                sync=sync,
                cf_values=cf_values,
                marker=str(perc),
            )
            weight_result.save(analysis.output_dir)

    if selected(condition, "baseline"):
        baseline = analysis_mod.Baseline(
            analysis,
            analysis.task,
        )

        for bins in [
            [20, 60],  # [30, 70], [50, 80], [20, 50], [40, 70], [50, 90]
        ]:
            _, cf_bins = analysis_mod.get_counterfactual_bins(
                analysis.task.logits, bins
            )
            baseline_result = baseline(
                sync=sync,
                cf_bins=cf_bins,
                marker="_".join(map(str, bins)),
            )
            print(baseline_result)
            baseline_result.save(analysis.output_dir)

    if selected(condition, "print_prototypes_config"):
        for proto_config, visual_configs in get_prototype_configs():
            for visual_config in visual_configs:
                print("Exporting Prototypes")
                print("Prototypes Config: ", proto_config)
                print("Visual Config:", visual_config)

    if selected(condition, "prototypes"):
        config_func = {
            "default": get_prototype_configs,
            "depth": get_prototype_config_depth,
            "conv1x1": get_prototype_config_conv1x1,
            "n_components": get_prototype_config_n_components,
        }[prototype_config]

        for proto_config, visual_configs in config_func():
            prototypes = prototypes_mod.compute_prototypes(analysis, proto_config)
            for visual_config in visual_configs:
                print("Exporting Prototypes")
                print("Prototypes Config: ", proto_config)
                print("Visual Config:", visual_config)
                export_proto = prototypes_mod.ExportPrototypes(
                    analysis,
                    prototypes,
                    proto_config,
                    visual_config,
                    overwrite_force=True,
                )
                export_proto.export_prototype(prototype_accuracy)


class Args(tap.Tap):
    model: str  # Model directory to analyse
    condition: str = "all"
    """Condition to export. Possible Values: all, baseline, inn, prototypes."""

    prototype_accuracy: bool = False  # Export prototypes accuracies.
    prototype_config: str = "default"  # Render Prototype config


def main(args: Optional[list[str]] = None):
    parsed_args = Args().parse_args(args)
    try:
        run(
            parsed_args.model,
            parsed_args.condition,
            parsed_args.prototype_accuracy,
            parsed_args.prototype_config,
        )
    except Exception:
        pdb.post_mortem()
