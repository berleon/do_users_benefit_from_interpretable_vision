"""script to export user study figures."""

from __future__ import annotations

import pdb
from typing import Optional

import tap
import torch

from dubfiv import analysis as analysis_mod


def run(model_dir: str):
    analysis = analysis_mod.Analysis.from_checkpoint(model_dir)

    print("number layers:", len(analysis.ex.model.layers))
    print(analysis.summary())

    print("classifier weight norm: ")
    print(torch.norm(analysis.explained_classifier.fc.weight).item())
    # print('treatment')
    # print(analysis.treatment.summary())
    # print()
    # print('task')
    # print(analysis.task.summary())

    logit_distr = analysis_mod.LogitDistribution(analysis)
    logit_distr()


class Args(tap.Tap):
    model: str = 'Model directory to analyse'


def main(argv: Optional[list[str]] = None):
    args = Args().parse_args(argv)
    try:
        run(args.model)
    except Exception:
        pdb.post_mortem()
