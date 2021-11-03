"""script to export user study figures."""

from __future__ import annotations

import argparse
import pdb
from typing import Optional

from dubfiv import analysis as analysis_mod


def run(model_dir: str,
        supervised_dir: str,
        sync: bool = True):

    analysis = analysis_mod.Analysis.from_checkpoint(model_dir)
    two4two_intervention = analysis_mod.Two4TwoIntervention(
        analysis
    )
    two4two_intervention()

    two4two_supervised = analysis_mod.Two4TwoSupervisedAnalysis(
        analysis=analysis,
        supervised_dir=supervised_dir,
    )
    two4two_supervised()


def main(args: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model directory to analyse.')
    parser.add_argument('--supervised_dir', type=str,
                        help='Supervised model directory.')

    parsed_args = parser.parse_args(args)
    try:
        run(parsed_args.model, parsed_args.supervised_dir)
    except Exception:
        pdb.post_mortem()
