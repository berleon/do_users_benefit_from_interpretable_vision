"""."""

from __future__ import annotations

import os
from typing import Callable, Optional
import warnings

# import GPUtil
# import wandb
import numpy as np
import tap

from dubfiv import analysis as analysis_mod


def slurm_user_study(output_dir: str):
    model_path = os.path.join(
        output_dir, 'models/models.torch')

    if os.path.exists(model_path):
        print(f"dubfiv_export_user_study --model {output_dir}")
    else:
        print(f"# does not exist {model_path}")


def slurm_two4two_analysis(output_dir: str, supervised_dir: str):
    model_path = os.path.join(
        output_dir, 'models/models.torch')

    if os.path.exists(model_path):
        print(f"dubfiv_two4two_analysis --supervised_dir {supervised_dir} --model {output_dir}")
    else:
        warnings.warn(f"Does not exist {model_path}")


class Args(tap.Tap):
    command: str = 'user_study'  # which command to print
    supervised_dir: Optional[str] = None  # supervised model for `two4two_analysis`.

    def get_command_func(self) -> Callable[[str], None]:
        if self.command == 'user_study':  # which command to print
            return slurm_user_study
        elif self.command == 'two4two_analysis':
            if self.supervised_dir is None:
                raise ValueError("two4two_analysis requires `supervised_dir` to be set.")
            supervised_dir = self.supervised_dir
            return lambda x: slurm_two4two_analysis(x, supervised_dir)
        else:
            raise Exception()


def main():
    args = Args().parse_args()

    runs_df = analysis_mod.load_runs_from_wandb()
    runs_df['output_dir'] = runs_df.config.apply(lambda x: x['output_dir'])
    runs_df['is_taurus'] = runs_df.meta.apply(lambda x: x['host'].startswith('taurus'))

    runs_df[
        np.logical_and(runs_df.is_taurus, runs_df.state == 'finished')
    ].output_dir.apply(args.get_command_func())

    runs_df.to_csv('/tmp/runs_df.csv')
    runs_df.to_pickle('/tmp/runs_df.pickle')
