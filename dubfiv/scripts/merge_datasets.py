"""Script to merge two4two datsets."""

from __future__ import annotations

import glob
import json
import os
import shutil
import tarfile
from typing import List

import tap
import tqdm.auto
import two4two
import two4two.cli_tool

from dubfiv import gt_eval
from dubfiv import utils


class Args(tap.Tap):
    destination: str
    datasets: List[str]


def get_destination_path(
        destination: str,
        merge_from: str,
        path: str,
) -> str:
    return os.path.join(destination, os.path.relpath(path, merge_from))


def make_tarfile(output_filename: str, source_dir: str):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def assert_params_are_ordered(destination: str):
    interventions_paths = glob.glob(f'{destination}/**/intervention.json')
    interventions = []
    for path in interventions_paths:
        with open(path) as f:
            interventions.append(
                two4two.cli_tool.InterventionArgs.from_dict(json.load(f)))

    for intervention in interventions:
        if intervention.is_original():
            continue
        original = intervention.get_original_split()
        with open(f'{destination}/{original.dirname}/parameters.jsonl') as f:
            original_params = two4two.SceneParameters.load_jsonl(f)
        with open(f'{destination}/{intervention.dirname}/parameters.jsonl') as f:
            modified_params = two4two.SceneParameters.load_jsonl(f)

        for orig_param, mod_param in zip(original_params, modified_params):
            assert orig_param.id == mod_param.original_id


def main():
    args = Args().parse_args()

    assert not os.path.exists(args.destination)

    print(f"Got {len(args.datasets)} to merge")
    for dataset in args.datasets:
        jsonl_files = glob.glob(f"{dataset}/**/parameters.jsonl", recursive=True)
        print("In path", dataset)
        print("Found", len(jsonl_files), "splits.")
        print("Will be copied to",
              get_destination_path(args.destination, dataset,
                                   jsonl_files[0]))

    # copy config files from first dataset
    dataset = args.datasets[0]

    for config_file in glob.glob(f"{dataset}/**/*.json", recursive=True):
        dest_path = get_destination_path(args.destination, dataset, config_file)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(config_file, dest_path)

    for merge_from in tqdm.tqdm(args.datasets):
        jsonl_files = glob.glob(f"{merge_from}/**/parameters.jsonl", recursive=True)
        for jsonl in tqdm.tqdm(jsonl_files):
            dest_path = get_destination_path(args.destination, merge_from, jsonl)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            if not os.path.exists(dest_path):
                open(dest_path, 'w').close()

            # read lines from source file
            with open(jsonl, 'r') as f:
                jsonl_content = f.readlines()

            with open(dest_path, 'a') as f:
                f.writelines(jsonl_content)

        png_files = glob.glob(f"{merge_from}/**/*.png", recursive=True)
        for png_file in tqdm.tqdm(png_files):
            dest_path = get_destination_path(args.destination, merge_from, png_file)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(png_file, dest_path)

    with utils.pdb_post_mortem():
        assert_params_are_ordered(args.destination)
        interventions = gt_eval.Two4TwoInterventionAnalysis.from_path(args.destination)
        interventions.align_datasets()

        tar_file = args.destination + '.tar.gz'
        print(f"Creating tar file: {tar_file}")
        make_tarfile(tar_file, args.destination)
