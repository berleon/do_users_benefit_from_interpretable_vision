# Code for "Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset"

Welcome to our code release of our ICLR 2022 paper:
"Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset"

This repository contains code to:

* generate the dataset, 
* train the model, 
* generate the users input for the conditions.

The Two4Two dataset can be found at [mschuessler/two4two](https://github.com/mschuessler/two4two/)

## Install

```bash
python -m venv ../venv
source ../venv/bin/activate
pip install -U pip wheel
# install the right pytorch version. see: https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# install repo
pip install -e .
```

## Donwload Links

You can execute the following script:

```bash
./download_artifacts.sh
```

Or just download them directly:

[[Model Weights]](
https://f002.backblazeb2.com/file/iclr2022/do_users_benefit_from_interpretable_vision_model.tar.gz
)
[[Dataset Biased]](
https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15.tar
)

[[Dataset Unbiased]](
https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15_unbiased.tar
)
[[Export Study]](
https://f002.backblazeb2.com/file/iclr2022/ICLR2022_Export_Do_Users_Benefit_From_Interpretable_Vision.qsf
)

## How to use the model

See `./Example_Analysis.ipynb` for how to load and use the pretrained model.

## From datasets to final models

The following commands generate a dataset, train a model, and then renders the 
model's explanations.

```bash
python -m two4two --download_blender config/dataset_config.toml
# change `cuda` to your preferred device
dubfiv_train --device cuda --output_base_dir ./models --experiment config/model.toml 
dubfiv_export_user_study --model ./models/<model_dir>
```

To reproduce the supervised expirement, you need to train a network on an unbiased dataset:

```bash
# render unbiased dataset
python -m two4two --download_blender config/dataset_config_unbiased.toml

# train supervised model
dubfiv_supervised \
    --data_output_dir=`realpath models/supervised` \
    --dataset="two4two_obj_color_and_spherical" \
    --model_name=mobilenet_v2 \
    '--model_kwargs={"width_mult": 0.5}' \
    --batch_size=50 \
    --num_epochs=61 \
    --ckpt_freq=20
# run analysis with supervised model and INN
dubfiv_supervised_analysis --supervised_dir ./models/supervised/<supervised_model_dir> --model ./models/<inn_model>
```

## Points to the code

* The Two4Two sampler: [`./dubfiv/data_generation.py`](dubfiv/data_generation.py)
* The Training loop: [`./dubfiv/train.py`](dubfiv/train.py)


## Citation

```
@inproceedings{
    sixt2022do,
    title={Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset},
    author={Leon Sixt and Martin Schuessler and Oana-Iuliana Popescu and Philipp Wei{\ss} and Tim Landgraf},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=v6s3HVjPerv}
}
```