#! /usr/bin/env bash


if [ ! -e "do_users_benefit_from_interpretable_vision_model.tar.gz" ]; then
  wget https://f002.backblazeb2.com/file/iclr2022/do_users_benefit_from_interpretable_vision_model.tar.gz
fi
tar -xvf do_users_benefit_from_interpretable_vision_model.tar.gz

(
    mkdir -p data
    cd data
    if [ ! -e "two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15.tar" ]; then
        wget https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15.tar
    fi
    tar -xvf two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15.tar

    if [ ! -e "two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15_unbiased.tar" ]; then
        wget https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15_unbiased.tar
    fi
    tar -xvf two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15_unbiased.tar
)
