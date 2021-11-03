"""Module with samplers of the two4two dataset."""

from __future__ import annotations

import copy
import dataclasses
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import two4two
from two4two import bias
from two4two import utils


def _rescale_cmap(value: float) -> float:
    if value > 0.5:
        return 0.9 * (value - 0.5) + 0.5
    else:
        return value


def arm_position_center() -> dict[str, bias.Continouos]:
    return {
        'peaky': utils.truncated_normal(mean=0.5, std=0.3, lower=0, upper=0.52),
        'stretchy': utils.truncated_normal(mean=0.5, std=0.3, lower=0.48, upper=1.0)
    }


def arm_position_outwards() -> dict[str, bias.Continouos]:
    return {
        'peaky': utils.truncated_normal(mean=0.0, std=0.3, lower=0, upper=0.52),
        'stretchy': utils.truncated_normal(mean=1.0, std=0.3, lower=0.48, upper=1.0)
    }


def arm_position_uniform() -> dict[str, bias.Continouos]:
    return {
        'peaky': scipy.stats.uniform(0.0, 0.52),
        'stretchy': scipy.stats.uniform(0.48, 1 - 0.48),
    }


@dataclasses.dataclass
class SamplerBase(bias.Sampler):
    arm_position: bias.Continouos = dataclasses.field(
        default_factory=arm_position_center)

    def get_obj_color_rgba(self, obj_color: float) -> two4two.utils.RGBAColor:
        cmap = plt.get_cmap(self.obj_color_map)
        return tuple(cmap(_rescale_cmap(obj_color)))  # type: ignore

    def get_bg_color_rgba(self, bg_color: float) -> two4two.utils.RGBAColor:
        cmap = plt.get_cmap(self.bg_color_map)
        return tuple(cmap(_rescale_cmap(bg_color)))  # type: ignore

    def _sample_biased_but_not_predictive(
        self,
        params: two4two.SceneParameters,
        distribution: bias.Distribution,
        uniform_prob: float = 0.,
        intervention: bool = False,
    ) -> float:
        """Sample value not predictive."""

        if intervention:
            param_copy = copy.deepcopy(params)
            self.sample_arm_position(param_copy, intervention=True)
            arm_position = param_copy.arm_position
            obj_name = self._sample_name()
        else:
            arm_position = params.arm_position
            obj_name = params.obj_name

        value = self._sample(obj_name, distribution)
        if arm_position < 0.45:
            value = self._sample_truncated(
                obj_name, distribution, max=0.5)

        if arm_position > 0.55:
            value = self._sample_truncated(
                obj_name, distribution, min=0.5)

        if np.random.uniform() <= uniform_prob:
            value = self._sample(obj_name, distribution)

        return value


def sample_biased_and_predictive(
    params: two4two.SceneParameters,
    slope: float = 2.5,
    value_at_05: float = 0.25,
    uniform_prob: float = 0.2,
    intervention: bool = False,
) -> float:
    if intervention:
        if params.arm_position > 0.52:
            is_a_stretchy = True
        elif params.arm_position < 0.48:
            is_a_stretchy = False
        else:
            is_a_stretchy = np.random.uniform() < 0.5
    else:
        is_a_stretchy = params.obj_name == 'stretchy'

    if is_a_stretchy:
        offset = params.arm_position - 0.5
        value = 1 - np.random.uniform(0, np.clip(value_at_05 + slope * offset, 0, 1))
    else:
        offset = 0.5 - params.arm_position
        value = np.random.uniform(0, np.clip(value_at_05 + slope * offset, 0, 1))

    if np.random.uniform(0, 1) < uniform_prob:
        value = np.random.uniform(0, 1)
    return value


@dataclasses.dataclass
class SamplerDoubleColors(SamplerBase):

    obj_rotation_pitch_max_value: float = np.pi / 6
    obj_rotation_yaw_range: float = 2 * np.pi
    obj_color_uniform_prob: float = 0.2
    bg_color_uniform_prob: float = 0.2
    spherical_uniform_prob: float = 0.2
    arm_position_type: str = 'centered'

    obj_rotation_pitch: bias.Continouos = two4two.utils.truncated_normal(
        0, 0.5 * np.pi / 4,
        -np.pi / 6, np.pi / 6)

    bg_color: bias.Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': two4two.utils.truncated_normal(mean=1., std=0.4, lower=0, upper=1),
            'stretchy': two4two.utils.truncated_normal(mean=0, std=0.4, lower=0, upper=1.0)
        })

    def __post_init__(self):
        if self.arm_position_type == 'uniform':
            self.arm_position = arm_position_uniform()
        elif self.arm_position_type == 'centered':
            self.arm_position = arm_position_center()
        else:
            raise ValueError()

    def sample_obj_color(
        self,
        params: two4two.SceneParameters,
        intervention: bool = False,
    ):
        value = sample_biased_and_predictive(
            params, intervention=intervention,
            uniform_prob=self.obj_color_uniform_prob,

        )
        params.obj_color = float(value)
        params.obj_color_rgba = self.get_obj_color_rgba(params.obj_color)
        params.mark_sampled('obj_color')

    def sample_spherical(self, params: two4two.SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        params.spherical = self._sample_biased_but_not_predictive(
            params, self.spherical, self.spherical_uniform_prob, intervention)
        params.mark_sampled('spherical')

    def sample_bg_color(self, params: two4two.SceneParameters, intervention: bool = False):
        """Samples the ``bg_color_rgba`` and ``bg_color``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        params.bg_color = 1 - sample_biased_and_predictive(
            params, intervention=intervention,
            uniform_prob=self.bg_color_uniform_prob,
        )
        params.bg_color_rgba = self.get_bg_color_rgba(params.bg_color)
        params.mark_sampled('bg_color')


@dataclasses.dataclass
class SamplerDoubleColorsArmsUniform(SamplerDoubleColors):
    arm_position: bias.Continouos = dataclasses.field(
        default_factory=arm_position_uniform)


@dataclasses.dataclass
class SamplerDoubleColorsArmsUniformLessYawRotation(SamplerDoubleColors):
    arm_position: bias.Continouos = dataclasses.field(
        default_factory=arm_position_uniform)

    obj_rotation_yaw: bias.Continouos = scipy.stats.uniform(-np.pi / 4, np.pi / 2)


@dataclasses.dataclass
class SamplerSingleColor(SamplerDoubleColors):
    bg_color: bias.Continouos = scipy.stats.uniform(0, 1)

    def sample_bg_color(self, params: two4two.SceneParameters, intervention: bool = False):
        """Samples the ``bg_color_rgba`` and ``bg_color``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.bg_color = float(self._sample(obj_name, self.bg_color))
        params.bg_color_rgba = tuple(  # type: ignore
            self._bg_cmap(params)(_rescale_cmap(params.bg_color)))  # type: ignore
        params.mark_sampled('bg_color')


@dataclasses.dataclass
class SamplerSingleColorPastelBG(SamplerSingleColor):
    bg_color: bias.Continouos = scipy.stats.uniform(0, 1)
    bg_color_map: str = 'Pastel2'

    def sample_bg_color(self, params: two4two.SceneParameters, intervention: bool = False):
        """Samples the ``bg_color_rgba`` and ``bg_color``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.bg_color = float(self._sample(obj_name, self.bg_color))
        params.bg_color_rgba = tuple(  # type: ignore
            self._bg_cmap(params)(params.bg_color))  # type: ignore
        params.mark_sampled('bg_color')


@dataclasses.dataclass
class SamplerUnbiased(SamplerSingleColor):
    def sample_spherical(self, params: two4two.SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.spherical = self._sample(obj_name, self.spherical)
        params.mark_sampled('spherical')

    def sample_obj_color(
        self,
        params: two4two.SceneParameters,
        intervention: bool = False,
    ):
        super().sample_obj_color(params, intervention=True)

    def sample_bg_color(
        self,
        params: two4two.SceneParameters,
        intervention: bool = False,
    ):
        super().sample_bg_color(params, intervention=True)


@dataclasses.dataclass
class SamplerSpherical(SamplerBase):
    obj_rotation_pitch: bias.Continouos = two4two.utils.truncated_normal(
        0, 0.5 * np.pi / 4,
        -np.pi / 6, np.pi / 6)

    spherical: bias.Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': two4two.utils.truncated_normal(mean=1., std=0.4, lower=0, upper=1),
            'stretchy': two4two.utils.truncated_normal(mean=0, std=0.4, lower=0, upper=1.0)
        })

    def sample_obj_color(self, params: two4two.SceneParameters, intervention: bool = False):
        obj_name = self._sample_name() if intervention else params.obj_name

        slope = 2.5
        value_at_05 = 0.25
        if obj_name == 'stretchy':
            offset = params.arm_position - 0.5
            value = 1 - np.random.uniform(0, np.clip(value_at_05 + slope * offset, 0, 1))
        elif obj_name == 'peaky':
            offset = 0.5 - params.arm_position
            value = np.random.uniform(0, np.clip(value_at_05 + slope * offset, 0, 1))

        if np.random.uniform(0, 1) < 0.2:
            value = np.random.uniform(0, 1)

        params.obj_color = float(value)
        params.obj_color_rgba = tuple(self._object_cmap(params)(  # type: ignore
            _rescale_cmap(params.obj_color)))  # type: ignore
        params.mark_sampled('obj_color')

    def sample_bg_color(self, params: two4two.SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        params.bg_color = self._sample_biased_but_not_predictive(
            params, self.bg_color, intervention)
        params.bg_color_rgba = tuple(  # type: ignore
            self._bg_cmap(params)(_rescale_cmap(params.bg_color)))  # type: ignore
        params.mark_sampled('bg_color')


@dataclasses.dataclass
class SamplerObjColorAndSpherical(SamplerBase):
    obj_rotation_pitch_max_value: float = np.pi / 6
    obj_rotation_yaw_range: float = 2 * np.pi
    obj_color_uniform_prob: float = 0.2
    spherical_uniform_prob: float = 0.2
    obj_color_slope: float = 2.5
    obj_color_value_at_05: float = 0.25

    arm_position: bias.Continouos = dataclasses.field(
        default_factory=arm_position_uniform)

    obj_rotation_pitch: bias.Continouos = two4two.utils.truncated_normal(
        0, 0.5 * np.pi / 4,
        -np.pi / 6, np.pi / 6)

    bg_color: bias.Continouos = scipy.stats.uniform(0, 1)

    def sample_obj_color(
        self,
        params: two4two.SceneParameters,
        intervention: bool = False,
    ):
        value = sample_biased_and_predictive(
            params, intervention=intervention,
            uniform_prob=self.obj_color_uniform_prob,
            slope=self.obj_color_slope,
            value_at_05=self.obj_color_value_at_05,
        )
        params.obj_color = float(value)
        params.obj_color_rgba = self.get_obj_color_rgba(params.obj_color)
        params.mark_sampled('obj_color')

    def sample_spherical(self, params: two4two.SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        params.spherical = self._sample_biased_but_not_predictive(
            params, self.spherical, self.spherical_uniform_prob, intervention)
        params.mark_sampled('spherical')


def clone(
    param: two4two.SceneParameters,
    obj_name: Optional[str] = None,
    labeling_error: Optional[bool] = None,
    spherical: Optional[float] = None,
    bending: Optional[float] = None,
    obj_rotation_roll: Optional[float] = None,
    obj_rotation_pitch: Optional[float] = None,
    obj_rotation_yaw: Optional[float] = None,
    fliplr: Optional[bool] = None,
    position_x: Optional[float] = None,
    position_y: Optional[float] = None,
    arm_position: Optional[float] = None,
    obj_color: Optional[float] = None,
    # When Optional[passing] 0None,
    obj_color_rgba: Optional[utils.RGBAColor] = None,
    bg_color: Optional[float] = None,
    bg_color_rgba: Optional[utils.RGBAColor] = None,
    resolution: Optional[tuple[int, int]] = None,
    id: Optional[str] = None,
    original_id: Optional[str] = None
) -> two4two.SceneParameters:
    """Clones and overwrites some parameters."""

    new_param = param.clone()
    # ensure we do not accidentially change the original params
    del param

    if obj_name is not None:
        new_param.obj_name = obj_name

    if labeling_error is not None:
        new_param.labeling_error = labeling_error

    if spherical is not None:
        new_param.spherical = spherical

    if bending is not None:
        new_param.bending = bending

    if obj_rotation_roll is not None:
        new_param.obj_rotation_roll = obj_rotation_roll

    if obj_rotation_pitch is not None:
        new_param.obj_rotation_pitch = obj_rotation_pitch

    if obj_rotation_yaw is not None:
        new_param.obj_rotation_yaw = obj_rotation_yaw

    if fliplr is not None:
        new_param.fliplr = fliplr

    if position_x is not None:
        new_param.position_x = position_x

    if position_y is not None:
        new_param.position_y = position_y

    if arm_position is not None:
        new_param.arm_position = arm_position

    if obj_color is not None:
        new_param.obj_color = obj_color

    if obj_color_rgba is not None:
        new_param.obj_color_rgba = obj_color_rgba

    if bg_color is not None:
        new_param.bg_color = bg_color

    if bg_color_rgba is not None:
        new_param.bg_color_rgba = bg_color_rgba

    if resolution is not None:
        new_param.resolution = resolution

    if id is not None:
        new_param.id = id

    if original_id is not None:
        new_param.original_id = original_id

    return new_param
