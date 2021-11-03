"""Module with functions for plotting."""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import io
import os
import shutil
from typing import Any, Optional, Sequence, Union


# from IPython import get_ipython
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import seaborn as sns
import torch


def get_figure_size(
    fraction: float = 0.5,
    width: float = 397.48499,  # iclr textwidth
    ratio: float = (5 ** 0.5 - 1) / 2,  # gold ratio
    subplots: tuple[int, int] = (1, 1),
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Args:
        width: float or string
                Document width in points, or string of predined document type
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy
        ratio: Ratio of plot
        subplots: array-like, optional
                The number of rows and columns of subplots.

    Returns:
        fig_dim: tuple Dimensions of figure in inches
    """

    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "pnas":
        width_pt = 246.09686
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def get_latex_defs(**dictionary: dict[str, str]) -> str:
    defs = ""
    for key, val in dictionary.items():
        if "_" in key:
            key = underscore_to_camelcase(key)
        assert key.isalpha(), f"not a valid latex macro name: {key}"
        defs += f"\\newcommand{{\\{key}}}{{{val}}}\n"
    return defs


def equal_vmin_vmax(x: np.ndarray) -> dict[str, float]:
    vmax = float(np.abs(x).max())
    return {"vmin": -vmax, "vmax": vmax}


def export_latex_defs(filename: str, **dictionary: dict[str, str]):
    with open(filename, "w") as f:
        f.write(get_latex_defs(**dictionary))


def underscore_to_camelcase(value: str) -> str:
    def camelcase():
        yield str
        while True:
            yield str.capitalize

    c = camelcase()
    return "".join(next(c)(x) if x else "_" for x in value.split("_"))


@contextlib.contextmanager
def latexify(
    dark_gray: str = ".15",
    light_gray: str = ".8",
    small_size: int = 8,
    tiny_size: int = 7,
    linewidth_thin: float = 0.33,
    linewidth: float = 0.5,
    n_colors: Optional[int] = None,
):
    style = latex_style()
    with mpl_style(style), sns.color_palette("colorblind", n_colors=n_colors):
        yield


@contextlib.contextmanager
def mpl_style(style: dict[str, Any]):
    mpl_orig = copy.deepcopy(mpl.rcParams)
    for key, value in style.items():
        mpl.rcParams[key] = value
    try:
        yield
    finally:
        for key, value in mpl_orig.items():
            mpl.rcParams[key] = value


def latex_style(
    dark_gray: str = ".15",
    light_gray: str = ".8",
    small_size: int = 8,
    tiny_size: int = 7,
    linewidth_thin: float = 0.33,
    linewidth: float = 0.5,
) -> dict[str, Any]:
    # Common parameters
    return {
        "figure.facecolor": "white",
        "axes.labelcolor": dark_gray,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": dark_gray,
        "ytick.color": dark_gray,
        "axes.axisbelow": True,
        "grid.linestyle": "-",
        "text.color": dark_gray,
        # "font.family": ["serif"],
        # "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans",
        #                     "Bitstream Vera Sans", "sans-serif"],
        "font.family": "serif",
        # "font.serif": ['Times', "DejaVu Sans"],
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "image.cmap": "rocket",
        "xtick.top": False,
        "ytick.right": False,
        "axes.facecolor": "white",
        "xtick.major.width": linewidth,
        "xtick.minor.width": linewidth,
        "ytick.major.width": linewidth,
        "ytick.minor.width": linewidth,
        "grid.linewidth": linewidth_thin,
        "axes.linewidth": linewidth_thin,
        "lines.linewidth": linewidth,
        "lines.markersize": linewidth_thin,
        "patch.linewidth": linewidth_thin,
        "xtick.bottom": True,
        "ytick.left": True,
        "axes.facecolor": "white",
        "axes.edgecolor": dark_gray,
        "grid.color": light_gray,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.labelsize": small_size,
        "font.size": small_size,
        "axes.titlesize": small_size,
        "legend.fontsize": small_size,
        "xtick.labelsize": tiny_size,
        "ytick.labelsize": tiny_size,
    }


def display_pdf(fname: str, iframe_size: tuple[int, int] = (800, 400)):
    from IPython.display import display, IFrame

    display(IFrame(fname, *iframe_size))


def savefig_pdf(
    fname: str,
    figure: mpl.figure.Figure,
    display: bool = False,
    iframe_size: tuple[int, int] = (800, 400),
    **kwargs: Any,
):
    figure.savefig(fname, bbox_inches="tight", pad_inches=0.01, **kwargs)
    if display:
        display_pdf(fname, iframe_size)


def savefig_pgf(
    fname: str,
    figure: mpl.figure.Figure,
    display: bool = False,
    iframe_size: tuple[int, int] = (800, 400),
    pdf: bool = True,
    **kwargs: Any,
):
    figure.savefig(fname, bbox_inches="tight", pad_inches=0.0, **kwargs)
    if pdf or display:
        pdf_fname, _ = os.path.splitext(fname)
        pdf_fname += ".pdf"
        savefig_pdf(pdf_fname, figure, **kwargs)
    if display:
        display_pdf(pdf_fname)


# grid


@dataclasses.dataclass
class FigureSizes:
    figwidth: float
    figheight: float
    item_size: float
    padding_w: float
    padding_h: float
    label_size: float
    label_position: str
    ticks_left_size: float
    ticks_top_size: float
    n_rows: int
    n_cols: int


# def calculate_protypes_sizes(
def calculate_sizes(
    figwidth: float,
    n_rows: int,
    n_cols: int,
    label_size: float = 0.0,
    label_position: str = "left",
    pad_ratio_w: float = 0.075,
    pad_ratio_h: float = 0.075,
    odd: bool = False,
) -> FigureSizes:
    if label_position == "left":
        pair_size_w_pad = (figwidth - label_size) / (n_cols // 2)
    else:
        pair_size_w_pad = (figwidth) / (n_cols // 2)

    padding_w = pad_ratio_w * pair_size_w_pad
    padding_h = pad_ratio_h * pair_size_w_pad

    pair_size = pair_size_w_pad - padding_w
    item_size = pair_size / 2
    figheight = n_rows * (item_size + padding_h)
    if label_position == "top":
        figheight = figheight + label_size

    # TODO: convert to dataclass
    return FigureSizes(
        figwidth=figwidth,
        figheight=figheight,
        item_size=item_size,
        padding_w=padding_w,
        padding_h=padding_h,
        label_size=label_size,
        label_position=label_position,
        ticks_left_size=0.0,
        ticks_top_size=0.0,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def adapt_figsize(
    sizes: FigureSizes,
    n_rows: int,
    n_cols: int,
    label_size: float = None,
    ticks_left_size: Optional[float] = None,
    ticks_top_size: Optional[float] = None,
    padding_w: Optional[float] = None,
) -> FigureSizes:
    sizes = copy.deepcopy(sizes)

    if label_size is None:
        label_size = sizes.label_size
    if ticks_left_size is None:
        ticks_left_size = sizes.ticks_left_size
    if ticks_top_size is None:
        ticks_top_size = sizes.ticks_top_size

    if padding_w is not None:
        sizes.padding_w = padding_w

    sizes.figwidth = n_cols // 2 * (2 * sizes.item_size + sizes.padding_w)
    sizes.figheight = n_rows * (sizes.item_size + sizes.padding_h)

    sizes.label_size = label_size

    if sizes.label_position == "left":
        sizes.figwidth += label_size
    elif sizes.label_position == "top":
        sizes.figheight += label_size

    sizes.figwidth += ticks_left_size
    sizes.figheight += ticks_top_size

    sizes.ticks_left_size = ticks_left_size
    sizes.ticks_top_size = ticks_top_size

    sizes.n_rows = n_rows
    sizes.n_cols = n_cols
    return sizes


def plot_grid(
    image_grid: torch.Tensor,
    sizes: FigureSizes,
    logit_name: str = None,
    fontsize: int = 10,
    borderwidth: float = 0.5,
    plus_label: str = "$+$",
    minus_label: str = "$-$",
    logit_values: list[str] = [],
    pca_names: list[Union[str, tuple[str, str]]] = [],
) -> tuple[mpl.figure.Figure, list[list[mpl.axes.Axes]]]:
    n_rows = len(image_grid)
    n_cols = len(image_grid[0])
    n_pca_vecs = (n_cols + 1) // 2

    padding_w = sizes.padding_w
    padding_h = sizes.padding_h
    item_size = sizes.item_size
    figheight = sizes.figheight
    figwidth = sizes.figwidth

    ticks_left_size = sizes.ticks_left_size
    ticks_top_size = sizes.ticks_top_size
    label_size = sizes.label_size

    fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=False)
    fig.subplots_adjust(top=1, bottom=0, right=1.0, left=0.0, hspace=0, wspace=0)

    def flatten(list: Any) -> Any:
        return [item for sublist in list for item in sublist]

    def grid_idx(i: int, j: int) -> tuple[int, int]:
        return (2 * i + 1, (j // 2) * 3 + j % 2 + 1 + 1)

    widths = [label_size, ticks_left_size] + flatten(
        [(item_size, item_size, padding_w) for _ in range(n_pca_vecs)]
    )

    # remove if odd
    if n_cols % 2 != 0:
        widths[:-1]

    heights = [ticks_top_size] + flatten(
        [(item_size, padding_h) for _ in range(n_rows)]
    )

    # remove last paddings
    heights = heights[:-1]
    widths = widths[:-1]

    gridspec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.00,
        hspace=0.0,
        figure=fig,
    )

    if logit_name is not None:
        ax = fig.add_subplot(gridspec[1:, 0])
        ax.set_axis_off()
        ax.text(
            0,
            0.9,
            minus_label,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.text(
            0,
            0.5,
            logit_name,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.text(
            0,
            0.1,
            plus_label,
            # weight='bold',
            fontsize=fontsize,
            rotation=90,
            horizontalalignment="center",
            verticalalignment="center",
        )

        ax.set_ylim(0, 1)

    for idx, val in enumerate(logit_values):
        r, c = grid_idx(idx, 0)
        ax = fig.add_subplot(gridspec[r, 1])
        ax.set_axis_off()
        ax.text(
            0.25,
            0.5,
            val,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

    for idx, name in enumerate(pca_names):
        r, c = grid_idx(0, 2 * idx)
        if isinstance(name, str):
            ax = fig.add_subplot(gridspec[0, c : c + 2])
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, name, horizontalalignment="center", verticalalignment="center"
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
        elif isinstance(name, (list, tuple)):
            for offset, pc_name in enumerate(name):
                ax = fig.add_subplot(gridspec[0, c + offset : c + 1 + offset])
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    pc_name,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.set_ylim(0, 1)
                ax.set_xlim(0, 1)
        else:
            raise ValueError()

    img_axes: list[list[mpl.axes.Axes]] = []
    with torch.no_grad():
        for row_i, imgs_row in enumerate(image_grid):
            img_axes.append([])
            for col_i, img in enumerate(imgs_row):
                gi, gj = grid_idx(row_i, col_i)
                ax = fig.add_subplot(gridspec[gi, gj])
                img = img.cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                img_axes[-1].append(ax)
                [i.set_linewidth(borderwidth) for i in ax.spines.values()]

    return fig, img_axes


@contextlib.contextmanager
def overwrite_dir(*path: str, upload: bool = False):
    fullpath = os.path.join(*path)
    if os.path.exists(fullpath):
        shutil.rmtree(fullpath)
    os.makedirs(fullpath)
    yield fullpath
    if upload:
        raise NotImplementedError()


def get_svg_with_new_image(
    image_path: str,
    svg_path: str,
) -> bytes:
    with open(svg_path) as f:
        base_svg = f.read()

    byte_img_io = io.BytesIO()

    img = PIL.Image.open(image_path)
    img.save(byte_img_io, "PNG")
    byte_img_io.seek(0)
    img_bytes = byte_img_io.read()

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    new_svg = []
    for line in base_svg.splitlines():
        if "base64" in line:
            new_svg.append(f'     xlink:href="data:image/png;base64,{img_base64}"')
            # print(line[:40], line[-10:])
            # print(new_svg[-1][:40], new_svg[-1][-10:])
        else:
            new_svg.append(line)
    return "\n".join(new_svg).encode("utf-8")


def plot_images_in_grid(
    images: list[np.ndarray],
    n_cols: int = 5,
    n_rows: Optional[int] = None,
    golden_frame_at: Optional[tuple[int, int]] = None,
    hspace: float = 0.0,
    wspace: float = 0.0,
    border_width: float = 0.33,
) -> tuple[mpl.figure.Figure, Sequence[Sequence[mpl.axes.Axes]],]:
    """Plots images in grids.

    Args:
        images: List of images to plot (np.ndarray with shape (h, w, 3)
        n_cols: Number of columns
        n_rows: Number of rows. If not given inferred from `n_cols`
        golden_frame_at: Put a golden frame around one image.
        hspace, wspace: arguments for fig.subplots_adjust

    Returns:
        (figure, axes)
    """

    if n_rows is None:
        n_rows = len(images) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), squeeze=False)
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        [i.set_linewidth(border_width) for i in ax.spines.values()]
        [i.set_color("black") for i in ax.spines.values()]

    if golden_frame_at is not None:
        ax = axes[golden_frame_at]
        ax.set_zorder(1000)
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        [i.set_color("gold") for i in ax.spines.values()]

    fig.subplots_adjust(wspace=wspace, hspace=hspace, left=0, right=1, bottom=0, top=1)
    fig.set_dpi(300)
    return fig, axes


two4two_nice_names = {
    "obj_name": "Type",
    "arm_position": "Legs' Position",
    "spherical": "Shape",
    "obj_color": "Color",
    "obj_rotation_yaw": "Rotation Yaw",
    "obj_rotation_roll": "Rotation Roll",
    "obj_rotation_pitch": "Rotation Pitch",
    "position_x": "Position X",
    "position_y": "Position Y",
    "bg_color": "Background",
    "bending": "Bending",
}
