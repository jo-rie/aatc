from pathlib import Path
from typing import Tuple, Union

from matplotlib import pyplot as plt
import matplotlib as mpl


layout = 'diss'
# layout = 'paper'

MPL_ALPHA = .8
MPL_S = .5

cycler_sym = mpl.cm.get_cmap('PRGn')  # plt.cm.PRGn  # Symmetric plot colors
cycler_01 = mpl.cm.get_cmap('YlGn')  # 0-1 plot colors
plot_style_4q = {'s': MPL_S, 'alpha': MPL_ALPHA}
list_of_cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

if layout == 'paper':
    # Widths and heights
    # text_width_pts = 468 # Elsevier
    # text_width_pts = 441 # De Gruyter
    # text_width_pts = 441 # Standard article
    text_width_pts = 371 # Springer nature
    pts_to_inch = 1 / 72.27
    text_width = text_width_pts * pts_to_inch
    default_fig_height = text_width / 3.5
    default_fig_width = text_width
    fig_factor_horizontally = 1.05  # Additional space for room between figures
    text_height = text_width * 1.5
    fig_factor_vertically = 1.4  # Additional space for room between figures for caption etc.


    def update_mpl_rcparams():
        plt.rcParams.update({
            'figure.dpi': 600,
            "text.usetex": True,
            'font.size': 5,
            "font.family": "serif",
            # "font.serif": ["Palatino"],
            "figure.figsize": (default_fig_width, default_fig_height),
            'axes.labelsize': 5,
            'legend.fontsize': 5,
        })
        mpl.use('pgf')
elif layout == 'diss':
    # Dissertation version
    text_width_pts = 418
    pts_to_inch = 1 / 72.27
    text_width = text_width_pts * pts_to_inch
    default_fig_height = text_width / 3.1
    default_fig_width = text_width
    fig_factor_horizontally = 1.05  # Additional space for room between figures
    fig_factor_vertically = 1.4  # Additional space for room between figures for caption etc.


    def update_mpl_rcparams() -> None:
        plt.rcParams.update({
            'figure.dpi': 600,
            "text.usetex": True,
            'font.size': 4,
            # "font.family": "serif",
            # "font.serif": ["Palatino"],
            'pgf.preamble': r'\usepackage{unicode-math,fontspec}\setmathfont[Extension={.otf},Path=/System/Library/Fonts/Supplemental/]{STIXTwoMath}\setsansfont[Path=/System/Library/Fonts/Supplemental/]{STIXTwoText}',
            'pgf.rcfonts': False,
            "figure.figsize": (default_fig_width, default_fig_height),
            'axes.labelsize': 5,
            'legend.fontsize': 6,
        })
        mpl.use('pgf')


def fig_with_size(nb_horizontally=1, nb_vertically=1, fig_height=None,
                  fig_width=None, factor_height=None, layout='constrained') -> Tuple[plt.Figure, plt.Axes]:
    """Return a figure so that nb_horizontally fit next to each other and nb_horizontally fit below each other"""
    if fig_height is None:
        if factor_height is None:
            if nb_vertically == 1:
                fig_height = default_fig_height
            else:
                fig_height = text_height / (nb_vertically * fig_factor_vertically)
        else:
            fig_height = default_fig_height * factor_height
    if fig_width is None:
        fig_width = default_fig_width / (nb_horizontally * fig_factor_horizontally)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout=layout)
    return fig, ax


def save_fig(fig: plt.Figure, path: Union[str, Path]):
    """
        The save_fig function saves a figure to the specified path.

        :param fig: plt.Figure: Specify that the function expects a matplotlib figure object
        :param path: str: Specify the path where the figure should be saved
        :return: A figure
        :doc-author: Trelent
        """
    # fig.tight_layout() # Use constrained layout instead
    fig.savefig(path)
    plt.close(fig)




