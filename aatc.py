from typing import Union, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from matplotlib import patches
from scipy import stats
from scipy.stats import DegenerateDataWarning
from statsmodels.tools.validation import array_like

from python_utils.mpl_helpers import fig_with_size, plot_style_4q
from setup import config


class ExclusionArea:
    def __init__(self, eps_x=0, eps_y=0, combination='and'):
        """
        Initialize an ExclusionArea instance. The ExclusionArea is within the epsilon values for x and y and the
        combination logic (if 'and' a point lies out of the exclusion area if both values are at least eps_y/eps_x; if
        'or' a point is out the exclusion area if at least one point fullfills with in the condition). Note that for a
        exclusion area in only one coordinate, the other epsilon value should be set to 0 and combination should be
        'and'.

        :param eps_x: The epsilon value for x values.
        :param eps_y: The epsilon value for y values.
        :param combination: The logic to use when checking if a point is in the exclusion area. Must be 'or' or 'and'.
        """
        self.eps_x = eps_x
        self.eps_y = eps_y
        self.combination = combination

    @classmethod
    def from_quantile(cls, diffy, diffx, q, combination='or'):
        """
        Create an ExclusionArea instance from a quantile of the data.

        :param diffy: vector of differences of realisations
        :param diffx: vector of differences of nowcasts
        :param q: The quantile to calculate. Must be between 0 and 1 or None; then an empty exclusion area is returned.
        :param combination: The logic to use when checking if a point is in the exclusion area. Must be 'or' or 'and'. If 'or', the exclusion area is rectangular with twice the length of the corresponding quantile; if 'and', the exclusion area is a cross.
        :return: An ExclusionArea instance with eps_x and eps_y set to the q-quantile of the data.
        """
        if q is None:
            return cls()
        eps_y = np.nanquantile(np.abs(diffy), q)
        eps_x = np.nanquantile(np.abs(diffx), q)
        return cls(eps_x, eps_y, combination)

    def get_mask(self, diffy, diffx):
        """
        Get a mask for diffy and diffx vectors.

        The mask is False for all points that are in the exclusion area.

        :param diffy: The diffy vector.
        :param diffx: The diffx vector.
        :return: A mask that is False for all points that are in the exclusion area.
        """
        if self.combination == 'or':
            mask = (np.abs(diffy) >= self.eps_y) | (np.abs(diffx) >= self.eps_x)
        elif self.combination == 'and':
            mask = (np.abs(diffy) >= self.eps_y) & (np.abs(diffx) >= self.eps_x)
        else:
            raise ValueError("combination must be 'or' or 'and'")
        # If either diffy or diffx is nan, the corresponding mask value should be False
        mask = mask & (~np.isnan(diffy)) & (~np.isnan(diffx))
        return mask

    def get_mask_diffy(self, diffy):
        """
        Get a mask for diffy vector.

        The mask is False for all points that are in the exclusion area.

        :param diffy: The diffy vector.
        :return: A mask that is False for all points that are in the exclusion area.
        """
        mask = np.abs(diffy) >= self.eps_y
        # If diffy is nan, the corresponding mask value should be False
        mask = mask & (~np.isnan(diffy))
        return mask

    def get_mask_diffx(self, diffx):
        """
        Get a mask for diffx vector.

        The mask is False for all points that are in the exclusion area.

        :param diffx: The diffx vector.
        :return: A mask that is False for all points that are in the exclusion area.
        """
        mask = np.abs(diffx) >= self.eps_x
        # If diffx is nan, the corresponding mask value should be False
        mask = mask & (~np.isnan(diffx))
        return mask

    def get_list_for_plotting(self) -> list[tuple[float, float]]:
        """
        Get a list of exclusion area rectangles (width / 2, height / 2) centered in the zero point for plotting.

        :return: A list of the exclusion area for plotting.
        """
        if self.combination == 'and':
            return [(np.inf, self.eps_x), (self.eps_y, np.inf)]
        elif self.combination == 'or':
            return [(self.eps_y, self.eps_x)]
        else:
            raise ValueError(f'Combination {self.combination} invalid.')

    def __repr__(self):
        return f'ExclusionArea(eps_x={self.eps_x}, eps_y={self.eps_y}, combination={self.combination})'

    def __str__(self):
        return f'ExclusionArea(eps_x={self.eps_x}, eps_y={self.eps_y}, combination={self.combination})'


def plot_4q(diffy: np.ndarray, diffx: np.ndarray, ax: Union[None, plt.Axes] = None,
            excl_area: ExclusionArea = None, include_time: bool = False, plot_points_in_excl_area: bool = True):
    """ create 4Q plot from the data in delta_y (on x-Axis) and delta_x (on y-Axis)

    :param include_time: if True, the points are colored according to their index
    :param excl_area: Exclusion Area object
    :param diffy: numpy-array with differences of realisations
    :param diffx: numpy-array with differences of nowcasts
    :param ax: Axes-object to contain the plot (optional)
    :param plot_points_in_excl_area: if True, points in the exclusion area are plotted
    :return: the Axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
    if not include_time:
        ax.scatter(diffy, diffx, **plot_style_4q, zorder=3)
    else:
        ax.scatter(diffy, diffx, **plot_style_4q, zorder=3, c=np.arange(len(diffy)), cmap='winter')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lim_max_abs = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) * 2

    def rec_from_0(width, height, col):
        return patches.Rectangle((0, 0), width, height, facecolor=col, edgecolor=None, zorder=1)

    for (w, h, col) in zip([xmin, xmax, xmin, xmax], [ymin, ymax, ymax, ymin],
                           [config['gr'], config['gr'], config['red'], config['red']]):
        ax.add_patch(rec_from_0(w * 2, h * 2, col))

    # Plot grid better visible
    ax.axvline(0, color='black', linewidth=0.2, zorder=1.6)
    ax.axhline(0, color='black', linewidth=0.2, zorder=1.6)
    ax.plot(np.linspace(-lim_max_abs, lim_max_abs, 2), np.linspace(-lim_max_abs, lim_max_abs, 2),
            color='grey', linewidth=0.2, linestyle='dashed', zorder=1.6, dashes=[6, 4])
    ax.plot(np.linspace(-lim_max_abs, lim_max_abs, 2), - np.linspace(-lim_max_abs, lim_max_abs, 2),
            color='grey', linewidth=0.2, linestyle='dashed', zorder=1.6, dashes=[6, 4])

    if excl_area is not None:
        for excl_area_el in excl_area.get_list_for_plotting():
            ax.add_patch(patches.Rectangle(
                (-min(excl_area_el[0], lim_max_abs), -min(excl_area_el[1], lim_max_abs)),
                min(excl_area_el[0], lim_max_abs) * 2, min(excl_area_el[1], lim_max_abs) * 2,
                facecolor='lightgrey', edgecolor=None, linewidth=0.1, zorder=1.5))

    # ax.scatter(delta_y, delta_x, **plot_style_4q)

    # Plot Quadrants in Background
    # ax.autoscale(False, tight=True)
    # extent = (delt_x.min())
    # arr = np.array([[1, 0], [0, 1]])
    # ax.imshow(arr, extent=extent, cmap='RdYlGn', interpolation='none', alpha=.1)

    ax.set(xlabel=r'$y^{\Delta}$', ylabel=r'$x^{\Delta}$', xlim=(xmin, xmax), ylim=(ymin, ymax))
    # ax.set_aspect('equal')
    return ax


def compute_cond_prob(delta_y: np.ndarray, delta_x: np.ndarray, x_bounds: tuple = None,
                   nels_linspace: int = 1000, excl_area: ExclusionArea = None,
                   bw_kde: Union[array_like, str] = None) -> Tuple[np.array, np.array]:
    if excl_area is not None:
        mask = excl_area.get_mask(delta_y, delta_x)
    else:
        mask = np.ones(len(delta_y), dtype=bool)

    dens_conditional = sm.nonparametric.KDEMultivariateConditional(
        endog=delta_y[mask], exog=delta_x[mask], dep_type='c', indep_type='c', bw=bw_kde)

    if x_bounds is None:
        x_linspace = np.linspace(np.nanquantile(delta_x, 0.05) - delta_x.std(),
                                 np.nanquantile(delta_x, 0.95) + delta_x.std(), nels_linspace)
    else:
        x_linspace = np.linspace(x_bounds[0], x_bounds[1], nels_linspace)
    cond_prob_estimate = dens_conditional.cdf(np.zeros(nels_linspace), x_linspace)
    x_linspace_positive = x_linspace > 0
    cond_prob_estimate[x_linspace_positive] = 1 - cond_prob_estimate[x_linspace_positive]

    return x_linspace, cond_prob_estimate


def plot_cond_prob(delta_y: np.ndarray = None, delta_x: np.ndarray = None, ax: Union[None, plt.Axes] = None, x_bounds: tuple = None,
                   nels_linspace: int = 1000, label: Union[None, str] = None, excl_area: ExclusionArea = None,
                   bw_kde: Union[array_like, str] = None,
                   x_linspace: np.ndarray = None, cond_prob_estimate: np.ndarray = None) -> plt.Axes:
    """ Plot conditional probability plot for delta_y and delta_x. If ax is not provided, a new plt.Axes is creates.
    nels_linspace corresponds to the number of elements along the x-axis for which the conditional probability is computed."""
    if (x_linspace is None) | (cond_prob_estimate is None):
        x_linspace, cond_prob_estimate = compute_cond_prob(delta_y=delta_y, delta_x=delta_x, x_bounds=x_bounds,
                                                       nels_linspace=nels_linspace, excl_area=excl_area, bw_kde=bw_kde)

    # mpl.use('pdf')
    if ax is None:
        fig, ax = fig_with_size(factor_height=1)

    if excl_area is not None:  # Exclude the area from -eps to eps if eps > 0
        cond_prob_estimate = np.ma.array(cond_prob_estimate)
        cond_prob_estimate[~excl_area.get_mask_diffx(x_linspace)] = np.ma.masked

    if label is None:
        ax.plot(x_linspace, cond_prob_estimate)
    else:
        ax.plot(x_linspace, cond_prob_estimate, label=label, alpha=0.5)

    return ax


def atc_ratio(diffy: np.ndarray, diffx: np.ndarray, excl_area: ExclusionArea = None) -> float:
    """ get share of points that have the same sign

    :param diffy: numpy-array with differences of realisations
    :param diffx: numpy-array with differences of nowcasts
    :return: the proportion of pairs that are concordant
    """
    if excl_area is None:
        excl_area = ExclusionArea()  # empty exclusion area that excludes points with nan values
    mask = excl_area.get_mask(diffy, diffx)
    return np.nanmean(diffx[mask] * diffy[mask] > 0)


def atc_ratio_pos(diffy: np.ndarray, diffx: np.ndarray, excl_area: ExclusionArea = None) -> float:
    """ get share of points that have the same sign and for which diffx is positive and not in the exclusion area.
    :param diffy: numpy-array with differences of realisations
    :param diffx: numpy-array with differences of nowcasts
    :param excl_area: Exclusion Area object
    :return: the proportion of pairs that are concordant
    """
    if excl_area is None:
        excl_area = ExclusionArea()  # empty exclusion area that excludes points with nan values
    mask = excl_area.get_mask(diffy, diffx) & (diffx > 0)
    return np.nanmean(diffx[mask] * diffy[mask] > 0)


def atc_ratio_neg(diffy: np.ndarray, diffx: np.ndarray, excl_area: ExclusionArea = None) -> float:
    """ get share of points that have the same sign and for which diffx is positive and not in the exclusion area.
    :param diffy: numpy-array with differences of realisations
    :param diffx: numpy-array with differences of nowcasts
    :param excl_area: Exclusion Area object
    :return: the proportion of pairs that are concordant
    """
    if excl_area is None:
        excl_area = ExclusionArea()  # empty exclusion area that excludes points with nan values
    mask = excl_area.get_mask(diffy, diffx) & (diffx < 0)
    return np.nanmean(diffx[mask] * diffy[mask] > 0)


def atc_with_bootstrap(diffy: np.ndarray, diffx: np.ndarray, excl_area: ExclusionArea = None, n_bootstrap: int = 9999,
                       pos_neg: str = None, rng: np.random.Generator = None, confidence_level: float = 0.9,
                       bootstrap_method: str = 'BCa') -> tuple[float, float, float]:
    """Compute bootstrap confidence intervals for the ATC ratio.
    :param rng: numpy random number generator
    :param bootstrap_method: method to use for bootstrap confidence interval
    :param confidence_level: confidence level for the bootstrap confidence interval
    :param diffy: numpy-array with differences of realisations
    :param diffx: numpy-array with differences of nowcasts
    :param excl_area: Exclusion Area object
    :param n_bootstrap: number of bootstrap samples
    :param pos_neg: 'positive' or 'negative' to compute the ATC ratio for positive or negative nowcasts
    :return: the proportion of pairs that are concordant, the lower and upper bounds of the 95% confidence interval
    """
    if excl_area is None:
        excl_area = ExclusionArea()
    if rng is None:
        rng = np.random.default_rng()
    mask = excl_area.get_mask(diffy, diffx)
    if pos_neg is None:
        # Compute the ATC ratio for all points
        data = (diffy[mask], diffx[mask])
    elif pos_neg == 'positive':
        data = (diffy[mask & (diffx > 0)], diffx[mask & (diffx > 0)])
    elif pos_neg == 'negative':
        data = (diffy[mask & (diffx < 0)], diffx[mask & (diffx < 0)])
    else:
        raise ValueError('pos_neg must be "positive", "negative" or None')

    def tr_func(sample1, sample2):
        """Local ATC ratio function for boostrapping. The exclusion area and x conditions are already applied."""
        return (sample1 * sample2 > 0).mean()

    tr = tr_func(*data)
    bootstrap_sample = stats.bootstrap(data, statistic=tr_func, confidence_level=confidence_level,
                                       random_state=rng, vectorized=False, n_resamples=n_bootstrap,
                                       paired=True, method=bootstrap_method)
    return tr, bootstrap_sample.confidence_interval.low, bootstrap_sample.confidence_interval.high
