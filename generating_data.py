from typing import Union, Tuple

from scipy import stats
import numpy as np


def generate_yx_1d_with_atc(n: int, k: float, z: stats.rdist = stats.truncnorm(0, np.inf)):
    """ Generate realisation and nowcast data using the scheme
        Delta y_t ~ N,
        x_t = x_{t-1} + Z * ()

    :param n: number of observation-pairs
    :param k: probability, that x has the right ATC
    :param z: distribution of deviation from y_t with support in [0, inf)
    :return: two numpy arrays with realizations and nowcasts
    """
    delta_y = stats.norm.rvs(size=n)
    y = np.cumsum(delta_y)
    u = stats.uniform.rvs(size=n-1)
    # b = 2 * stats.bernoulli.rvs(p=0.5, size=n-1) - 1
    z_rvs = z.rvs(size=n-1)
    # x = np.concatenate([[0], y[:-1]]) + (2 * (u < k) - 1) * z * np.sign(delta_y)
    delta_x = (2 * (u < k) - 1) * np.sign(delta_y[1:]) * z_rvs
    x = np.zeros(n)
    x[0] = y[0]
    x[1:] = y[:-1] + delta_x
    return y, x


def generate_yx_1d_with_atc_distorted(n: int, k: float, z: stats.rdist = stats.truncnorm(0, np.inf),
                                      e: Union[None, Tuple] = None):
    """ Generate realisation and nowcast data using the scheme
        Delta y_t ~ N,
        x_t = x_{t-1} + Z * ()

    :param n: number of observation-pairs
    :param k: probability, that x has the right ATC
    :param z: distribution of deviation from y_t with support in [0, inf)
    :param e: pair of distributions to distort the values of y (e[0]) and x (e[1]) by an error (default is None with standard normal distribution)
    :return: two numpy arrays with (distorted) realizations and (distorted) nowcasts
    """
    if e is None:
        e = (stats.norm(), stats.norm())
    y, x = generate_yx_1d_with_atc(n=n, k=k, z=z)  # Generate data without distortion
    e_y = e[0].rvs(n)  # Distort data with the specified distribution
    e_x = e[1].rvs(n)
    return y + e_y, x + e_x



