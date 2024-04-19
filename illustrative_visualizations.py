import argparse
import pickle
from datetime import datetime, timedelta
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns

from python_utils.mpl_helpers import fig_with_size, save_fig, update_mpl_rcparams
from aatc import plot_4q, atc_ratio, plot_cond_prob, ExclusionArea, atc_with_bootstrap, compute_cond_prob

exp_path = Path('plots/illustrative_examples')


def annotate_pts(pts, ax):
    """Annotate the points pts in the 4Q plot."""
    settings = {'va': 'center', 'xycoords': 'data', 'fontsize': 4, 'c': 'C0'}
    for i_pt in range(len(pts)):
        if i_pt == 2:
            ax.annotate(f'{i_pt + 1}', xy=(pts[i_pt, 0] - 0.2, pts[i_pt, 1]), ha='right', **settings)
        else:
            ax.annotate(f'{i_pt + 1}', xy=(pts[i_pt, 0] + 0.2, pts[i_pt, 1]), ha='left', **settings)


def sample_4q():
    """Plots sample four-quadrant plots."""
    T = 365 * 4 + 1
    rng = np.random.default_rng(42)
    for (dgp_name, dgp_method) in zip(['dgp1', 'dgp1_time', 'dgp1_asym', 'dgp2'],
                                      [dgp1, dgp1_timevarying, dgp1_asym, dgp2]):
        res = dgp_method(T, rng)
        diffy, diffx = res[0], res[1]
        fig, ax = fig_with_size(4)
        plot_4q(diffy=diffy, diffx=diffx, ax=ax)
        save_fig(fig, exp_path / f'appendix_4q_{dgp_name}.pdf')


def plot_4q_with_without_excl():
    """Plot sample four-quadrant plots with and without exclusion areas."""
    # Sample points
    fig, ax = fig_with_size(4, factor_height=1)
    pts = np.array([[0.1, -0.1], [4, 8], [6, 2], [-6, 6], [-4, -8], [0.5, 4], [4, -0.5]])
    plot_4q(pts[:, 0], pts[:, 1], ax=ax)
    annotate_pts(pts, ax)
    save_fig(fig, exp_path / '4q_without_excl.pdf')
    fig, ax = fig_with_size(4, factor_height=1)
    excl_area_box = ExclusionArea(1, 1, 'or')
    plot_4q(pts[:, 0], pts[:, 1], ax=ax, excl_area=excl_area_box)
    annotate_pts(pts, ax)
    save_fig(fig, exp_path / '4q_excl_box.pdf')
    fig, ax = fig_with_size(4, factor_height=1)
    excl_area_x = ExclusionArea(eps_x=1)
    plot_4q(pts[:, 0], pts[:, 1], ax=ax, excl_area=excl_area_x)
    annotate_pts(pts, ax)
    save_fig(fig, exp_path / '4q_excl_axis.pdf')
    fig, ax = fig_with_size(4, factor_height=1)
    excl_area_cross = ExclusionArea(1, 1, 'and')
    plot_4q(pts[:, 0], pts[:, 1], ax=ax, excl_area=excl_area_cross)
    annotate_pts(pts, ax)
    save_fig(fig, exp_path / '4q_excl_cross.pdf')


def sample_4q_with_time():
    """Plots sample four-quadrant plots with time-varying ATC ratio."""
    T = 4 * 365 + 1
    rng = np.random.default_rng(42)
    diffy, diffx, t_array, k_array = dgp1_timevarying(T=T, rng=rng)

    # Normal 4Q plot
    fig, ax = fig_with_size(4, factor_height=1)
    plot_4q(diffy=diffy, diffx=diffx, ax=ax)
    save_fig(fig, exp_path / '4q_sample_without_time.pdf')
    # 4Q plot with time
    fig, ax = fig_with_size(4, factor_height=1)
    plot_4q(diffy=diffy, diffx=diffx, include_time=True, ax=ax)
    save_fig(fig, exp_path / '4q_sample_with_time.pdf')

    # Accuracy rolling estimate
    df = pd.DataFrame({'diffy': diffy, 'diffx': diffx})
    window_length = int(365 / 4)
    excl_area_x = ExclusionArea(eps_x=1)
    for i in range(window_length, len(df)):
        df.loc[i, 'acc'] = atc_ratio(df.loc[(i - window_length):i, 'diffy'],
                                          df.loc[(i - window_length):i, 'diffx'])
        df.loc[i, 'acc-eps'] = atc_ratio(df.loc[(i - window_length):i, 'diffy'],
                                              df.loc[(i - window_length):i, 'diffx'], excl_area=excl_area_x)
    fig, ax = fig_with_size(2, factor_height=1)
    ax.plot(t_array, df['acc'], label=r'$\mu_{t, 91}$', alpha=0.8, zorder=2)
    # ax.plot(t_array, df['acc-eps'], label=r'$\mu_{' + f'{1:.1f}' + r'}$', alpha=0.8, zorder=2)
    ax.plot(t_array, k_array, label='$k_t$', alpha=0.8, zorder=1.5)
    # Create legend without visual border
    ax.legend(frameon=False)
    ax.set(xlabel='t')
    print(f"Overall accuracies: {atc_ratio(df['diffy'], df['diffx']):.4f}; "
          f"{atc_ratio(df['diffy'], df['diffx'], excl_area=excl_area_x):.4f}")
    save_fig(fig, exp_path / 'atc_ratio_time_series.pdf')


def cond_probs():
    """Plot conditional ATC plots for different scenarios."""
    # Create data
    T = 4 * 365 + 1
    rng = np.random.default_rng(42)
    k_const = 0.8

    diffx = stats.norm.rvs(size=T, random_state=rng) + stats.uniform(-10, 20).rvs(size=T, random_state=rng)

    # Time varying nowcast
    t_array = np.arange(T)
    k_array_time = 0.75 + np.sin(t_array * 4 / T * 2 * np.pi) / 4
    k_array_const = np.ones_like(k_array_time) * k_const

    def k_asymmetric_func(x): return 0.5 + np.minimum(np.maximum((x + 5) / 10, 0), 1) / 2

    k_asymmetric = k_asymmetric_func(diffx)

    diffy_dict = {}

    fig, ax = fig_with_size(2)

    for (nb, name, k_array) in zip([1, 2, 3], ['const.', 'time var.', 'asym.'],
                                   [k_array_const, k_array_time, k_asymmetric]):
        diffy_dict[name] = diffx * stats.truncnorm(loc=1, scale=0.5, a=-2, b=np.inf).rvs(size=T, random_state=rng) * \
                           (2 * stats.bernoulli.rvs(k_array, size=T, random_state=rng) - 1)
        plot_cond_prob(diffy_dict[name], diffx, label=f'{nb}', ax=ax, x_bounds=(-10, 10))
        excl_area_x = ExclusionArea(eps_x=2)
        plot_cond_prob(diffy_dict[name], diffx, label=f'{nb}' + r', $\varepsilon \leq 2$', excl_area=excl_area_x, ax=ax,
                       x_bounds=(-10, 10))
    ax.axhline(y=k_const, color='C0', linestyle='dashed', linewidth=0.2, label='(1)')
    x_linspace = np.arange(-10, 10, 0.5)
    ax.plot(x_linspace, k_asymmetric_func(x_linspace), color='C2', linestyle='dashed', linewidth=0.2, label='(3)')
    ax.legend(ncols=2)
    save_fig(fig, exp_path / 'cond_prob_plot.pdf')


def compute_bw(data_name):
    """Illustrations of different bandwidth methods in KDE for the conditional ATC plots."""
    update_mpl_rcparams()
    data_path = Path('data', 'illustrative_examples')

    T = 4 * 365 + 1
    rng = np.random.default_rng(42)
    bw_list = ['normal_reference', 'cv_ml', 'cv_ls']
    bw_name_list = ['thumb', 'cv_ml', 'cv_ls']

    excl_area_x = ExclusionArea(eps_x=2)
    fig, ax = fig_with_size(2)
    if data_name == 'butterfly':
        diffy, diffx, k_array, k_asymmetric_func = dgp1_asym(T, rng=rng)
        x_linspace = np.arange(-10, 10, 0.5)
        ax.plot(x_linspace, k_asymmetric_func(x_linspace), color='C2', linestyle='dashed', linewidth=0.4)
    elif data_name == 'normal':
        diffy, diffx, k_cond_normal = dgp2(T, rng)
        x_linspace = np.arange(-10, 10, 0.5)
        ax.plot(x_linspace, k_cond_normal(x_linspace), color='grey', linestyle='dashed', linewidth=0.4)
    else:
        raise ValueError('Unknown data name')
    for (bw_name, bw) in zip(bw_name_list, bw_list):
        print(f'Starting {data_name} {bw} at {datetime.now():%H:%M:%S}')
        if (data_path / f'cond_prob_{data_name}_{bw}.npz').exists():
            results = np.load(data_path / f'cond_prob_{data_name}_{bw}.npz', allow_pickle=True)
        else:
            results = dict()
            for (name, excl_area) in zip([f'{bw_name}', f'{bw_name}' + r', $\varepsilon \leq 2$'],
                                         [None, excl_area_x]):
                x_linspace, cond_prob = compute_cond_prob(diffy, diffx,
                                                          excl_area=excl_area, x_bounds=(-10, 10), bw_kde=bw)
                results[name] = (x_linspace, cond_prob)
            np.savez(data_path / f'cond_prob_{data_name}_{bw}.npz', **results)

        for key in results.keys():
            x_linspace, cond_prob = results[key]
            if r'$\varepsilon \leq 2$' in key:
                plot_cond_prob(x_linspace=x_linspace, cond_prob_estimate=cond_prob,
                           label=key.replace(r'\leq', r'='), ax=ax, excl_area=excl_area_x)
            else:
                plot_cond_prob(x_linspace=x_linspace, cond_prob_estimate=cond_prob,
                               label=key.replace(r'\leq', r'='), ax=ax)
            # ax.plot(x_linspace, cond_prob, label=key, alpha=0.5)
    fig.legend(loc='outside upper center', ncols=3, fontsize='x-small')
    fig.savefig(exp_path / f'cond_prob_plot_bw_{data_name}.pdf')


def cond_probs_bw():
    """Start conditional ATC plots with different bandwidths in parallel."""
    pool = Pool(2)
    pool.map(compute_bw, ['butterfly', 'normal'])


def dgp1(T: int, rng: np.random.Generator):
    """The first data generation process."""
    t_array = np.arange(T)
    k = 0.75
    diffx = stats.norm.rvs(size=T, random_state=rng) + stats.uniform(-5, 10).rvs(size=T, random_state=rng)
    diffy = diffx * stats.truncnorm(loc=1, scale=0.5, a=-2, b=np.inf).rvs(size=T, random_state=rng) * \
            (2 * stats.bernoulli.rvs(k, size=T, random_state=rng) - 1)
    return diffy, diffx


def dgp1_timevarying(T: int, rng: np.random.Generator):
    """The first data generation process with time-varying ATC ratio."""
    t_array = np.arange(T)
    k_array = 0.75 + np.sin(t_array / 365.25 * 2 * np.pi) / 4

    # Analyze k array
    # fig, ax = fig_with_size(2)
    # ax.plot(t_array, k_array)
    # save_fig(fig, exp_path / 'k_array.pdf')
    # Test Bernoulli random variable
    # fig, ax = fig_with_size(2)
    # import pandas as pd
    # ax.plot(t_array, pd.Series(stats.bernoulli.rvs(k_array, size=T)).rolling(100, center=True).mean())
    # save_fig(fig, exp_path / 'test_bernoulli.pdf')

    diffx = stats.norm.rvs(size=T, random_state=rng) + stats.uniform(-5, 10).rvs(size=T, random_state=rng)
    diffy = diffx * stats.truncnorm(loc=1, scale=0.5, a=-2, b=np.inf).rvs(size=T, random_state=rng) * \
            (2 * stats.bernoulli.rvs(k_array, size=T, random_state=rng) - 1)
    return diffy, diffx, t_array, k_array


def dgp1_asym(T, rng: np.random.Generator):
    """The first data generation process with asymmetric ATC ratio."""
    diffx = stats.norm.rvs(size=T, random_state=rng) + stats.uniform(-10, 20).rvs(size=T, random_state=rng)

    def k_asymmetric_func(x): return 0.5 + np.minimum(np.maximum((x + 5) / 10, 0), 1) / 2

    k_array = k_asymmetric_func(diffx)
    diffy = diffx * stats.truncnorm(loc=1, scale=0.5, a=-2, b=np.inf).rvs(size=T, random_state=rng) * \
            (2 * stats.bernoulli.rvs(k_array, size=T, random_state=rng) - 1)
    return diffy, diffx, k_array, k_asymmetric_func


def dgp2(T: int, rng: np.random.Generator) -> tuple[np.array, np.array, Callable]:
    """
    This function generates a multivariate normal distribution and a conditional normal distribution.

    :param T: The size of the distribution to be generated.
    :param rng: An instance of numpy's random number generator.

    :return: The first column of the multivariate normal distribution, the second column of the multivariate normal distribution, and a function representing the conditional normal distribution.
    """

    vars, rho = [4, 4], 3 / 4  # variances and correlation
    cov_rho = rho * (vars[0] * vars[1]) ** 0.5
    mv_normal = stats.multivariate_normal(mean=np.zeros(2), cov=[[vars[0], cov_rho], [cov_rho, vars[1]]]).rvs(
        size=T, random_state=rng)

    def k_cond_normal(x):
        return stats.norm.cdf(
            0,
            loc=- np.abs((vars[0] / vars[1]) ** 0.5 * rho * x),
            scale=((1 - rho ** 2) * vars[0]) ** 0.5)

    # normal conditional distribution, bivariate
    # (https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case_2)

    diffx, diffy = mv_normal[:, 0], mv_normal[:, 1]
    return diffy, diffx, k_cond_normal


def run_bootstrap_simulation():
    """Run bootstrap simulations sequentially."""
    # Do sequentially to avoid problems with random state
    # Setup
    confidence_level = 0.9
    n_per_comb = 10000
    B = 10000
    rng = np.random.default_rng(42)
    T_list = [30, 52, 168, 365, 720, 1024]
    for T in T_list:
        print(f'Starting T={T} at {datetime.now():%H:%M:%S}')
        for method in ['percentile', 'basic', 'bca']:
            print(f'- Starting method={method} at {datetime.now():%H:%M:%S}')
            for (name_dgp, dgp) in zip(['butterfly', 'normal'], [dgp1_asym, dgp2]):
                print(f'-- Starting {name_dgp} at {datetime.now():%H:%M:%S}')
                ci_low = np.zeros(n_per_comb)
                ci_high = np.zeros(n_per_comb)
                time_per_loop = np.zeros(n_per_comb)
                for i_comb in range(n_per_comb):
                    dgp_res = dgp(T=T, rng=rng)
                    diffy, diffx = dgp_res[0], dgp_res[1]
                    data = (diffy, diffx)
                    start = datetime.now()
                    _, ci_low[i_comb], ci_high[i_comb] = atc_with_bootstrap(
                        diffy=diffy, diffx=diffx, confidence_level=confidence_level,
                        rng=rng, bootstrap_method=method, n_bootstrap=B)
                    end = datetime.now()
                    time_per_loop[i_comb] = (end - start) / timedelta(milliseconds=1)
                np.savez(f'data/illustrative_examples/T_{T}_method_{method}_dgp_{name_dgp}',
                         ci_low=ci_low, ci_high=ci_high, time_per_loop=time_per_loop, method=method, T=T,
                         name_dgp=name_dgp, confidence_level=confidence_level)


def evaluate_bootstrap_simulation():
    """Create evaluation plots for bootstrap simulations."""
    # Compute theoretical quantities
    T = int(1e8)
    rng = np.random.default_rng(42)
    theo_dict = dict()
    path_to_theo_dict = Path('data/illustrative_examples/theoretic.pickle')
    if not path_to_theo_dict.is_file():
        for (name_dgp, dgp) in zip(['butterfly', 'normal'], [dgp1_asym, dgp2]):
            dgp_res = dgp(T=T, rng=rng)
            diffy, diffx = dgp_res[0], dgp_res[1]
            theo_dict[name_dgp] = atc_ratio(diffy, diffx)
            print(f'{name_dgp} - {theo_dict[name_dgp]:.4f}')
            with open(path_to_theo_dict, 'wb') as f:
                pickle.dump(theo_dict, f)
    else:
        with open(path_to_theo_dict, 'rb') as f:
            theo_dict = pickle.load(f)

    res_dict = dict()
    for f in glob('data/illustrative_examples/T_*.npz'):
        fpath = Path(f)
        res_dict[fpath.stem] = np.load(f)

    # Build pandas dataframe per method with computation times
    df_butterfly = pd.DataFrame(columns=['T', 'method', 'comp_time'])
    df_normal = pd.DataFrame(columns=['T', 'method', 'comp_time'])
    for f in res_dict.keys():
        np_dict = res_dict[f]
        df_new = pd.DataFrame({'T': np_dict['T'], 'method': np_dict['method'], 'comp_time': np_dict['time_per_loop']})
        if 'butterfly' in f:
            df_butterfly = pd.concat([df_butterfly, df_new])
        else:
            df_normal = pd.concat([df_normal, df_new])
    for (name, df) in zip(['butterfly', 'normal'], [df_butterfly, df_normal]):
        fig, ax = fig_with_size(2)
        sns.boxplot(x='T', y='comp_time', hue='method', data=df, ax=ax, fill=False,
                    hue_order=['percentile', 'basic', 'bca'], showfliers=False)
        ax.set(ylabel='Computation time [ms]')
        save_fig(fig, exp_path / f'boxplot_comp_time_{name}.pdf')

    T = pd.Series([30, 52, 168, 365, 720, 1024])
    df_dict = dict()
    df_dict['butterfly'] = pd.DataFrame(columns=['percentile', 'basic', 'bca'], index=T)
    df_dict['normal'] = pd.DataFrame(columns=['percentile', 'basic', 'bca'], index=T)
    # Build pandas dataframe per method with share of cis covering the true value
    for f in res_dict.keys():
        np_dict = res_dict[f]
        name_dgp = str(np_dict['name_dgp'])
        # Compute share of cis
        share_ci = ((np_dict['ci_low'] <= theo_dict[name_dgp]) & (theo_dict[name_dgp] <= np_dict['ci_high'])).mean()
        # Compute average width
        av_width = np.nanmean(np_dict['ci_high'] - np_dict['ci_low'])
        # Save to dataframe
        df_dict[str(np_dict['name_dgp'])].loc[
            int(np_dict['T']), str(np_dict['method'])] = f'{share_ci:.2f} ({av_width:.3f})'
    for name in ['butterfly', 'normal']:
        df_dict[name].to_latex(exp_path / f'bootstrap_{name}_cis.tex', index=True, index_names=True, header=['percentile', 'basic', 'BCa'])


def main():
    """
    The main function of the script.
    The arguments control the functionalities of the script, such as generating 4Q plots, running bootstrap simulations, etc.
    The function then parses the arguments and updates matplotlib parameters.
    Depending on the arguments provided, the function calls other functions to perform the requested tasks.

    :return: None
    """
    # Create the directory for storing plots if it doesn't exist
    exp_path.mkdir(parents=True, exist_ok=True)

    # Initialize an argument parser
    arg_parser = argparse.ArgumentParser()

    # Add arguments to the parser for different functionalities
    # --4qplots: If provided, the program will generate 4Q plots
    arg_parser.add_argument('--4qplots', dest='do_4q_plots', action='store_true')
    arg_parser.add_argument('--4qsample', dest='do_4q_sample', action='store_true')
    arg_parser.add_argument('--cond_probs', dest='do_cond_probs', action='store_true')
    arg_parser.add_argument('--cond_probs_bw', dest='do_cond_probs_bw', action='store_true')
    arg_parser.add_argument('--run_bootstrap', dest='do_bootstrap', action='store_true')
    arg_parser.add_argument('--eval_bootstrap', dest='do_eval_bootstrap', action='store_true')
    arg_parser.add_argument('--all', dest='do_all', action='store_true')

    # Parse the arguments
    args = arg_parser.parse_args()

    # Update matplotlib parameters
    update_mpl_rcparams()

    if args.do_all:
        args.do_4q_plots = True
        args.do_4q_sample = True
        args.do_cond_probs = True
        args.do_cond_probs_bw = True
        args.do_eval_bootstrap = True
    if args.do_4q_plots:
        plot_4q_with_without_excl()
    if args.do_4q_sample:
        sample_4q()
        sample_4q_with_time()
    if args.do_cond_probs:
        cond_probs()
    if args.do_cond_probs_bw:
        cond_probs_bw()
    if args.do_bootstrap:
        run_bootstrap_simulation()
    if args.do_eval_bootstrap:
        evaluate_bootstrap_simulation()


if __name__ == '__main__':
    main()
