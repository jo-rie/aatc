from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from sklearn.calibration import CalibrationDisplay

from python_utils.mpl_helpers import (fig_with_size, default_fig_width,
                                      save_fig, update_mpl_rcparams, default_fig_height)
from utils_covid import (create_df_for_model_and_lags,
                         create_df_truth_lags, read_nowcast_data,
                         read_truth_data, rename_models, read_probabilistic_nowcast_data)
from setup import get_atc_ratio_name
from aatc import plot_4q, plot_cond_prob, ExclusionArea, atc_ratio, atc_with_bootstrap

plot_export_path = Path('plots/covid_nowcast')

start_eval_default = pd.to_datetime('2021-11-22')
end_eval_default = pd.to_datetime('2022-04-29')

end_wave_4 = pd.to_datetime('2021-12-26')


def read_truth_nowcasts(outlier_bound: Union[float, None] = 4 * 1e5):
    """
    The read_truth_nowcasts function reads in the truth data and nowcasts of lag 0.
    
    :param outlier_bound: float: Remove outliers from the nowcast data (None for no bound)
    :return: A tuple of two dataframes
    :doc-author: Trelent
    """
    df_truth = read_truth_data().sort_index(ascending=True)
    df_nowcasts = read_nowcast_data(0).rename(columns=rename_models)
    if outlier_bound is not None:
        df_nowcasts[df_nowcasts > outlier_bound] = None
    # Restrict df_truth and df_nowcasts to the time range from start_eval_default to end_eval_default
    df_truth = df_truth.loc[start_eval_default:end_eval_default, :]
    df_nowcasts = df_nowcasts.loc[start_eval_default:end_eval_default, :]
    return df_truth, df_nowcasts


def create_model_lag_dict(model_subset: list[str], lag_list: list[int], start_eval: datetime = None,
                          end_eval: datetime = None,
                          outlier_bound: Union[float, None] = 4 * 1e5) -> dict[str, pd.DataFrame]:
    """
    The create_df_for_models_and_lags function takes a list of models and a list of lags as input, and returns a
    dictionary. The dictionary contains a dataframe for each model in the model_subset, with a column for each
    lag with the difference values.

    :param outlier_bound: float: Remove outliers from the nowcast data (None for no bound)
    :param model_subset: list[str]: Specify which models we want to use
    :param lag_list: list[int]: Specify the number of days before the target date that we want to use
    :param start_eval: datetime: Specify the start date for the evaluation
    :param end_eval: datetime: Specify the end date for the evaluation
    :return: dictionary with the model's nowcast for the lags
    """
    model_short_list = [rename_models[m] for m in model_subset]
    model_df_dict = dict()

    for (name, model) in zip(model_short_list, model_subset):
        df_model = create_df_for_model_and_lags(model,
                                                lag_list)  # naming, e.g., 'ILM-prop_target_1d_before',
        # 'ILM-prop_diff_lag1'
        # df_model = df_model.loc[start_eval:end_eval, :] # Some dates are missing
        df_model = df_model.reindex(pd.date_range(start_eval, end_eval, freq='D'), fill_value=None)
        if outlier_bound is not None:
            df_model[df_model > outlier_bound] = None
        model_df_dict[name] = df_model
    return model_df_dict


def create_basic_plots():
    """
    The create_basic_plots function creates two plots:
        1. The first plot shows the true data (without outliers) and the nowcasts for all four quarters.
        2. The second plot shows only the nowcasts for all four quarters, but without outliers.
    
    :return: Nothing
    """
    df_truth, df_nowcasts = read_truth_nowcasts(outlier_bound=None)  # without removal of outliers

    def add_wave_4_to_plot(ax_loop: plt.Axes):
        # Transform end_wave_4 to matplotlib date
        end_wave_4_mpl = mdates.date2num(end_wave_4)
        # Add a vertical line on end_wave_4 with text annotation 'end wave 4' on the left and 'start wave 5' on the
        # right
        ax_loop.axvline(end_wave_4_mpl, color='grey', linestyle='--', linewidth=0.2)
        ax_loop.text(end_wave_4_mpl - 2, 0.2, 'wave 4', verticalalignment='center', horizontalalignment='right',
                     transform=ax_loop.get_xaxis_transform())
        ax_loop.text(end_wave_4_mpl + 2, 0.2, 'wave 5', verticalalignment='center',
                     horizontalalignment='left',
                     transform=ax_loop.get_xaxis_transform())

    # Plot Truth data
    fig, ax = fig_with_size(2)
    df_truth.plot(ax=ax)
    ax.get_legend().remove()
    ax.set(ylabel='7-day hospitalization index', xlabel='Date', ylim=(0, 15000))
    add_wave_4_to_plot(ax)
    save_fig(fig, plot_export_path / '00_true_data.pdf')

    # Plot nowcasts
    fig, ax = fig_with_size(2)
    df_nowcasts.plot(ax=ax)
    ax.get_legend().remove()
    fig.legend(loc='outside upper center', ncol=4)
    ax.set(ylabel='7-day hospitalization index', xlabel='Target date', ylim=(0, 15000))
    add_wave_4_to_plot(ax)
    fig.savefig(plot_export_path / '00_nowcast_data.pdf')
    plt.close(fig)

    df_truth, df_nowcasts = read_truth_nowcasts()  # read data with removal of outliers
    fig, ax = fig_with_size(2)
    df_nowcasts.plot(ax=ax)
    ax.get_legend().remove()
    fig.legend(loc='outside upper center', ncol=4)
    ax.set(ylabel='7-day hospitalization index', xlabel='Target date', ylim=(0, 15000))
    fig.savefig(plot_export_path / '01_nowcast_data_outliers_removed.pdf')
    plt.close(fig)


def compute_overall_scores():
    """
    The compute_overall_scores function reads in the truth and nowcast data,
    computes the RMSE, MAEs, MSEs and number of values for each model. 
    It then returns a DataFrame with these scores.
    
    
    :return: A dataframe with the following columns:
    """
    df_truth, df_nowcasts = read_truth_nowcasts()  # read data with removal of outliers

    count_df = df_nowcasts.count(axis=0).to_frame('count')

    df_rmse = pd.DataFrame(index=df_nowcasts.columns, columns=['RMSE', 'MAE', 'mse'])
    for col in df_nowcasts.columns:
        df_loop = df_nowcasts.loc[:, [col]].join(df_truth)
        df_loop['diff'] = (df_loop[col] - df_loop['true'])
        df_rmse.loc[col, 'mse'] = (df_loop['diff'] ** 2).mean()
        df_rmse.loc[col, 'RMSE'] = df_rmse.loc[col, 'mse'] ** 0.5
        df_rmse.loc[col, 'MAE'] = df_loop['diff'].abs().mean()
    df_rmse.sort_values('MAE', ascending=True)
    df_rmse = df_rmse.join(count_df)  # Add count data

    df_rmse.reset_index().sort_values('RMSE', ascending=True).to_latex(
        buf=plot_export_path / '01_model_scores.tex',
        columns=['model', 'RMSE', 'MAE', 'count'], header=['Model', 'RMSE', 'MAE', 'Count'], column_format='l r r r',
        float_format="{:,.0f}".format, escape=True, index=False)
    return df_rmse


def plot_additional_material_for_subset(subset_models: list[str], lags: list[int],
                                        start_eval: datetime = start_eval_default,
                                        end_eval: datetime = end_eval_default):
    """
    The plot_additional_material_for_subset function creates additional plots (4Q plot, 
    conditional probability, accuracy-over-epsilon) for a subset of models.
    
    :param subset_models: list[str]: Specify which models should be used for the plot
    :param lags: list[int]: Specify the lags for the plots
    :param start_eval: datetime: Specify the start date for the evaluation
    :param end_eval: datetime: Specify the end date for the evaluation
    :return: The plots of the accuracy and conditional probability for a subset of models
    """
    model_short = [rename_models[m] for m in subset_models]

    # Use only subset of models
    df_truth = create_df_truth_lags(lags)  # naming, e.g., 'target_minus_1', 'diff_lag_1'
    df_truth = df_truth.loc[start_eval:end_eval, :]

    fig, axes = plt.subplots(len(lags), len(model_short), figsize=(default_fig_width, default_fig_width), sharex='row',
                             sharey='row', layout='constrained')
    model_df_dict = create_model_lag_dict(subset_models, lags, start_eval, end_eval)
    for (i_name, name, model) in zip(range(len(model_short)), model_short, subset_models):
        # Create plots (4Q)
        for (i_lag, lag) in enumerate(lags):
            ax = axes[i_lag, i_name]
            # fig, ax = fig_with_size(4)
            plot_4q(df_truth[f'diff_lag_{lag}'], model_df_dict[name][model + f'_diff_lag{lag}'], ax=ax)
            ax.set(title=f'{name} - horizon {lag}d')
            # save_fig(fig, plot_export_path / f'30_{name}_4q_lag_{lag}.pdf')
    save_fig(fig, plot_export_path / f'30_4q_plots.pdf')

    # Create Cond Prob Plots and Accuracy over eps
    eps_linspace = np.linspace(0, 1000, 500)
    for lag in lags:
        fig_acc, ax_acc = fig_with_size(2)
        fig_cond_prob, ax_cond_prob = fig_with_size(2)
        for (name, model) in zip(model_short, subset_models):
            df_model = model_df_dict[name]
            acc = np.zeros_like(eps_linspace)
            for (i_eps, eps) in enumerate(eps_linspace):
                excl_area = ExclusionArea(eps_x=eps, eps_y=0)
                acc[i_eps] = atc_ratio(df_truth[f'diff_lag_{lag}'], df_model[model + f'_diff_lag{lag}'],
                                       excl_area=excl_area)
            ax_acc.plot(eps_linspace, acc, label=f'{name}')
            # Create Exclusion area and plot conditional ATC plot
            excl_area = ExclusionArea.from_quantile(df_truth[f'diff_lag_{lag}'], df_model[model + f'_diff_lag{lag}'],
                                                    q=0.1, combination='or')
            plot_cond_prob(df_truth[f'diff_lag_{lag}'], df_model[model + f'_diff_lag{lag}'], ax=ax_cond_prob,
                           label=name, excl_area=excl_area)
            ax_acc.legend()
            ax_cond_prob.legend()
        save_fig(fig_acc, plot_export_path / f'40_acc_eps_lag_{lag}.pdf')
        save_fig(fig_cond_prob, plot_export_path / f'40_cond_prob_lag_{lag}.pdf')


def compute_ratios(eps_q: list[float], lags: list[int], start_eval: datetime = start_eval_default,
                   end_eval: datetime = end_eval_default) -> True:
    """ Compute a dataframe with the ATC ratio for all the models and the different q.

    :param eps_q: list[float]: List of quantile values to exclude, None for no exclusion.
    :param lags: list[int]: Specify the lags for the plots
    :param start_eval: datetime: Specify the start date for the evaluation
    :param end_eval: datetime: Specify the end date for the evaluation
    :return: A dataframe with the ATC ratio for all the models and the different q
    """
    df_truth = create_df_truth_lags(lags)  # naming, e.g., 'target_minus_1', 'diff_lag_1'
    df_truth = df_truth.loc[start_eval:end_eval, :]
    rng = np.random.default_rng(42)  # Random Number generator for bootstrapping

    def format_bootstrap(estimator, low, high):
        return r'{' + f'{estimator:.2f}' + r'\newline' + f'({low:.2f}, {high:.2f})' + r'}'

    for lag in lags:
        column_list = [item for sublist in [
            [get_atc_ratio_name(lag=lag, quantile=q),
             get_atc_ratio_name(lag=lag, quantile=q, pos_neg='positive'),
             get_atc_ratio_name(lag=lag, quantile=q, pos_neg='negative')] for q in eps_q]
                       for item in sublist]
        df_result = pd.DataFrame(index=[rename_models[m] for m in list(rename_models.keys())],
                                 columns=column_list)
        # Compute ATC ratio for all models and the different q (accordingly to the dict)
        # and store the results in a dataframe
        # Compute positive and negative ATC ratio additionally
        # Create a separate table for all lags
        # Load the data
        for (long_name, short_name) in rename_models.items():
            df_nowcasts = create_df_for_model_and_lags(long_name, [lag], start_eval, end_eval)
            for q in eps_q:
                diffy = df_truth[f'diff_lag_{lag}']
                diffx = df_nowcasts[long_name + f'_diff_lag{lag}']
                # Create Exclusion area
                excl_area = ExclusionArea.from_quantile(diffy=diffy, diffx=diffx,
                                                        q=q, combination='or')
                # Compute the ATC ratios; Use Bootstrapping to compute confidence intervals
                df_result.loc[short_name, get_atc_ratio_name(lag=lag, quantile=q)] = format_bootstrap( \
                    *atc_with_bootstrap(diffy=diffy, diffx=diffx, excl_area=excl_area, rng=rng))
                df_result.loc[short_name, get_atc_ratio_name(lag=lag, quantile=q, pos_neg='positive')] = \
                    format_bootstrap(
                        *atc_with_bootstrap(diffy=diffy, diffx=diffx, excl_area=excl_area, rng=rng, pos_neg='positive'))
                df_result.loc[short_name, get_atc_ratio_name(lag=lag, quantile=q, pos_neg='negative')] = \
                    format_bootstrap(*atc_with_bootstrap(diffy=diffy, diffx=diffx, excl_area=excl_area, rng=rng,
                                                         pos_neg='negative'))
        # Save df_result to latex table
        df_result.to_latex(buf=plot_export_path / f'30_atc_ratios_lag_{lag}.tex', float_format="{:,.2f}".format,
                           column_format='l' + r' p{0.11\textwidth}' * 6)
    return True


def create_marginal_analysis_table(lags: list[int], model_subset: list[str] = None,
                                   start_eval: datetime = start_eval_default, end_eval: datetime = end_eval_default):
    """
    The create_marginal_analysis_table function creates a table with the mean, standard deviation, and 10 % quantile of
    the absolute difference for each model and lag.

    :param model_subset:
    :param lags: list[int]: Specify the lags for the plots
    :param start_eval: datetime: Specify the start date for the evaluation
    :param end_eval: datetime: Specify the end date for the evaluation
    :return: A dataframe with the mean and standard deviation of the absolute difference for each model and lag
    """
    if model_subset is None:
        model_subset = list(rename_models.keys())
    df_truth = create_df_truth_lags(lags)  # naming, e.g., 'target_minus_1', 'diff_lag_1'
    df_truth = df_truth.loc[start_eval:end_eval, :]

    def std_col_name(lag_local: int) -> str:
        return r'$\sigma_{x^{\Delta, ' + f'{lag_local}' + r'}}$'

    def quantile_col_name(lag_local: int) -> str:
        return r'$q_{0.1} (x^{\Delta, ' + f'{lag_local}' + r'})$'

    def sum_geq_0(lag_local: int) -> str:
        return f'(1), l={lag_local}'

    # Create list of column names
    column_list = [item for sublist in [
        [sum_geq_0(lag), std_col_name(lag), quantile_col_name(lag)] for lag in lags] for item in sublist]

    df_result = pd.DataFrame(index=[rename_models[m] for m in model_subset],
                             columns=column_list)
    df_nowcast_dict = create_model_lag_dict(list(rename_models.keys()), lags, start_eval, end_eval)
    for lag in lags:
        df_result.loc['True', sum_geq_0(lag)] = (df_truth[f'diff_lag_{lag}'] > 0).sum()
        df_result.loc['True', std_col_name(lag)] = int(df_truth[f'diff_lag_{lag}'].std())
        df_result.loc['True', quantile_col_name(lag)] = int(np.quantile(np.abs(df_truth[f'diff_lag_{lag}']), 0.1))
        for (long_name, short_name) in rename_models.items():
            df_nowcasts = df_nowcast_dict[short_name]
            df_result.loc[short_name, sum_geq_0(lag)] = (df_nowcasts[long_name + f'_diff_lag{lag}'] > 0).sum()
            df_result.loc[short_name, std_col_name(lag)] = int(df_nowcasts[long_name + f'_diff_lag{lag}'].std())
            df_result.loc[short_name, quantile_col_name(lag)] = int(np.nanquantile(
                np.abs(df_nowcasts[long_name + f'_diff_lag{lag}']), 0.1))
    df_result.to_latex(buf=plot_export_path / '10_marginal_analysis.tex', float_format='%.2f')
    return df_result


def plot_kdes(lags: list[int], model_subset: list[str] = None,
              start_eval: datetime = start_eval_default, end_eval: datetime = end_eval_default):
    """ Plot a separate plot for each lag in lags with the KDEs of the nowcasts and the truth data for the specified
    lag.

    :param model_subset: list[str]: Specify which models should be used for the plot
    :param lags: list[int]: List of lags
    :param start_eval: datetime: Specify the start date for the evaluation
    :param end_eval: datetime: Specify the end date for the evaluation
    :return: None.
    """
    if model_subset is None:
        model_subset = list(rename_models.keys())
    df_truth = create_df_truth_lags(lags)  # naming, e.g., 'target_minus_1', 'diff_lag_1'
    df_truth = df_truth.loc[start_eval:end_eval, :]

    df_nowcast_dict = create_model_lag_dict(model_subset, lags, start_eval, end_eval)
    for lag in lags:
        fig, ax = fig_with_size(2)
        # Add truth
        df_truth[f'diff_lag_{lag}'].plot.kde(ax=ax, label='truth', color='black')

        # Plot the KDEs of the nowcasts and the truth data for the specified lag
        for (long_name, short_name) in rename_models.items():
            df_nowcasts = df_nowcast_dict[short_name]
            df_nowcasts[long_name + f'_diff_lag{lag}'].plot.kde(ax=ax, label=short_name)
        ax.legend()
        save_fig(fig, plot_export_path / f'20_kde_lag_{lag}.pdf')


def compute_probabilistic_evaluation(lag_list, model_list):
    """
    This function computes the probabilistic evaluation for a given list of lags and models. It calculates the Brier Score
    for each model and lag, and plots the reliability diagram and histogram of predicted probabilities.

    :param lag_list: list of lags for which the evaluation is to be performed.
    :param model_list: list of models for which the evaluation is to be performed.
    :return: None. The function saves the plots and Brier scores as files.
    """
    # Define markers for the plots
    markers = ["^", "v", "s", "o", '*', 'd', 'P', 'X']

    # Initialize a DataFrame to store the Brier scores
    df_summary = pd.DataFrame(columns=[f'{lag} d' for lag in lag_list], index=[rename_models[m] for m in model_list])

    # Initialize a figure and axes for the histogram plots
    fig_hist, axes_hist = plt.subplots(1, 3, figsize=(default_fig_width, default_fig_height),
                                       layout='constrained')
    axes_hist[0].set(ylabel='Count', xlabel='Predicted prob. of increase', title='1 d')
    axes_hist[1].set(xlabel='Predicted prob. of increase', title='7 d')
    axes_hist[2].set(xlabel='Predicted prob. of increase', title='14 d')

    # Loop over the lags
    for (i_lag, lag) in enumerate(lag_list):
        # Read the probabilistic nowcast data for the current lag
        df_all_models = read_probabilistic_nowcast_data(lag).loc[start_eval_default:end_eval_default]

        # Initialize a figure and axis for the reliability diagram
        fig, ax = fig_with_size(2, factor_height=1.2)

        # Loop over the models
        for (i_model, model) in enumerate(model_list):
            # Compute Brier Score and save as latex table
            p_t = df_all_models[model].values
            z_t = df_all_models['obs'].values
            z_t = z_t[~np.isnan(p_t)]
            p_t = p_t[~np.isnan(p_t)]
            brier_score = np.nanmean((z_t - p_t) ** 2)  # brier score and ignore nans
            df_summary.loc[rename_models[model], f'{lag} d'] = brier_score

            # Compute reliability diagram; separate plot for each lag
            display = CalibrationDisplay.from_predictions(z_t, p_t, n_bins=5, strategy='quantile',
                                                          marker=markers[i_model], ax=ax,
                                                          name=rename_models[model], linewidth=0.5,
                                                          markersize=2, ref_line=False)
            if i_lag == 0:
                axes_hist[i_lag].hist(p_t, bins=10, alpha=0.5, label=rename_models[model], histtype='step', linewidth=0.5)
            else:
                axes_hist[i_lag].hist(p_t, bins=10, alpha=0.5, histtype='step', linewidth=0.5)

        # Add the 'Perfectly calibrated' line manually with the desired name
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perf. calibrated', zorder=1.5, linewidth=0.5)

        # Remove legend from ax
        ax.get_legend().remove()
        ax.set(xlabel='Predicted probability of increase', ylabel='Fraction of increases')

        ax.grid(alpha=0.3, linewidth=0.2)

        # Modify the legend text size
        fig.legend(fontsize=4, loc='outside upper center', ncol=2)

        # Save the reliability diagram as a PDF
        fig.savefig(plot_export_path / f'60_reliability_diagram_lag_{lag}.pdf')
        plt.close(fig)

    # Save the Brier scores as a LaTeX table
    df_summary.to_latex(buf=plot_export_path / '70_brier_scores.tex', float_format='%.4f')

    # Add legend to the histogram plot and save it as a PDF
    fig_hist.legend(fontsize=4, loc='outside upper center', ncol=4)
    fig_hist.savefig(plot_export_path / '70_prob_hist.pdf')


def main():
    lag_list_main = [1, 7, 14]
    # 10 % quantile of values;
    # Choose exclusion areas that are rectangles around zero point -> argument: small noise in both components!
    subset_models_main = ['ILM-prop', 'RIVM-KEW', 'RKI-weekly_report', 'NowcastHub-MedianEnsemble']
    update_mpl_rcparams() # Update MPL params for plots

    import argparse

    argParser = argparse.ArgumentParser()
    argParser.add_argument('--basic_plots', dest='do_basic_plots', action='store_true')
    argParser.add_argument('--compute_scores', dest='do_compute_scores', action='store_true')
    argParser.add_argument('--marginal_analysis', dest='do_marginal_analysis', action='store_true')
    argParser.add_argument('--subset_analysis', dest='do_subset_analysis', action='store_true')
    argParser.add_argument('--prob', dest='do_probabilistic_analysis', action='store_true')
    argParser.add_argument('--all', dest='do_all', action='store_true')

    args = argParser.parse_args()

    if args.do_all:
        for arg in ['do_basic_plots', 'do_compute_scores', 'do_marginal_analysis', 'do_subset_analysis',
                    'do_probabilistic_analysis']:
            setattr(args, arg, True)
    if args.do_basic_plots:
        create_basic_plots()
    if args.do_compute_scores:
        compute_overall_scores()
        compute_ratios(eps_q=[None, 0.1], lags=lag_list_main)
    if args.do_marginal_analysis:
        plot_kdes(lags=lag_list_main)
        create_marginal_analysis_table(lags=lag_list_main)
    if args.do_subset_analysis:
        plot_additional_material_for_subset(subset_models=subset_models_main, lags=lag_list_main)
    if args.do_probabilistic_analysis:
        compute_probabilistic_evaluation(lag_list_main, subset_models_main)


if __name__ == '__main__':
    main()