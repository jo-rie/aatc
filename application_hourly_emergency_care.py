from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

import argparse

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss

from utils_eda import (read_models_from_pickle, model_names,
                       read_probabilistic_models_from_pickle)
from python_utils.mpl_helpers import fig_with_size, save_fig, update_mpl_rcparams, default_fig_width
from setup import get_atc_ratio_name
from aatc import plot_4q, plot_cond_prob, atc_ratio, ExclusionArea, atc_with_bootstrap

plot_export_path = Path('plots/ed_arrival')


def plot_4q_for_models_and_lags(lag_list: list[int], model_list: list[str], df_dict: dict[str, pd.DataFrame],
                                file_prefix: str = '50_4Q', plot_title: bool = False):
    """
    Function to plot 4Q (Quadrant) plots for each model.

    This function iterates over each model and for each lag in the lag_list, it generates a 4Q plot.
    The 4Q plot is a scatter plot of observed differences against forecasted differences for a given lag.
    The plot is saved as a PDF file with a name based on the model and lag.

    Parameters:
    lag_list (list[int]): List of lags for which the 4Q plot should be generated.
    model_list (list[str]): List of models for which the 4Q plot should be generated.
    df_dict (dict[str, pd.DataFrame]): Dictionary where keys are model names and values are pandas DataFrames containing the data for the models.
    file_prefix (str, optional): Prefix for the output file name. Defaults to '4Q'.
    plot_title (bool, optional): If True, the model name is set as the title of the plot. Defaults to False.

    Returns:
    None
    """
    excl_area = ExclusionArea(eps_x=1, eps_y=1, combination='and')
    mpl.use('pdf')
    fig, axes = plt.subplots(figsize=(default_fig_width, default_fig_width / 2), nrows=len(lag_list),
                             ncols=len(model_list))
    for (i_model, model) in enumerate(model_list):
        df = df_dict[model]
        for (i_lag, lag) in enumerate(lag_list):
            ax = axes[i_lag, i_model]
            plot_4q(df.loc[:, f'obs_diff_lag{lag}d'], df.loc[:, f'forecast_diff_lag{lag}d'], ax=ax, excl_area=excl_area)
            if plot_title:
                ax.set_title(f'{model} - lag {lag}d')
    save_fig(fig, plot_export_path / f'{file_prefix}_4q.pdf')
    # plt.close('all')
    mpl.use('pgf')


def excl_area_plot(lag_list: list[int], model_list: list[str], df_dict: dict[str, pd.DataFrame],
                   eps_linspace: np.ndarray = np.linspace(0, 20, 500),
                   file_name: str = 'Accuracies_exclusion_area'):
    """ Plot accuracy over exclusion area - only in x"""
    fig, axes = plt.subplots(len(lag_list), figsize=(default_fig_width, default_fig_width), squeeze=False)
    axes = np.atleast_2d(axes)
    for (i_lag, lag) in enumerate(lag_list):
        for (i_m, m) in enumerate(model_list):
            df = df_dict[m]
            acc_x = np.zeros_like(eps_linspace)
            acc_xy = np.zeros_like(eps_linspace)
            for (i_eps, eps) in enumerate(eps_linspace):
                excl_area_x = ExclusionArea(eps_x=eps)
                excl_area_xy = ExclusionArea(eps_x=eps, eps_y=eps, combination='or')
                acc_x[i_eps] = atc_ratio(df.loc[:, f'obs_diff_lag{lag}d'], df.loc[:, f'forecast_diff_lag{lag}d'],
                                         excl_area=excl_area_x)
                acc_xy[i_eps] = atc_ratio(df.loc[:, f'obs_diff_lag{lag}d'], df.loc[:, f'forecast_diff_lag{lag}d'],
                                          excl_area=excl_area_xy)
            axes[i_lag, 0].plot(eps_linspace, acc_x, label=f'{m} - x', color=f'C{i_m}', alpha=0.8)
            axes[i_lag, 0].plot(eps_linspace, acc_xy, label=f'{m} - xy', color=f'C{i_m}', linestyle='dashed', alpha=0.8)
        axes[i_lag, 0].set(title=f'Lag {lag} days')
        axes[i_lag, 0].legend()
    save_fig(fig, plot_export_path / f'{file_name}.pdf')


def cond_prob_plot(lag_list: list[int], model_list: list[str], df_dict: dict[str, pd.DataFrame],
                   file_prefix: str = '50_Cond_Prob'):
    """
    This function generates conditional ATC plots for a given list of models and lags.

    The function iterates over each model and for each lag in the lag_list, it generates a conditional ATC plot.
    The plot is saved as a PDF file with a name based on the file_prefix and lag.

    Parameters:
    lag_list (list[int]): List of lags for which the conditional ATC plot should be generated.
    model_list (list[str]): List of models for which the conditional ATC plot should be generated.
    df_dict (dict[str, pd.DataFrame]): Dictionary where keys are model names and values are pandas DataFrames containing the data for the models.
    file_prefix (str, optional): Prefix for the output file name. Defaults to '50_Cond_Prob'.

    Returns:
    None
    """
    excl_area = ExclusionArea(eps_x=1, eps_y=1, combination='and')
    for (i_lag, lag) in enumerate(lag_list):
        fig, ax = fig_with_size(2)
        for (i_m, m) in enumerate(model_list):
            df = df_dict[m]

            plot_cond_prob(df.loc[:, f'obs_diff_lag{lag}d'], df.loc[:, f'forecast_diff_lag{lag}d'],
                           ax=ax, label=m, excl_area=excl_area)
        fig.legend(fontsize=5, loc='outside upper center', ncol=3)
        ax.set(ylim=(0.5, 1.05), xlabel='x')
        fig.savefig(plot_export_path / f'{file_prefix}_lag_{lag}.pdf')
        plt.close(fig)


def compute_point_evaluation_measures(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    This function computes point evaluation measures for each model in the provided dictionary.

    Parameters:
    df_dict (dict[str, pd.DataFrame]): A dictionary where keys are model names and values are pandas DataFrames containing the data for the models.

    Returns:
    pd.DataFrame: A DataFrame containing the computed point evaluation measures (RMSE, MAE, Count) for each model.
    """

    # Initialize a DataFrame to store the scores for each model
    scores = pd.DataFrame(columns=['RMSE', 'MAE'], index=model_names)

    # Iterate over each model
    for model in model_names:
        df = df_dict[model]

        # Compute the Root Mean Square Error (RMSE) for the model
        scores.loc[model, 'RMSE'] = ((df.loc[:, model] - df.loc[:, 'obs']) ** 2).mean() ** 1 / 2

        # Compute the Mean Absolute Error (MAE) for the model
        scores.loc[model, 'MAE'] = (np.abs(df.loc[:, model] - df.loc[:, 'obs'])).mean()

        # Compute the count of observations for the model
        scores.loc[model, 'Count'] = str(int(df.loc[:, model].count()))

    # Reset the index of the scores DataFrame and sort it by RMSE in ascending order
    # The resulting DataFrame is saved as a LaTeX file
    scores.reset_index(names='Model').sort_values('RMSE', ascending=True).to_latex(
        plot_export_path / '00_point_evaluation_measures.tex',
        column_format='l r r r', float_format="{:,.3f}".format, escape=True, index=False)

    return scores


def compute_atc_ratio_tables(df_dict_main: dict[str, pd.DataFrame], lag_list: list[int],
                             model_list: list[str] = None):
    """
    This function computes ATC ratio tables for a given list of models and lags.

    The function uses bootstrapping to compute the ATC ratio for each model and lag.
    The function returns a DataFrame containing the computed ATC ratios.

    Parameters:
    df_dict_main (dict[str, pd.DataFrame]): A dictionary where keys are model names and values are pandas DataFrames containing the data for the models.
    lag_list (list[int]): A list of lags for which the ATC ratio should be computed.
    model_list (list[str], optional): A list of models for which the ATC ratio should be computed. If not provided, the function will use the global variable model_names.

    Returns:
    pd.DataFrame: A DataFrame containing the computed ATC ratios.
    """
    rng = np.random.default_rng(42)  # Random Number generator for bootstrapping

    excl_area = ExclusionArea(eps_x=1, eps_y=1, combination='and')

    def format_bootstrap(estimator, low, high):
        return r'{' + f'{estimator:.2f}' + r'\newline' + f'({low:.2f}, {high:.2f})' + r'}'

    col_list = [item for sublist in [[get_atc_ratio_name(lag=lag),
                                      get_atc_ratio_name(lag=lag, pos_neg='positive'),
                                      get_atc_ratio_name(lag=lag, pos_neg='negative')] for lag in lag_list] for
                item in sublist]

    # Initialize resulting dataframe
    atc_ratio_df = pd.DataFrame(columns=col_list, index=model_list)
    for lag in lag_list:
        for model in model_list:
            df = df_dict_main[model]
            diffy = df.loc[:, f'obs_diff_lag{lag}d']
            diffx = df.loc[:, f'forecast_diff_lag{lag}d']
            atc_ratio_df.loc[model, get_atc_ratio_name(lag=lag)] = format_bootstrap(
                *atc_with_bootstrap(diffy=diffy, diffx=diffx, excl_area=excl_area, rng=rng))
            atc_ratio_df.loc[model, get_atc_ratio_name(lag=lag, pos_neg='positive')] = \
                format_bootstrap(
                    *atc_with_bootstrap(diffy=diffy, diffx=diffx, excl_area=excl_area, rng=rng, pos_neg='positive'))
            atc_ratio_df.loc[model, get_atc_ratio_name(lag=lag, pos_neg='negative')] = \
                format_bootstrap(*atc_with_bootstrap(diffy=diffy, diffx=diffx, excl_area=excl_area, rng=rng,
                                                     pos_neg='negative'))
    atc_ratio_df.to_latex(buf=plot_export_path / '50_atc_ratio.tex', float_format='%.2f',
                               column_format='l' + r' p{0.11\textwidth}' * 6)
    return atc_ratio_df


def plot_marginal_kdes_per_difference(df_dict: dict[str, pd.DataFrame], lag_list: list[int],
                                      model_list: list[str] = None):
    """
    This function plots the Kernel Density Estimation (KDE) for the differences of each model and the truth data.

    Parameters:
    df_dict (dict): A dictionary where keys are model names and values are pandas DataFrames containing the data for the models.
    lag_list (list): A list of lags for which the KDE should be plotted.
    model_list (list): A list of models for which the KDE should be plotted. If not provided, the function will use the global variable model_names.

    Returns:
    None
    """
    if model_list is None:
        model_list = model_names
    for lag in lag_list:
        fig, ax = fig_with_size(1)
        # Add truth
        df_dict[model_list[0]][f'obs_diff_lag{lag}d'].plot.kde(ax=ax, label='true', color='black')
        # Plot the KDEs of the nowcasts and the truth data for the specified lag

        for model in model_list:
            df = df_dict[model]
            df[f'forecast_diff_lag{lag}d'].plot.kde(ax=ax, label=model)
        ax.legend(fontsize=5, ncol=2)
        save_fig(fig, plot_export_path / f'20_kde_lag_{lag}.pdf')


def compute_marginal_statistics(df_dict_main: dict[str, pd.DataFrame], lag_list: list[int],
                                model_list: list[str] = None):
    """
    This function computes marginal statistics for a given list of models and lags.

    Parameters:
    df_dict_main (dict[str, pd.DataFrame]): A dictionary where keys are model names and values are pandas DataFrames containing the data for the models.
    lag_list (list[int]): A list of lags for which the marginal statistics should be computed.
    model_list (list[str], optional): A list of models for which the marginal statistics should be computed. If not provided, the function will use the global variable model_names.

    Returns:
    pd.DataFrame: A DataFrame containing the computed marginal statistics.
    """

    # If no model list is provided, use the global variable model_names
    if model_list is None:
        model_list = model_names

    # Helper function to generate column names for standard deviation
    def std_col_name(lag_local: int) -> str:
        return r'$\sigma_{x^{\Delta, ' + f'{lag_local}' + r'}}$'

    # Helper function to generate column names for quantiles
    def quantile_col_name(lag_local: int) -> str:
        return r'$q_{0.1} (x^{\Delta, ' + f'{lag_local}' + r'})$'

    # Helper function to generate column names for fraction of values greater than or equal to 0
    def frac_geq_0(lag_local: int) -> str:
        return f'(1), l={lag_local}'

    # Helper function to generate column names for fraction of absolute values greater than or equal to 1
    def frac_abs_geq_1(lag_local: int) -> str:
        return f'(2), l={lag_local}'

    # Generate column names for the resulting DataFrame
    column_list = [item for sublist in [
        [frac_geq_0(lag), std_col_name(lag), quantile_col_name(lag), frac_abs_geq_1(lag)] for lag in lag_list] for item
                   in sublist]

    # Initialize the resulting DataFrame
    df_result = pd.DataFrame(index=model_list + ['True'],
                             columns=column_list)

    # Compute the marginal statistics for each model and lag
    for lag in lag_list:
        for model in model_list:
            df = df_dict_main[model]
            df_result.loc[model, frac_geq_0(lag)] = (df.loc[:, f'forecast_diff_lag{lag}d'] >= 0).mean()
            df_result.loc[model, std_col_name(lag)] = df.loc[:, f'forecast_diff_lag{lag}d'].std()
            df_result.loc[model, quantile_col_name(lag)] = np.nanquantile(
                df.loc[:, f'forecast_diff_lag{lag}d'].abs(), 0.1)
            df_result.loc[model, frac_abs_geq_1(lag)] = (df.loc[:, f'forecast_diff_lag{lag}d'].abs() >= 1).mean()

        # Compute the marginal statistics for the true values
        df_result.loc['True', frac_geq_0(lag)] = (df_dict_main[model_list[0]].loc[:, f'obs_diff_lag{lag}d'] >= 0).mean()
        df_result.loc['True', std_col_name(lag)] = df_dict_main[model_list[0]].loc[:, f'obs_diff_lag{lag}d'].std()
        df_result.loc['True', quantile_col_name(lag)] = np.quantile(
            df_dict_main[model_list[0]].loc[:, f'obs_diff_lag{lag}d'].abs(), 0.1)
        df_result.loc['True', frac_abs_geq_1(lag)] = (
                    df_dict_main[model_list[0]].loc[:, f'obs_diff_lag{lag}d'].abs() >= 1).mean()

    # Save the resulting DataFrame as a LaTeX file
    df_result.to_latex(buf=plot_export_path / '10_marginal_analysis.tex', float_format='%.2f')

    return df_result


def compute_probabilistic_evaluation(df_dict_prob, lag_list, model_list):
    """
    This function computes the probabilistic evaluation for a given list of lags and models. It calculates the Brier Score
    for each model and lag, and plots the reliability diagram and histogram of predicted probabilities.

    :param df_dict_prob: dictionary of dataframes for each model.
    :param lag_list: list of lags for which the evaluation is to be performed.
    :param model_list: list of models for which the evaluation is to be performed.
    :return: None. The function saves the plots and Brier scores as files.
    """
    # Define markers for the plots
    markers = ["^", "v", "s", "o", '*', 'd', 'P', 'X']

    # Initialize a DataFrame to store the Brier scores
    df_summary = pd.DataFrame(columns=[f'{lag} d' for lag in lag_list], index=model_list)

    # Loop over the lags
    for lag in lag_list:
        # Initialize a figure and axis for the reliability diagram
        fig, ax = fig_with_size(3, factor_height=1.2)

        # Loop over the models
        for (i_model, model) in enumerate(model_list):
            # Compute Brier Score and save as latex table
            df = df_dict_prob[model]
            brier_score = brier_score_loss(df[f'obs_diff_lag{lag}d'] > 0, df[f'p_lag{lag}d'])
            df_summary.loc[model, f'{lag} d'] = brier_score

            # Compute reliability diagram; separate plot for each lag
            display = CalibrationDisplay.from_predictions(df[f'obs_diff_lag{lag}d'] > 0, df[f'p_lag{lag}d'], n_bins=10,
                                                          marker=markers[i_model], ax=ax, name=model, linewidth=0.5,
                                                          markersize=2, ref_line=False)

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


def main():
    """
    Main function to perform analysis on models.

    This function reads models from a pickle file, performs various types of analysis on them, and plots the results.
    The types of analysis performed are conditional ATC plot, 4Q plot, and accuracy computation.
    The function also supports command line arguments to perform marginal analysis, basic analysis of all models,
    and specialized analysis of a subset of the models.

    Command line arguments:
    --marginal: Perform marginal analysis
    --basic: Perform basic analysis of all models
    --subset: Perform specialized analysis of a subset of the models

    Note: The subset of models should be passed as space-separated values after the --subset argument.
    """

    # Define the list of lags
    lag_list_def = [3, 7]
    # Define the subset of models for further analysis
    models_subset = ['Benchmark-1', 'GBM-2', 'NBI-2', 'Poisson-2', 'qreg-1']

    # Update matplotlib rcparams for consistent plot styling
    update_mpl_rcparams()

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Analysis parameters')

    # Add command line arguments
    parser.add_argument('--marginal', action='store_true', help='Perform marginal analysis')
    parser.add_argument('--basic', action='store_true', help='Perform basic analysis of all models')
    parser.add_argument('--subset', action='store_true', help='Perform specialized analysis of a subset of the models')
    parser.add_argument('--probabilistic', action='store_true',
                        help='Perform probabilistic analysis of a subset of the models')
    parser.add_argument('--all', action='store_true', help='Perform all analyses')

    # Parse command line arguments
    args = parser.parse_args()

    # Read and save all models from R
    # read_and_save_all_models_from_r(lag_list_def)

    # Read models from pickle file
    df_dict_main = read_models_from_pickle(lag_list=lag_list_def, model_list=model_names)

    if args.all:
        args.marginal = True
        args.basic = True
        args.subset = True
        args.probabilistic = True

    # Perform marginal analysis if the corresponding command line argument is passed
    if args.marginal:
        compute_point_evaluation_measures(df_dict_main)
        compute_marginal_statistics(df_dict_main, lag_list=lag_list_def, model_list=model_names)
        print('Marginal analysis completed')

    # Perform basic analysis of all models if the corresponding command line argument is passed
    if args.basic:
        compute_atc_ratio_tables(df_dict_main, lag_list=lag_list_def, model_list=model_names)
        plot_marginal_kdes_per_difference(df_dict_main, lag_list=lag_list_def, model_list=model_names)
        print('Basic analysis completed')

    # Perform specialized analysis of a subset of the models if the corresponding command line argument is passed
    if args.subset:
        plot_4q_for_models_and_lags(lag_list=lag_list_def, model_list=models_subset, df_dict=df_dict_main)
        # excl_area_plot(lag_list=lag_list_def, model_list=models_subset, df_dict=df_dict_main)
        cond_prob_plot(lag_list=lag_list_def, model_list=models_subset, df_dict=df_dict_main)
        print('Specialized analysis completed')

    # Perform probabilistic analysis
    if args.probabilistic:
        df_dict_prob = read_probabilistic_models_from_pickle(lag_list=lag_list_def, model_list=models_subset)
        compute_probabilistic_evaluation(df_dict_prob, lag_list=lag_list_def, model_list=models_subset)


if __name__ == '__main__':
    main()
