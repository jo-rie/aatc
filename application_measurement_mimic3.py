import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wfdb import rdrecord

from aatc import plot_4q, plot_cond_prob, ExclusionArea, atc_with_bootstrap
from python_utils.mpl_helpers import fig_with_size, save_fig, update_mpl_rcparams, default_fig_width
from python_utils.utils import Timer
from setup import get_atc_ratio_name
from utils_mimic import download_signal_data, download_files, \
    local_db_path, get_sig_index, create_file_list_with_abp_nbp

plot_path = Path('plots', 'mimic')


def run_sequential_analysis(lag: int, sys_mean_dias: str, export_path: Path = None) -> pd.DataFrame:
    """
    Run the sequential analysis for all files with both abp and nbp values.
    :param export_path: path to export the results
    :param lag: lag to use to count how many nbp and abp measurements satisfy the condition
    :return: dataframe with the results
    """
    file_list_path = Path(local_db_path, 'keys_with_both_bp.txt')
    file_list = []
    with open(file_list_path, 'r') as f:
        for line in f:
            file_list.append('measurement_data/mimic3/' + line.strip())  # list without extension
    df_summary = pd.concat(
        [pd.DataFrame(analyze_single_file(Path(file), lag=lag, sys_mean_dias=sys_mean_dias), index=[0]) for file in
         file_list],
        ignore_index=True)
    if export_path is not None:
        df_summary.to_pickle(export_path.with_suffix('.pickle'))
        df_summary.to_csv(export_path.with_suffix('.csv'))
    # print(df_summary.sort_values('max_sequence', ascending=False).head(10))
    print(f'Lag: {lag}, sys_mean_dias: {sys_mean_dias}')
    print(f'Number of files with at least one sequential measurement: '
          f'{len(df_summary[df_summary["count_successive"] > 0])}')
    return df_summary


def analyze_single_file(path: Path, lag: int, sys_mean_dias: str) -> dict[str, Any]:
    """ Analyze for a single file the structure of NBP and ABP values

    :param path: path to the file to analyze (without extension)
    :param lag: lag to use to count how many nbp and abp measurements satisfy the condition
    :param sys_mean_dias: 'sys', 'mean', or 'dias'
    :return: a dictionary with entries specifying various parameters for the abp and nbp values
    """
    # Load data, Replace 0 by nan
    record = rdrecord(str(path.with_suffix('')))
    df = record.to_dataframe()
    time_length = record.get_elapsed_time(1)
    # Number of ABP values
    try:
        nbp_index = get_sig_index(list(df.columns), nbp_abp='nbp', sys_mean_dias=sys_mean_dias)
        abp_index = get_sig_index(list(df.columns), nbp_abp='abp', sys_mean_dias=sys_mean_dias)
    except ValueError:
        return {'path': path, 'nbp_count': 0, 'abp_count': 0, 'abp_nbp_count': 0, 'count_successive': 0,
                'max_sequence': 0, 'step_delta': time_length}
    abp_count = (df.iloc[:, abp_index] > 0).sum()
    nbp_count = (df.iloc[:, nbp_index] > 0).sum()
    # Number of ABP & NBP values
    df['abp_and_nbp'] = (df.iloc[:, abp_index] > 0) & (df.iloc[:, nbp_index] > 0)
    abp_nbp_count = df['abp_and_nbp'].sum()
    # Number of two ABP / NBP values after each other
    abp_nbp_index = df.columns.get_loc('abp_and_nbp')
    count_succ, max_sequence, current_sequence = 0, 0, 0  # count the number of present successive values, number of most elements in sequence
    for i in range(len(df) - lag):
        if (df.iloc[i, abp_nbp_index] == 1) & (df.iloc[i + lag, abp_nbp_index] == 1):
            # print(f'{path}: {i}/{i + 1}: {abp_and_nbp[i, :]} - {abp_and_nbp[i+1, :]}')
            count_succ += 1
            current_sequence += 1
            if current_sequence > max_sequence:
                max_sequence = current_sequence
        else:
            current_sequence = 0
    return {'path': path, 'nbp_count': nbp_count, 'abp_count': abp_count, 'abp_nbp_count': abp_nbp_count,
            'count_successive': count_succ, 'max_sequence': max_sequence, 'step_delta': time_length}


def get_middle_difference(path: Path, lag: int, sys_mean_dias: str):
    """ Return the difference between the middle values (with respect to all successive measurements of lag lag) of ABP and NBP for the file specified by path.
    :param path: path to the file to analyze (without extension)
    :param lag: lag to use when differencing nbp and abp measurements
    :return: a tuple with the differences between the middle values of ABP and NBP"""
    record = rdrecord(str(path.with_suffix('')))
    df = record.to_dataframe()
    nbp_index = get_sig_index(list(df.columns), nbp_abp='nbp', sys_mean_dias=sys_mean_dias)
    abp_index = get_sig_index(list(df.columns), nbp_abp='abp', sys_mean_dias=sys_mean_dias)
    # Add column to indicate when both values are present
    df['abp_and_nbp'] = (df.iloc[:, abp_index] > 0) & (df.iloc[:, nbp_index] > 0)
    abp_nbp_index = df.columns.get_loc('abp_and_nbp')
    diffs_set: list[tuple[float, float]] = []
    # Iterate over all rows and check if the condition of successive measurements is satisfied
    for i in range(len(df) - lag):
        if (df.iloc[i, abp_nbp_index] == 1) & (df.iloc[i + lag, abp_nbp_index] == 1):
            diffs_set.append((
                df.iloc[i + lag, abp_index] - df.iloc[i, abp_index],
                df.iloc[i + lag, nbp_index] - df.iloc[i, nbp_index]
            ))
    return diffs_set[int(len(diffs_set) / 2)]


def save_sequential_data(lag: int, sys_mean_dias: str):
    """ Reads the summary of sequential analysis for a given lag and blood pressure measurement type, filters the summary to include only files with at least one sequential measurement, creates a vector of differences, and saves this vector.

    :param lag: The lag used in the sequential analysis. This is a part of the filename to distinguish between different analyses.
    :param sys_mean_dias: The type of blood pressure measurement used in the analysis. This can be 'sys', 'mean', or 'dias'. This is also a part of the filename to distinguish between different analyses.
    """
    # Read pickle file with summary of sequential analysis for lag args.lag
    df_summary = pd.read_pickle(local_db_path / f'sequential_analysis_lag{lag}_{sys_mean_dias}.pickle')
    # Restrict df_summary to files with at least one sequential measurement
    df_summary = df_summary[df_summary['count_successive'] > 0]
    # Create Vector of differences
    diff_vector = np.array([get_middle_difference(file, lag, sys_mean_dias) for file in df_summary['path']])
    # Save diff_vector
    np.save(get_numpy_filename(lag, sys_mean_dias), diff_vector)


def get_numpy_filename(lag: int, sys_mean_dias: str) -> Path:
    """ Generate a filename for saving numpy data related to the sequential analysis of ABP and NBP values.

    :param lag: The lag used in the sequential analysis. This is a part of the filename to distinguish between
    different analyses.
    :param sys_mean_dias: The type of blood pressure measurement used in the analysis. This can
    be 'sys', 'mean', or 'dias'. This is also a part of the filename to distinguish between different analyses.
    :return: A Path object representing the filename. The filename is in the format 'diff_vector_lag{lag}_{
    sys_mean_dias}.npy', where {lag} and {sys_mean_dias} are replaced by the input parameters.
    """
    return local_db_path / f'diff_vector_lag{lag}_{sys_mean_dias}.npy'


def sys_mean_to_str(sys_mean_dias: str) -> str:
    """ Convert the abbreviated blood pressure measurement type to its full form.

    :param sys_mean_dias: The abbreviated type of blood pressure measurement. This can be 'sys', 'mean', or 'dias'.
    :return: The full form of the blood pressure measurement type. This can be 'systolic', 'mean', or 'diastolic'.
    """
    if sys_mean_dias == 'sys':
        return 'systolic'
    elif sys_mean_dias == 'mean':
        return 'mean'
    elif sys_mean_dias == 'dias':
        return 'diastolic'


def create_cond_prob_plots(lag_list: list[int], type_list: list[str] = None):
    """ Create conditional probability plots for different lags and blood pressure measurement types.

    :param lag_list: A list of lags to use in the analysis.
    :param type_list: A list of blood pressure measurement types to use in the analysis. This can be 'sys', 'mean', or 'dias'. If not provided, it defaults to ['sys', 'mean'].
    """
    if type_list is None:
        type_list = ['sys', 'mean']
    for lag in lag_list:
        fig, ax = fig_with_size(3)
        for sys_mean_dias in type_list:
            print(f'Creating conditional probability plot for lag {lag} and sys_mean_dias {sys_mean_dias}')
            # Load the difference vector for the given lag and blood pressure measurement type
            diff_vector = np.load(get_numpy_filename(lag, sys_mean_dias))
            # Define the exclusion area for the plot
            excl_area = ExclusionArea.from_quantile(diff_vector[:, 0], diff_vector[:, 1], 0.1)
            # Create the conditional probability plot
            plot_cond_prob(diff_vector[:, 0], diff_vector[:, 1], ax=ax, excl_area=excl_area, label=f'{sys_mean_dias}')
        # Save the plot to a file
        ax.set(ylim=(0.4, 1.05))
        ax.legend()
        save_fig(fig, plot_path / f'cond_prob_diff_nbp_abp_lag{lag}.pdf')


def compute_tr_table(lag_list: list[int], type_list=None):
    """ Compute a summary table of ATC ratios for different lags and blood pressure measurement types.

    :param type_list: A list of blood pressure measurement types to use in the analysis. This can be 'sys', 'mean', or 'dias'.
    :param lag_list: A list of lags to use in the analysis.
    """
    if type_list is None:
        type_list = ['sys', 'mean']
    df_summary = pd.DataFrame(
        index=[f'{sys_mean_dias}, {lag}' for sys_mean_dias in type_list for lag in lag_list],
        columns=['Type', '$l$', get_atc_ratio_name(lag='l'),
                 get_atc_ratio_name(lag='l', pos_neg='positive'),
                 get_atc_ratio_name(lag='l', pos_neg='negative')]
    )
    rng = np.random.default_rng(42)

    def format_bootstrap(estimator, low, high):
        """ Format the bootstrap estimates for the ATC ratio.

        :param estimator: The point estimate of the ATC ratio.
        :param low: The lower bound of the confidence interval for the ATC ratio.
        :param high: The upper bound of the confidence interval for the ATC ratio.
        :return: A string representing the formatted bootstrap estimate.
        """
        return r'{' + f'{estimator:.2f}' + f' ({low:.2f}, {high:.2f})' + r'}'

    for lag in lag_list:
        for sys_mean_dias in type_list:
            diff_vector = np.load(get_numpy_filename(lag, sys_mean_dias))
            excl_area = ExclusionArea.from_quantile(diff_vector[:, 0], diff_vector[:, 1], 0.1)
            df_summary.loc[f'{sys_mean_dias}, {lag}'] = [
                sys_mean_to_str(sys_mean_dias).capitalize(),
                lag,
                format_bootstrap(*atc_with_bootstrap(
                    diffy=diff_vector[:, 0], diffx=diff_vector[:, 1], excl_area=excl_area, rng=rng)),
                format_bootstrap(*atc_with_bootstrap(
                    diffy=diff_vector[:, 0], diffx=diff_vector[:, 1], excl_area=excl_area, rng=rng,
                    pos_neg='positive')),
                format_bootstrap(*atc_with_bootstrap(
                    diffy=diff_vector[:, 0], diffx=diff_vector[:, 1], excl_area=excl_area, rng=rng,
                    pos_neg='negative'))]

    df_summary.to_latex(buf=plot_path / 'atc_ratio.tex', float_format='%.2f',
                        column_format='l' + ' l' + r' p{0.2\textwidth}' * 3, index=False,
                        columns=[])


def create_4q_plots(lag_list: list[int], type_list: list[str] = None):
    """ Create 4Q plots for different lags and blood pressure measurement types.

    This function creates 4Q plots for each combination of lag and blood pressure measurement type.
    The plots are saved to a file named 'plot_4q.pdf' in the 'plot_path' directory.

    :param lag_list: A list of lags to use in the analysis.
    :param type_list: A list of blood pressure measurement types to use in the analysis. This can be 'sys', 'mean', or
    'dias'. If not provided, it defaults to ['sys', 'mean'].
    """
    if type_list is None:
        type_list = ['sys', 'mean']
    fig, axes = plt.subplots(len(type_list), len(lag_list), figsize=(default_fig_width, default_fig_width / 2), layout='constrained')
    for (i_lag, lag) in enumerate(lag_list):
        for (i_sys, sys_mean_dias) in enumerate(type_list):
            # print(f'Creating 4Q plot for lag {lag} and sys_mean_dias {sys_mean_dias}')
            # Load the difference vector for the given lag and blood pressure measurement type
            diff_vector = np.load(get_numpy_filename(lag, sys_mean_dias))
            # Define the exclusion area for the plot
            excl_area = ExclusionArea.from_quantile(diff_vector[:, 0], diff_vector[:, 1], 0.1)
            # Create the 4Q plot
            plot_4q(diff_vector[:, 0], diff_vector[:, 1], ax=axes[i_sys, i_lag], excl_area=excl_area)
            # print(f'Number of points in 4Q plot: {len(diff_vector)}')
            axes[i_sys, i_lag].set(title=f'{sys_mean_to_str(sys_mean_dias).capitalize()}, horizon {lag} min.', xlabel=r'$\Delta ABP$', ylabel=r'$\Delta NBP$')
    # Save the figure to a file
    save_fig(fig, plot_path / f'plot_4q.pdf')


def main():
    """
    The main function of the script. It parses command line arguments and performs various operations based on the provided arguments.

    The function supports the following command line arguments:
    --max_nb: The maximum number of files to download for debugging.
    --downloadVars: If provided, the function will download the variables per numeric file in the database.
    --downloadFiles: If provided, the function will download the files with both abp and nbp values.
    --analyze_sequential: If provided, the function will analyze and save sequential data.
    --4qplot: If provided, the function will create 4Q plots.
    --condprob: If provided, the function will create conditional probability plots.
    --computeTR: If provided, the function will compute a summary table of ATC ratios.
    --all: If provided, the function will perform all of the above operations.

    :return: None
    """
    update_mpl_rcparams()
    default_lags = [1, 5, 15]

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--max_nb", type=int, default=None,
                           help="maximum number of files to download for debugging")
    argparser.add_argument('--downloadVars', dest='download_vars', action='store_true',
                           help='Download the variables per numeric file in the database.')
    argparser.add_argument('--downloadFiles', dest='download_files_with_abp_nbp', action='store_true',
                           help='Download the files with both abp and nbp values.')

    argparser.add_argument('--analyze_sequential', dest='analyze_and_save_sequential', action='store_true')
    argparser.add_argument('--4qplot', dest='do_4q_plot', action='store_true')
    argparser.add_argument('--condprob', dest='do_cond_prob_plot', action='store_true')
    argparser.add_argument('--computeTR', dest='do_compute_tr_table', action='store_true')
    argparser.add_argument('--all', dest='all', action='store_true')

    args = argparser.parse_args()

    # print(args)

    if args.all:
        for arg in ['do_4q_plot', 'do_cond_prob_plot', 'do_compute_tr_table']:
            setattr(args, arg, True)
    if args.download_vars:
        print(f'Starting to Download signal data at {datetime.now()}')
        with Timer('Time for downloading headers'):
            raw_dict = download_signal_data(max_nb_files=args.max_nb)
        print(f'Starting to analyze signal data at {datetime.now()}')
        for sys_mean_diast in ['sys', 'mean', 'dias']:
            create_file_list_with_abp_nbp(sys_mean_diast)

    if args.download_files_with_abp_nbp:
        print(f'Starting to download files with both abp and nbp values at {datetime.now()}')
        with Timer('Time for downloading all files'):
            download_files(local_db_path / 'keys_with_both_bp.txt')

    if args.analyze_and_save_sequential:
        print(f'Starting to analyze and save sequential data at {datetime.now():%H:%M:%S}')
        for lag in default_lags:
            for sys_mean_dias in ['sys', 'mean', 'dias']:
                export_path = Path(local_db_path, f'sequential_analysis_lag{lag}_{sys_mean_dias}')
                run_sequential_analysis(lag=lag, sys_mean_dias=sys_mean_dias, export_path=export_path)
                save_sequential_data(lag, sys_mean_dias)

    if args.do_4q_plot:
        print(f'Starting 4Q plots at {datetime.now():%H:%M:%S}')
        create_4q_plots(lag_list=default_lags)

    if args.do_cond_prob_plot:
        print(f'Starting conditional probability plots at {datetime.now():%H:%M:%S}')
        create_cond_prob_plots(lag_list=default_lags)

    if args.do_compute_tr_table:
        print(f'Starting to compute ATC ratio table at {datetime.now():%H:%M:%S}')
        compute_tr_table(lag_list=default_lags)


if __name__ == '__main__':
    main()
