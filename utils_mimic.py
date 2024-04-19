import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm
from wfdb import dl_files, rdsamp, rdrecord

from python_utils.mpl_helpers import fig_with_size, save_fig

local_db_path = Path('measurement_data/mimic3')
plot_export_path = Path('plots/measurement')


def download_signal_per_file(filename: str) -> list:
    """ Download a single signal name information from the MIMIC-3 database for the specified filename and return it as list.

    :param filename: the filename for which the signal data should be downloaded.
    :return: a list of the signal names."""
    folder, fname = filename.rsplit("/", 1)
    sig_names = rdsamp(fname, pn_dir="mimic3wdb/" + folder, sampto=1)[1]["sig_name"]
    return sig_names


def download_signal_data(max_nb_files: int = None) -> dict[str, list[str]]:
    """ Download for all numerical files in the mimic3 database the data header and save the variables of each file
    to a dict and save it as pickle in "var_per_file_dict.pickle".

    :param max_nb_files: Restricts the number of files to download to the first max_nb_files files
    :return the dictionary with file names as keys and the variables as items"""
    dl_files(db="mimic3wdb", dl_dir=str(local_db_path), files=["RECORDS-numerics"])

    with open(local_db_path / "RECORDS-numerics", "r") as f:
        numeric_file_list = f.read().splitlines()  # remove trailing newlines

    if max_nb_files is not None:
        numeric_file_list = numeric_file_list[:max_nb_files]

    pool = Pool(processes=4)

    results = pool.map(download_signal_per_file, numeric_file_list, chunksize=100)

    raw_dict = dict(zip(numeric_file_list, results))

    with open(local_db_path / "var_per_file_dict.pickle", "wb") as f:
        pickle.dump(raw_dict, f)

    return raw_dict


def download_files(file: str):
    """ Download the files listed in file and save them to the local_db_path directory.

    :param file: file with filenames (without extension)
    :return: the list of files downloaded
    """
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.strip() + '.dat')
            file_list.append(line.strip() + '.hea')
    dl_files(db="mimic3wdb", dl_dir=local_db_path, files=file_list)
    return file_list


def load_result_dict():
    """ Return the variables for the specified MIMIC database files as a dictionary."""
    with open(local_db_path / f"var_per_file_dict.pickle", "rb") as f:
        raw_dict = pickle.load(f)
    return raw_dict


def get_sig_index(sig_names: list[str], nbp_abp: str, sys_mean_dias: str = 'sys'):
    """ Return the index of the nbp systolic and abp systolic in sig_names.
    :param sys_mean_dias: 'sys', 'mean', or 'dias'
    :param nbp_abp: 'nbp' or 'abp'
    :param sig_names: list of the signal names
    :return: index of the nbp and abp systolic column"""
    assert sys_mean_dias in ['sys', 'mean', 'dias']
    assert nbp_abp in ['nbp', 'abp']
    sig_names = [s.lower() for s in sig_names]
    if nbp_abp == 'nbp':
        return sig_names.index(f'nbp{sys_mean_dias}')
    else:
        try:
            abp_index = sig_names.index(f'abp{sys_mean_dias}')
        except ValueError as e:
            abp_index = sig_names.index(f'art{sys_mean_dias}')
        return abp_index


def get_files_with_abp_nbp():
    """ Return the list of files with both abp and nbp values"""
    file_list_path = Path(local_db_path, 'keys_with_both_bp.txt')
    file_list = []
    with open(file_list_path, 'r') as f:
        for line in f:
            file_list.append(local_db_path / line.strip())  # list without extension
    return file_list


def create_overview_file():
    """
    Create an overview file for all files with both abp and nbp values.
    :return: None
    """
    file_list = get_files_with_abp_nbp()
    df_summary = pd.DataFrame(columns=['nbpsys_count', 'abpsys_count', 'nbpmean_count', 'abpmean_count', 'nbpdias_count', 'abpdias_count'], index=file_list)
    for file in tqdm(file_list):
        frame = rdrecord(str(file)).to_dataframe()
        for sys_mean_dias in ['sys', 'mean', 'dias']:
            for nbp_abp in ['nbp', 'abp']:
                try:
                    col_index = get_sig_index(frame.columns, nbp_abp, sys_mean_dias)
                    df_summary.loc[file, f'{nbp_abp}{sys_mean_dias}_count'] = (frame.iloc[:, col_index] > 0).sum()
                except ValueError:
                    df_summary.loc[file, f'{nbp_abp}{sys_mean_dias}_count'] = 0
    df_summary['min_count'] = df_summary.min(axis=1)
    df_summary.sort_values('min_count', ascending=False).to_csv(local_db_path / 'overview_counts.csv')
    return df_summary


def plot_abp_nbp(df_summary):
    """ Plot the abp and nbp values for the files in df_summary and save them to plot_export_path. """
    for file in df_summary.index:
        # Load raw data
        df_loop = pd.read_csv(Path(file).with_suffix('.csv'), index_col=0)
        # Plot
        fig, ax = fig_with_size(1)
        index_nbp = df_loop['nbp'] > 0
        index_abp = df_loop['abp'] > 0
        ax.plot(df_loop.index[index_nbp], df_loop.loc[index_nbp, 'nbp'], linewidth=0.5, marker='o', alpha=0.1,
                label='nbp')
        ax.plot(df_loop.index[index_abp], df_loop.loc[index_abp, 'abp'], linewidth=0.5, marker='o', alpha=0.1,
                label='abp')
        ax.legend()
        save_fig(fig, Path(plot_export_path, Path(file).stem).with_suffix('.pdf'))


def create_file_list_with_abp_nbp(sys_mean_diast: str = 'sys'):
    """ Create a file with the list of files with both abp and nbp values and save it to the local_db_path directory."""
    assert sys_mean_diast in ['sys', 'mean', 'dias']
    result_dict = load_result_dict()
    set_of_all_vars = set([])
    for variable_list in result_dict.values():
        set_of_all_vars.update([v.lower().replace(' ', '') for v in variable_list])
    # print(set_of_all_vars)
    list_off_all_vars = list(set_of_all_vars)
    list_off_all_vars.sort()
    print(f'Number of variables: {len(list_off_all_vars)}')

    keys_with_both = []
    for key, value in tqdm(result_dict.items()):
        value = [f.lower() for f in value]
        if (((f'art{sys_mean_diast}' in value) | (f'abp{sys_mean_diast}' in value)) &
                (f'nbp{sys_mean_diast}' in value)):
            keys_with_both.append(key)
    print(len(keys_with_both))
    with open(local_db_path / f'keys_with_both_bp_{sys_mean_diast}.txt', 'w') as f:
        for key in keys_with_both:
            f.write(key + '\n')
    all_keys = combine_all_keys()
    with open(local_db_path / 'keys_with_both_bp.txt', 'w') as f:
        for key in all_keys:
            f.write(key + '\n')


def combine_all_keys():
    """
    This function combines all keys from three different files: 'keys_with_both_bp_sys.txt',
    'keys_with_both_bp_mean.txt', and 'keys_with_both_bp_dias.txt'.

    :return: a list of unique keys from all three files.
    """
    all_keys = []
    if (local_db_path / 'keys_with_both_bp_sys.txt').exists():
        with open(local_db_path / 'keys_with_both_bp_sys.txt', 'r') as f:
            all_keys += f.read().splitlines()
    if (local_db_path / 'keys_with_both_bp_mean.txt').exists():
        with open(local_db_path / 'keys_with_both_bp_mean.txt', 'r') as f:
            all_keys += f.read().splitlines()
    if (local_db_path / 'keys_with_both_bp_dias.txt').exists():
        with open(local_db_path / 'keys_with_both_bp_dias.txt', 'r') as f:
            all_keys += f.read().splitlines()
    all_keys = list(set(all_keys))
    return all_keys
