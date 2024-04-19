from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

data_dir_r = Path('hourly-emergency-care/results_to_python')
data_dir_r_quantiles = Path('hourly-emergency-care/results_to_python_quantiles')
em_data_base = Path('em_arrival/data')


def get_model_file_names() -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]:
    """
    Get the names of the model files.

    This function returns the names of the model files, the names of the models, a dictionary mapping file names to model names, and a dictionary mapping model names to file names. The file and model names are sorted in ascending order.

    :return: A tuple containing a list of file names, a list of model names, a dictionary mapping file names to model names, and a dictionary mapping model names to file names.
    """
    default_file_names = ["Benchmark_1", "Benchmark_2", "Poisson-GAM-te_v1", "Poisson-GAM-te_v2", "gamlss-NOtr_v1",
                          "gamlss-NOtr_v2", "GBM", "gamlss-TF2tr_v3", "gamlss-NBI_v4", "qreg_boost_V1",
                          "gamlss-PO_Ilink_v1", "gamlss-NBI_Ilink_v4", "tbats", "faster", "iETSXSeasonal", "ETS(XXX)",
                          "RegressionPoisson", "iETSCeiling", "gamlss-GA_v2", 'prophet']
    default_model_names = ["Benchmark-1", "Benchmark-2", "Poisson-1", "Poisson-2", "NOtr-1",
                           "NOtr-2", "GBM-2", "Ttr-2", "NBI-2", "qreg-1",
                           "Poisson-2-I", "NBI-2", "tbats", "fasster", "ADAM-iETSX", "ETS",
                           "Regression-Poisson", "ADAM-iETSX-Ceiling", "ZAGA-2", 'Prophet']
    file_model_dict = dict(zip(default_file_names, default_model_names))
    file_model_dict.pop('iETSCeiling', None)  # Does not contain expectation
    file_model_dict.pop('faster', None)  # Missing
    file_model_dict.pop('gamlss-GA_v2', None)  # Missing
    file_model_dict.pop('gamlss-NBI_Ilink_v4', None)  # Missing

    # Create lists for returning
    files_list = list(file_model_dict.keys())
    models_list = list(file_model_dict.values())
    model_file_dict = dict(zip(models_list, files_list))

    files_list.sort()
    models_list.sort()

    return files_list, models_list, file_model_dict, model_file_dict


file_names, model_names, file2model, model2file = get_model_file_names()


def read_obs_from_r() -> pd.DataFrame:
    """
    Read the observations from a CSV file exported from R and return it as a DataFrame.

    This function reads a CSV file named 'observations.csv' located in the directory specified by the global variable 'data_dir_r'.
    The 'targetTime_UK' column is parsed as dates and set as the index of the DataFrame.
    Only the columns 'targetTime_UK' and 'n_attendance' are used, and 'n_attendance' is renamed to 'obs'.

    :return: A DataFrame containing the observations read from the CSV file.
    """
    obs = pd.read_csv(data_dir_r / 'observations.csv', parse_dates=['targetTime_UK'], index_col='targetTime_UK',
                      usecols=[1, 3]).rename(columns={'n_attendance': 'obs'})
    return obs


def read_model_from_r(model_name: str, lag_list: List[int], obs: pd.DataFrame = None,
                      export_to_file: bool = False) -> pd.DataFrame:
    """
    Read the specified model from the exported file from R, add columns based on the lags in lag_list.

    This function reads a specified model from a file exported from R. It then adds columns to the DataFrame based on
    the lags specified in lag_list. If an obs DataFrame is provided, it will not be read from a file again. If
    export_to_file is set to True, the created DataFrame will be saved to 'em_arrival/data/{model_name}.pickle'.

    :param model_name: The name of the model to be read.
    :param lag_list: A list of lags to be used for the difference columns.
    :param obs: A DataFrame containing the observations. If None, it will be read from a file. Defaults to None.
    :param export_to_file: If True, the created DataFrame will be saved to 'em_arrival/data/{model_name}.pickle'. Defaults to False.
    :return: A DataFrame containing the processed model data.
    """
    df = read_base_model_from_r(model_name, base_path=data_dir_r, obs=obs)

    for (i_d, d) in enumerate(lag_list):
        df.loc[:, f'target_lag{d}d'] = df.loc[:, 'targetTime_UK'] - pd.Timedelta(days=d)
        df = df.merge(obs, left_on=f'target_lag{d}d', right_index=True, how='left', suffixes=(None, f'_lag{d}d'))
        df.loc[:, f'forecast_diff_lag{d}d'] = df.loc[:, 'expectation'] - df.loc[:, f'obs_lag{d}d']
        df.loc[:, f'obs_diff_lag{d}d'] = df.loc[:, 'obs'] - df.loc[:, f'obs_lag{d}d']

    df = df.rename(columns={'expectation': model_name})

    if export_to_file:
        df.to_pickle(em_data_base / f'{model_name}.pickle')

    return df


def read_base_model_from_r(model_name: str, base_path: Path, obs: pd.DataFrame) -> pd.DataFrame:
    """
    Read the specified model from the exported file from r. The data is restricted to the longest lead time.

    :param model_name: The name of the model to be read.
    :param base_path: The path where the model file is located.
    :return: A DataFrame containing the processed model data.
    """
    if obs is None:
        obs = read_obs_from_r()

    file_name = model2file[model_name]

    df = pd.read_csv(
        base_path / f'{file_name}.csv', parse_dates=[0, 1], index_col=[0, 1]
    )

    df = df.reset_index()
    df['leadTime'] = df['targetTime_UK'] - df['issueTime']
    df = df[(df['leadTime'] >= pd.Timedelta(days=1, hours=12)) & (df['leadTime'] < pd.Timedelta(days=2))]

    df = df.merge(obs, left_on='targetTime_UK', right_index=True, how='left')

    return df


def read_and_save_all_models_from_r(lag_list: List[int]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """ Read all models from r and save them to pickle files. Return a dictionary with the dataframes and the
    observations.
    :param lag_list: List of lags to be used for the difference columns.
    :return: Tuple of dictionary with the dataframes and the observations.
    """
    obs = read_obs_from_r()
    df_dict: dict[str, DataFrame] = dict()
    for model_name in model_names:
        df = read_model_from_r(model_name, lag_list=lag_list, obs=obs, export_to_file=True)
        df_dict[model_name] = df
    return df_dict, obs


def read_models_from_pickle(lag_list: list[int], model_list=None) -> Dict[str, pd.DataFrame]:
    """
    Read the models from the pickle files and return them as a dictionary.

    :param lag_list: A list of lags to be used for the difference columns.
    :param model_list: A list of model names to be read. Defaults to None.
    :return: A dictionary where the keys are the model names and the values are the corresponding DataFrames.

    This function reads the specified models from the pickle files and returns them as a dictionary.
    If no model list is provided, it defaults to using the global variable `model_names`.
    If a model does not exist in the pickle file, it calls the `read_model_from_r` function to read the model.
    """
    if model_list is None:
        model_list = model_names
    df_dict: dict[str, DataFrame] = dict()
    for model_name in model_list:
        # check if file em_data_base / f'{model_name}.pickle' exists
        if not (em_data_base / f'{model_name}.pickle').exists():
            df = read_model_from_r(model_name, lag_list=lag_list, export_to_file=True)
        else:
            df = pd.read_pickle(em_data_base / f'{model_name}.pickle')
        df_dict[model_name] = df
    return df_dict


def p_from_quantiles(df: pd.DataFrame, lag: int, method: str = 'uniform') -> np.array:
    """
    Calculate the probability of a decrease using the quantiles.

    :param df: A dataframe with the quantiles. All columns starting with 'q' are used as quantiles and the columns starting with 'obs' is used as the observation with lag l.
    :param lag: The lag to be used for the observation.
    :param method: The method to be used for the calculation. Default is 'uniform'.
    :return: An array of probabilities for each row in the dataframe.

    This function calculates the probability of a decrease in a given observation based on the quantiles.
    It iterates over the rows of the dataframe and calculates the probability P(x < obs_lag_l) for each row.
    The method of calculation can be specified, with 'uniform' being the default.
    """
    # Extract quantiles from column names
    np.seterr(all='raise')

    quantiles_at = [int(c[1:]) / 100 for c in df.columns if c.startswith('q')]

    # Sort quantiles and columns
    quantiles_at.sort()
    quantile_columns = [f'q{int(q*100)}' for q in quantiles_at]

    # Iterate over the rows and calculate the probability P(x < obs_lag_l) for each row
    for i in range(len(df)):
        obs = df.loc[df.index[i], f'obs_lag{lag}d']
        quantiles = df.loc[df.index[i], quantile_columns].values
        quantiles_leq_obs = quantiles <= obs
        indices = np.where(quantiles_leq_obs)[0]  # Get indices where quantiles_leq_obs is True
        if method == 'uniform':
            if indices.size == len(quantiles):
                # obs is larger than largest quantile -> prob of decrease is between q[-1] and 1
                if quantiles[-1] == quantiles[-2]:
                    p = 1
                else:
                    p = min(((obs - quantiles[-1]) / (quantiles[-1] - quantiles[-2]) *
                         (quantiles_at[-1] - quantiles_at[-2]) + quantiles_at[-1]), 1)
            elif indices.size > 0:  # Check if there are any True values
                max_index = np.max(indices)  # Get the largest index
                p = ((obs - quantiles[max_index]) / (quantiles[max_index + 1] - quantiles[max_index]) *
                     (quantiles_at[max_index + 1] - quantiles_at[max_index]) + quantiles_at[max_index])
            else:
                # obs is smaller than smallest quantile -> prob of decrease is between 0 and q[0]
                if quantiles[0] == quantiles[1]:
                    p = 0
                else:
                    p = max(quantiles_at[0] - ((quantiles[0] - obs) /
                                               (quantiles[1] - quantiles[0]) * quantiles_at[0]), 0)
        else:
            raise ValueError(f'Unknown method: {method}')
        df.loc[df.index[i], f'p_lag{lag}d'] = 1 - p  # Probability for an increase
    return df[f'p_lag{lag}d'].values


def create_probabilistic_model_data(model: str, lag_list: list[int], obs: pd.DataFrame = None,
                                    export_to_file: bool = False) -> pd.DataFrame:
    """
    Create a dataframe with the probabilistic model data.

    :param model: The name of the model to be read.
    :param lag_list: A list of lags to be used for the difference columns.
    :param obs: A DataFrame containing the observations. If None, it will be read from a file.
    :param export_to_file: If True, the created dataframe will be saved to 'em_arrival/data/{model}_prob.pickle'.
    :return: A DataFrame containing the processed model data.

    This function reads the specified model from the exported file from r, adds columns based on the lags in lag_list,
    and calculates the probability of a decrease using the quantiles for each lag.
    """
    if obs is None:
        obs = read_obs_from_r()
    df = read_base_model_from_r(model, base_path=data_dir_r_quantiles, obs=obs)

    quantile_columns = [c for c in df.columns if c.startswith('q')]

    for lag in lag_list:
        df.loc[:, f'target_lag{lag}d'] = df.loc[:, 'targetTime_UK'] - pd.Timedelta(days=lag)
        df = df.merge(obs, left_on=f'target_lag{lag}d', right_index=True, how='left', suffixes=(None, f'_lag{lag}d'))
        df.loc[:, f'obs_diff_lag{lag}d'] = df.loc[:, 'obs'] - df.loc[:, f'obs_lag{lag}d']
        df.loc[:, f'p_lag{lag}d'] = p_from_quantiles(df, lag, method='uniform')

    if export_to_file:
        df.to_pickle(em_data_base / f'{model}_prob.pickle')
    return df


def read_probabilistic_models_from_pickle(lag_list: list[int], model_list=None) -> Dict[str, pd.DataFrame]:
    """
    Read the models from the pickle files and return them as a dictionary.

    :param lag_list: A list of lags to be used for the difference columns.
    :param model_list: A list of model names to be read. Defaults to None.
    :return: A dictionary where the keys are the model names and the values are the corresponding DataFrames.

    This function reads the specified models from the pickle files and returns them as a dictionary.
    If no model list is provided, it defaults to using the global variable `model_names`.
    If a model does not exist in the pickle file, it calls the `read_model_from_r` function to read the model.
    """
    if model_list is None:
        model_list = model_names
    df_dict: dict[str, pd.DataFrame] = dict()
    for model_name in model_list:
        if not (em_data_base / f'{model_name}_prob.pickle').exists():
            df = create_probabilistic_model_data(model_name, lag_list=lag_list, export_to_file=True)
        else:
            df = pd.read_pickle(em_data_base / f'{model_name}_prob.pickle')
        df_dict[model_name] = df
    return df_dict
