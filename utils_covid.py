from datetime import date, datetime
from os import listdir, makedirs
from os.path import isfile, join

import numpy as np
import pandas as pd
from tqdm import trange, tqdm

date_min = date(2021, 4, 6)
date_max = date(2023, 8, 10)

path_data_export = 'covid_nowcasting/data'
path_to_truth_file = 'hospitalization-nowcast-hub/data-truth/COVID-19/rolling-sum/2024-02-08_COVID-19_hospitalization.csv'

model_list = [
    'Epiforecasts-independent', 'ILM-prop', 'KIT-simple_nowcast', 'LMU_StaBLab-GAM_nowcast', 'NowcastHub-MeanEnsemble',
    'NowcastHub-MedianEnsemble', 'RIVM-KEW', 'RKI-weekly_report', 'SU-hier_bayes', 'SZ-hosp_nowcast']
shorthand_list = ['EPI', 'ILM', 'KIT', 'LMU', 'ENS-MEAN', 'ENS-MED', 'RIVM', 'RKI', 'SU', 'SZ']
rename_models = dict(zip(model_list, shorthand_list))
n_samples_prob = 10000
quantile_cols = ['q0.025', 'q0.1', 'q0.25', 'q0.5', 'q0.75', 'q0.9', 'q0.975']
q_quantiles = np.array([float(q[1:]) for q in quantile_cols])
rng = np.random.default_rng(42)


def create_truth_df(age_group: str = '00+', location: str = 'DE', save_file: bool = False):
    """
    The read_truth_data_rolling_sum function reads the truth data from the hospitalization-nowcast-hub/data-truth/COVID-19
    rolling sum folder and arranges it as a data frame with a single column.

    :param age_group: str: Specify the age group of the data that is read (['00+' (all data), '00-04', '05-14', '15-34', '35-59', '60-79', '80+'])
    :param location: str: Specify the location of the data ('DE' for whole Germany), see https://github.com/KITmetricslab/hospitalization-nowcast-hub/wiki/Data-format
    :return: A dataframe with the true values for a given age group and location
    """
    df = pd.read_csv(path_to_truth_file, parse_dates=[0])
    df = df[(df['location'] == location) & (df['age_group'] == age_group)]
    df = df.set_index('date')
    df = df.rename(columns={'value': 'true'})
    df = df[['true']]
    if save_file:
        save_csv_pickle(df, get_path_to_truth())
    return df


def create_nowcast_df(lag: int = 1, save_file: bool = False):
    """
    The read_nowcast_data function takes the data from all of the plot_data files and combines them into one
    pandas DataFrame. It then pivots this DataFrame so that each model is a column, and each date is a row. It adds the nowcasts for lag days before

    :param lag: int: Specify the number of days to shift the data, if 0, only the sameday-nowcasts are returned
    :return: A dataframe with the columns per model and lag, indexed by the date of nowcast
    """
    root_folder = 'hospitalization-nowcast-hub/nowcast_viz_de/plot_data'
    file_list = [f for f in listdir(root_folder) if f.startswith('20')]

    def combine_nowcasts_for_date_offset(offset: int = 0):
        """
        The combine_nowcasts_for_date_offset function takes a date offset as an argument and returns a dataframe with the nowcasts for date_offset days before that day. The function is used to create the nowcast_df, which contains all of the nowcasts for each day in our dataset.

        :param offset: int: Specify the number of days to offset from the nowcast date
        :return: A dataframe with a column for each model.
        """
        pd_list = []
        for f in file_list:
            date = datetime.strptime(f[:10], '%Y-%m-%d')
            new_data = pd.read_csv(join(root_folder, f), parse_dates=[2, 3])
            pd_list.append(
                new_data[(new_data['target_end_date'] == (date - pd.DateOffset(days=offset))) &
                         (new_data['location'] == 'DE') &
                         (new_data['age_group'] == '00+')]
            )
        df_nc = pd.concat(pd_list)
        if np.any(df_nc.groupby(['model', 'forecast_date']).count() > 1):
            df_twice = df_nc.value_counts(['model', 'forecast_date']).reset_index()
            df_nc_with_count = pd.merge(left=df_nc, right=df_twice, on=['model', 'forecast_date'])
            df_nc = df_nc_with_count[(df_nc_with_count['count'] < 2) | (df_nc_with_count['retrospective'])]
        df_nc = df_nc[['model', 'forecast_date', 'mean']]
        df_nc = df_nc.pivot(columns='model', index='forecast_date', values='mean')
        if offset > 0:
            df_nc = df_nc.rename(columns=lambda x: x + f'_target_{offset}d_before')
        return df_nc

    df_sameday = combine_nowcasts_for_date_offset()  # Read same-day-nowcasts

    if lag > 0:
        df_lagday = combine_nowcasts_for_date_offset(lag)  # Read lag-l nowcasts
        df_sameday = df_sameday.join(df_lagday)

    if save_file:
        save_csv_pickle(df_sameday,
                        get_path_to_nowcast(lag=lag))
    return df_sameday


def save_csv_pickle(df: pd.DataFrame, path_without_extension: str):
    """
    The save_csv_pickle function saves a pandas DataFrame to both CSV and pickle formats.
    
    :param df: pd.DataFrame: The pandas dataframe to save
    :param path_without_extension: str: Specify the path to save the file without an extension
    """
    makedirs(path_data_export, exist_ok=True)
    df.to_csv(path_without_extension + '.csv')
    df.to_pickle(path_without_extension + '.pickle')


def sample_from_quantiles(x_quans: np.ndarray, q_quans: np.ndarray = q_quantiles, n_samples: int = n_samples_prob):
    """
    The sample_from_quantiles function takes a list of quantiles as input and returns a random sample from the quantiles.

    :param x_quans: list: Specify the x-location of quantiles
    :param q_quans: list: Specify the q-location of quantiles
    :param n_samples: int: Specify the number of samples to draw
    :return: A random sample from the quantiles
    """
    # Get non-nan values of x_quans
    q_quans = q_quans[~np.isnan(x_quans)]
    x_quans = x_quans[~np.isnan(x_quans)]
    if len(x_quans) < 2:
        return np.full(n_samples, np.nan)
    # extrapolate lower bound
    quantiles_low = (x_quans[0] -
                     (x_quans[1] - x_quans[0]) * (q_quans[0] / (q_quans[1] - q_quans[0])))
    # extrapolate upper bound
    quantiles_high = (x_quans[-1] +
                      (x_quans[-1] - x_quans[-2]) * ((1 - q_quans[-1]) / (q_quans[-1] - q_quans[-2])))
    q_quans = np.concatenate(([0], q_quans, [1]))  # add 0 and 1 to q values of quantiles
    # add lower and upper extrapolation to quantiles
    quantiles_local = np.concatenate(([quantiles_low], x_quans.astype('float64'), [quantiles_high]))

    u_samples = rng.uniform(size=n_samples)
    x_samples = np.interp(u_samples, q_quans, quantiles_local)
    return x_samples


def create_probabilistic_model_for_model_lag(model_list: list[str], lag: int, save_file: bool = False):
    """
    The read_model_quantiles_for_lag function reads the model quantiles for a given lag.

    :param save_file: Whether to save the file
    :param model: Specify model for which to read the quantiles
    :param lag: int: Specify the number of days before the target date that we want to use as a base
    :return: A dataframe with the model quantiles
    """
    root_folder = 'hospitalization-nowcast-hub/nowcast_viz_de/plot_data'
    file_list = [f for f in listdir(root_folder) if f.startswith('20')]

    df_results = pd.DataFrame(index=pd.date_range(date_min, date_max, freq='D'))

    # Iterate over models
    for model in tqdm(model_list, desc=f'Computing probabilistic values for models with lag {lag}...'):
        def combine_nowcasts_for_date_offset(offset: int = 0):
            """
            The combine_nowcasts_for_date_offset function takes a date offset as an argument and returns a dataframe with the nowcasts for date_offset days before that day. The function is used to create the nowcast_df, which contains all of the nowcasts for each day in our dataset.

            :param offset: int: Specify the number of days to offset from the nowcast date
            :return: A dataframe with a column for each quantile.
            """
            pd_list = []
            for f in file_list:
                date = datetime.strptime(f[:10], '%Y-%m-%d')
                new_data = pd.read_csv(join(root_folder, f), parse_dates=[2, 3])
                pd_list.append(
                    new_data[(new_data['target_end_date'] == (date - pd.DateOffset(days=offset))) &
                             (new_data['location'] == 'DE') &
                             (new_data['age_group'] == '00+') &
                             (new_data['model'] == model)]
                )
            df_nc = pd.concat(pd_list)
            if np.any(df_nc.groupby(['model', 'forecast_date']).count() > 1):
                df_twice = df_nc.value_counts(['model', 'forecast_date']).reset_index()
                df_nc_with_count = pd.merge(left=df_nc, right=df_twice, on=['model', 'forecast_date'])
                df_nc = df_nc_with_count[(df_nc_with_count['count'] < 2) | (df_nc_with_count['retrospective'])]
            df_nc = df_nc[['forecast_date'] + quantile_cols]
            df_nc.set_index('forecast_date', inplace=True)
            if offset > 0:
                df_nc = df_nc.rename(columns=lambda x: x + get_lag_column_postfix(offset))
            return df_nc

        df_sameday = combine_nowcasts_for_date_offset()  # Read same-day-nowcasts
        df_lagday = combine_nowcasts_for_date_offset(lag)  # Read lag-l nowcasts
        df_sameday = df_sameday.join(df_lagday)
        # Iterate over rows
        for i in df_sameday.index:
            # If one of the quantile columns is missing, fill with nan
            # Sample from the quantiles, assume uniformity within the quantiles
            quantiles_today = df_sameday.loc[i, quantile_cols].values
            quantiles_lag = df_sameday.loc[i, [q + get_lag_column_postfix(lag) for q in quantile_cols]].values
            df_sameday.loc[i, model] = (
                    (sample_from_quantiles(quantiles_today) - sample_from_quantiles(quantiles_lag)) > 0).mean()
        df_sameday = df_sameday.sort_index()
        df_results = df_results.join(df_sameday[model], how='left')

    # Add observed values
    df_truth = create_df_truth_lags([lag])
    df_truth['obs'] = df_truth[f'diff_lag_{lag}'] > 0

    df_results = df_results.join(df_truth['obs'], how='left')

    if save_file:
        save_csv_pickle(df_results,
                        get_prob_path(lag=lag))

    return df_results


def get_lag_column_postfix(lag: int):
    return f'_target_{lag}d_before'


def get_prob_path(lag: int, ext: str = ''):
    """
    The get_quantile_file_name function returns the path to a probabilistic model file.
    :param lag: the lag of the probabilistic model
    :param ext: optional file extension
    :return: path to the probabilistic model file
    """
    return join(path_data_export, f'probabilistic_data_l{lag}' + ext)


def read_truth_data():
    """
    The read_truth_data function reads the data_truth_nowcasts.pickle file and returns a pandas DataFrame object.
    
    :return: A dataframe with the following columns: 
    :doc-author: Trelent
    """
    # trunk-ignore(bandit/B301)
    return pd.read_pickle(get_path_to_truth(ext='.pickle'))


def read_probabilistic_nowcast_data(lag: int):
    """
    The read_probabilistic_nowcast_data function reads the probabilistic nowcast data from a pickle file.
    If the pickle file does not exist, it creates one and saves it to disk.

    :param lag: int: Specify the lag of the probabilistic nowcast data
    :return: A dataframe with the probabilistic nowcast data
    :doc-author: Trelent
    """
    if isfile(get_prob_path(lag=lag, ext='.pickle')):
        # trunk-ignore(bandit/B301)
        return pd.read_pickle(get_prob_path(lag=lag, ext='.pickle'))
    else:
        print(f'Creating probabilistic nowcast data file for lag {lag}')
        return create_probabilistic_model_for_model_lag(model_list, lag, save_file=True)


def read_nowcast_data(lag: int = 0):
    """
    The read_nowcast_data function reads the nowcast data from a pickle file.
    If the pickle file does not exist, it creates one and saves it to disk.
    The function takes an optional argument lag which is used to specify how many days the lag should comprise.
    
    :param lag: int: Specify the lag of the nowcast data
    :return: A dataframe with all the nowcast data
    :doc-author: Trelent
    """
    if isfile(get_path_to_nowcast(lag=lag, ext='.pickle')):
        # trunk-ignore(bandit/B301)
        return pd.read_pickle(get_path_to_nowcast(lag=lag, ext='.pickle'))
    else:
        print(f'Creating nowcast data file for lag {lag}')
        return create_nowcast_df(lag=lag, save_file=True)


def create_df_for_model_and_lags(model: str, lag_list: list[int], start_date: datetime = None,
                                 end_date: datetime = None) -> pd.DataFrame:
    """
    The create_df_for_model_and_lags function takes a model name and a list of lags as input,
    and returns a dataframe with the following columns:
        - The model's nowcast values for each day in the dataset.
        - The target values for each lag in the lag_list. These are stored as separate columns, one per lag.
        - A column containing the difference between nowcast and target value for each lag.
    
    :param end_date: Specify the end date of the dataframe
    :param start_date: Specify the start date of the dataframe
    :param model: str: Specify which model we want to use (long format, e.g. 'RIVM-KEW')
    :param lag_list: list[int]: Specify the number of days before the target date that we want to use as a predictor
    :return: A dataframe with the model's nowcast and its lags
    :doc-author: Trelent
    """
    df = read_nowcast_data(0).loc[:, [model]]
    df_list = []
    for lag in lag_list:
        df_list.append(read_nowcast_data(lag).loc[:, [model + f'_target_{lag}d_before']])
    df = df.join(df_list)
    for lag in lag_list:
        df.loc[:, model + f'_diff_lag{lag}'] = df.loc[:, model] - df.loc[:, model + f'_target_{lag}d_before']
    df = df.sort_index()
    return df.loc[start_date:end_date, :]


def create_df_truth_lags(lag_list: list[int]) -> pd.DataFrame:
    """
    The create_df_truth_lags function takes a list of integers as input and returns a dataframe.
    The function reads in the truth data, and then adds a difference column for each lag in lag_list (naming: 'diff_lag_{lag}')
    
    :param lag_list: list[int]: Specify the number of days to lag
    :return: A dataframe with the following columns:
    :doc-author: Trelent
    """
    df = read_truth_data()
    df_truth = df.copy()
    for lag in lag_list:
        df.loc[:, f'target_minus_{lag}'] = df.index - pd.Timedelta(days=lag)
        df = df.merge(df_truth, left_on=f'target_minus_{lag}', right_index=True, how='left',
                      suffixes=(None, f'_target_minus_{lag}'))
        df.loc[:, f'diff_lag_{lag}'] = df.loc[:, 'true'] - df.loc[:, f'true_target_minus_{lag}']
    return df.sort_index()


def get_path_to_nowcast(lag: int = 0, ext: str = ''):
    """
    The get_path_to_nowcast function returns the path to a nowcast file.
    
    :param lag: int: Specify the lag of the nowcast
    :param ext: str: Specify the file extension
    :return: The path to the nowcast data
    :doc-author: Trelent
    """
    if lag == 0:
        return join(path_data_export, 'data_nowcasts' + ext)
    else:
        return join(path_data_export, f'data_nowcasts_l{lag}' + ext)


def get_path_to_truth(ext: str = ''):
    """
    The get_path_to_truth function returns the path to the truth data.
    
    :param ext: str: Specify the extension of the file
    :return: The path to the truth data
    :doc-author: Trelent
    """
    return join(path_data_export, 'data_truth' + ext)


def main():
    """
    The main function is the entry point of the script. It uses argparse to parse command line arguments.
    The function supports two arguments: --point and --prob.

    If --point is provided, the function will create the truth data and save it.
    It will also create and save nowcast data for lags ranging from 0 to 10.

    If --prob is provided, the function will create and save probabilistic model data for lags 1, 7, and 14.

    :return: None
    """
    import argparse

    # Initialize ArgumentParser with a description
    parser = argparse.ArgumentParser(description='Read and store forecasting hub data')

    # Add arguments for point-based data and probabilistic data
    parser.add_argument('--point', action='store_true', help='Read and store point-based data')
    parser.add_argument('--prob', action='store_true', help='Read and store probabilistic data')

    # Parse the arguments
    args = parser.parse_args()

    # If the point argument is provided, create and save the truth data and nowcast data for lags 0 to 10
    if args.point:
        create_truth_df(save_file=True)
        for lag in trange(0, 11, desc='Computing and Saving Lag Forecasts...'):
            create_nowcast_df(lag, save_file=True)

    # If the prob argument is provided, create and save probabilistic model data for lags 1, 7, and 14
    if args.prob:
        for lag in [1, 7, 14]:
            create_probabilistic_model_for_model_lag(model_list, lag, save_file=True)


if __name__ == '__main__':
    main()
