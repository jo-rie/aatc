from os import makedirs
from typing import Union, Literal

import yaml

with open('global_config.yml', 'r') as file:
    config = yaml.safe_load(file)

for c in config.keys():
    if c.endswith('path'):
        makedirs(config[c], exist_ok=True)


def get_atc_ratio_name(lag: Union[str, int] = None, epsilon: Union[float, str] = None, quantile: Union[float, str] = None,
                       pos_neg: Literal['positive', 'negative'] = None) -> str:
    """
    The get_atc_ratio_name function takes a lag and an epsilon as input and returns a string.
    The string is used to name the atc ratio column in the evaluation dataframe.

    :param pos_neg: str: The string 'positive' or 'negative'
    :param lag: int: The lag of the nowcast data
    :param epsilon: float: The epsilon value
    :return: A string with the name of the atc ratio column
    """
    res = r'\mu'
    if pos_neg is not None:
        if lag is not None:
            if pos_neg == 'positive':
                res += r'^{' + f'+, {lag}' + r'}'
            elif pos_neg == 'negative':
                res += r'^{' + f'-, {lag}' + r'}'
            else:
                raise ValueError('pos_neg must be either "positive" or "negative"')
    else:
        if lag is not None:
            res += r'^{' + f'{lag}' + r'}'
    if (epsilon is not None) & (epsilon != 'q_None') & (epsilon != r'q_{None}'):
        res += r'_{' + f'{epsilon}' + r'}'
    else:
        if quantile is not None:
            res += r'_{q_{' + f'{quantile}' + r'}}'
    return '$' + res + '$'
