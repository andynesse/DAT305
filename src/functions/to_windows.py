import numpy as np
import pandas as pd

from . import magnitude, dict_to_df

WINDOW_SIZE = 40
ACTIVITIES = {0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 0, 5 : 0}

def to_windows(df, window_method, *args, **kwargs):
    result = []
    left = 0
    while left < len(df) - WINDOW_SIZE:
        window_df = df.iloc[left : left + WINDOW_SIZE]
        left += WINDOW_SIZE
        if window_df['activity'].std() != 0.0:
            continue
        result.append(window_method(window_df, *args, **kwargs))
    return result

def window_by_parameters(df, target, predictors):
    """
    Creates a 1 row df, uses all parameters if parameters is empty
    """
    n = len(df)
    window = {target : ACTIVITIES[df[target].iloc[0]]}
    for predictor in predictors:
        for i in range(n):
            window[f"{predictor}_{i}"] = df[predictor].iloc[i]
    return window

def window_by_magnitude(df, target, predictor_groups):
    n = len(df)
    window = {target : ACTIVITIES[df[target].iloc[0]]}
    for predictor_group in predictor_groups:
        t = predictor_group[0]
        x,y,z = predictor_group[1:]
        for i in range(n):
            window[f"mag_{t}_{i}"] = magnitude(df[x].iloc[i], df[y].iloc[i], df[z].iloc[i])
    return window

def window_by_values(df, target, predictor_groups):
    n = len(df)
    window = {target : ACTIVITIES[df[target].iloc[0]]}
    for predictor_group in predictor_groups:
        t = predictor_group[0]
        x,y,z = predictor_group[1:]
        values = []
        for i in range(n):
            values.append(magnitude(df[x].iloc[i], df[y].iloc[i], df[z].iloc[i]))
        values = np.array(values)
        window[f'mean_{t}'] = np.mean(values)
        window[f'std_{t}'] = np.std(values)
        window[f'min_{t}'] = np.min(values)
        window[f'max_{t}'] = np.max(values)
        window[f'q1_{t}'] = np.quantile(values, 0.25)
        window[f'q2_{t}'] = np.quantile(values, 0.5)
        window[f'q3_{t}'] = np.quantile(values, 0.75)

    return window

def dfs_to_windows(dfs, window_method, *args, **kwargs):
    dfs_list = []
    for df in dfs:
        dfs_list.append(dict_to_df(to_windows(df, window_method, *args, **kwargs)))
    return pd.concat(dfs_list, axis=0, ignore_index=True)
