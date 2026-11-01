import pandas as pd

def dict_to_df(windows_dict):
    columns = windows_dict[0].keys()
    windows = {column: [] for column in columns}

    for window in windows_dict:
        for column in columns:
            windows[column].append(window[column])

    return pd.DataFrame.from_dict(windows)