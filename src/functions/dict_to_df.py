import pandas as pd
from src.functions import to_windows

def dict_to_df(data, method):
    windows_method = to_windows(data[0], method, 'activity', [("acc", 'aX', 'aY', 'aZ'), ("gyro", 'gX', 'gY', 'gZ')])

    columns = windows_method[0].keys()
    windows_dict = {column: [] for column in columns}

    for window in windows_method:
        for column in columns:
            windows_dict[column].append(window[column])

    return pd.DataFrame.from_dict(windows_dict)