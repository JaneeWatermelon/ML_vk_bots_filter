import os
import pandas as pd
import numpy as np

from core.vars import DATASETS_ROOT

def merge_datasets(bots_url: str, humans_url: str, is_bot_col: str="is_bot"):
    bots_dataset = pd.read_json(bots_url)
    bots_dataset[is_bot_col] = pd.Series(data=np.ones(bots_dataset.shape[0]))
    humans_dataset = pd.read_json(humans_url)
    humans_dataset[is_bot_col] = pd.Series(data=np.zeros(bots_dataset.shape[0]))

    result_dataset = pd.concat([bots_dataset, humans_dataset], axis=0, ignore_index=True)

    return result_dataset

def get_both_cols(datasets: list[pd.DataFrame]):
    intersection_cols = set(datasets[0].columns.tolist())

    for i in range(len(datasets)):
        intersection_cols = intersection_cols & set(datasets[i].columns.tolist())

    return list(intersection_cols)


