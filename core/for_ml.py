import os
import pandas as pd

from core.base import load_datasets, prepare_drop, prepare_complex_cols, override_tcl_tk
from core.visualisation import visualize_bots_and_peoples, visualize_gender_education
from core.analysis import get_flews_ids, get_bots
from core.vars import DATASETS_ROOT, OVERRIDE_TCL_TK
from core.feature_engineering import make_full_uniqueness_column, make_uniqueness_column, make_education_column, make_content_activity_column

if OVERRIDE_TCL_TK:
    override_tcl_tk()

def before_all(dataset: pd.DataFrame):
    dataset, counter_cols = prepare_complex_cols(dataset, to="json", columns=["counters"])
    dataset = prepare_drop(dataset, columns=["followers_count"])

    counter_cols = list(counter_cols)
    print(counter_cols)
    print(dataset.columns)
    # counter_cols.remove("wall_posts_date")

    return dataset, counter_cols

def feature_engeneering(dataset: pd.DataFrame, counter_cols: pd.Index):
    activity_features_exists = [
        "about",
        "activities",
        "books",
        "crop_photo",
        "home_phone",
        "home_town",
        "interests",
        # "mobile_phone",
        "movies",
        "music",

        "personal",

        "status",
        "city",
    ]
    activity_features_nums = [
        "counters_friends",
    ] + list(counter_cols)

    uniqueness_col = make_full_uniqueness_column(dataset, ["counters_friends"] + counter_cols)
    act_col = make_content_activity_column(dataset, activity_features_nums)
    education_col = make_education_column(dataset)

    dataset = pd.concat([dataset, uniqueness_col, act_col, education_col], axis=1)

    return dataset

