import pandas as pd
import numpy as np

from core.base import get_low_disp_features, prepare_drop, prepare_null, save_dataset
from core.feature_engineering import FeaturesModifier, FeaturesMaderBase, FeaturesMaderCustom, make_new_features, replace_some_features
from core.vars import DATASETS_ROOT, FeatureGroups

def merge_datasets(bots_url: str, humans_url: str, is_bot_col: str="is_bot"):
    bots_dataset = pd.read_json(bots_url)
    bots_dataset[is_bot_col] = pd.Series(data=np.ones(bots_dataset.shape[0]))
    humans_dataset = pd.read_json(humans_url)
    humans_dataset[is_bot_col] = pd.Series(data=np.zeros(bots_dataset.shape[0]))

    result_dataset = pd.concat([bots_dataset, humans_dataset], axis=0, ignore_index=True)

    return result_dataset

def prepare_complex(dataset: pd.DataFrame, features: list):
    for col in features:
        if col in dataset.columns:
            if col in ["career", "military", "schools", "universities", "relatives"]:
                dataset[col] = FeaturesMaderBase.make_count_features(dataset, features=[col])
            if col == "last_seen":
                dataset[col] = FeaturesModifier.modify_last_seen_column(dataset, name=col)
            if col == "personal":
                dataset[col] = FeaturesMaderBase.make_count_features(dataset, features=[col])
            if col == "screen_name":
                dataset[col] = FeaturesModifier.modify_screen_name_column(dataset, name=col)
            if col == "bdate":
                dataset[col] = FeaturesModifier.modify_bdate_column(dataset, name=col)
            if col == "occupation":
                dataset[col] = FeaturesModifier.modify_occupation_column(dataset, name=col)
            if col == "site":
                dataset[col] = FeaturesModifier.modify_site_column(dataset, name=col)
            if col == "home_phone":
                dataset[col] = FeaturesModifier.modify_phone_column(dataset, name=col)
            if col == "relation":
                dataset[col] = FeaturesMaderBase.make_exists_features(dataset, features=[col], num_false=0)

    return dataset

def prepare_exists(dataset: pd.DataFrame, features: list):
    dataset[features] = FeaturesMaderBase.make_exists_features(dataset, features=features).astype(int)

    return dataset

def prepare_dataset(dataset: pd.DataFrame):
    dataset = dataset.copy()
    dataset = dataset.dropna(subset=["is_bot"])
    dataset = prepare_drop(dataset, columns=FeatureGroups.DROP_COLS.value)

    dataset = FeaturesMaderCustom.make_counters_features(dataset)
    dataset = FeaturesMaderCustom.make_personal_features(dataset)

    dataset, new_features_cols, new_drop_cols = make_new_features(dataset)
    dataset = prepare_complex(dataset, FeatureGroups.COMPLEX_COLS.value)
    dataset = prepare_exists(dataset, FeatureGroups.EXISTS_COLS.value)
    
    dataset, unwanted_cols = replace_some_features(dataset)

    low_disp_cols = get_low_disp_features(dataset, 0.98).columns.tolist()
    print(low_disp_cols)

    drop_cols = new_drop_cols + unwanted_cols + low_disp_cols

    dataset = prepare_drop(dataset, columns=drop_cols)
    dataset = prepare_null(dataset)
    
    save_dataset(dataset, file_name="prepared_users.json")
    save_dataset(dataset, file_name="prepared_users.xlsx")

    return dataset, new_features_cols

