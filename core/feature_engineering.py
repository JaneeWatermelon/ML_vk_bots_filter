import datetime
import time
import numpy as np
import pandas as pd

from core.vars import NOW_YEAR, NewFeaturesNames, BulkFeatures
from core.analysis import get_flews_ids
from core.base import prepare_drop

def _round_col(x: int, nums_after_limit: int=2):
    return round(x, nums_after_limit)

def _is_value_exists(x, num_false: int = None, str_false: str = None):
    if x == None:
        return False
    if isinstance(x, int) or isinstance(x, float):
        is_missing = pd.isna(x)
        
        if not is_missing and x != num_false:
            return True
        else:
            return False
    if isinstance(x, str):
        if x.strip() != "" and x != str_false:
            return True
        else:
            return False
    if isinstance(x, list) or isinstance(x, dict):
        if len(x) != 0:
            return True
        else:
            return False
    return super(str(x))

def _value_count(x, num_false: int = None, str_false: str = None):
    if x == None:
        return 0
    if isinstance(x, int) or isinstance(x, float):
        if x != np.nan and x != num_false:
            return 1
        else:
            return 0
    if isinstance(x, str):
        if x.strip() != "" and x != str_false:
            return 1
        else:
            return 0
    if isinstance(x, list) or isinstance(x, dict):
        return len(x)
    return super(str(x))

def _nums_activity(dataset: pd.DataFrame, nums_features: pd.Index):
    activity_col = pd.Series(data=np.zeros(dataset.shape[0]))

    dataset_nums = dataset[nums_features].fillna(0)
    median_df = dataset_nums.median().apply(lambda x: max(x, 1))
    median_df.index = nums_features

    for col in nums_features:
        try:
            add_col = (dataset_nums[col] / median_df[col]).apply(lambda x: max(x, 1)).apply(np.log)
            activity_col += add_col
        except Exception as e:
            print(e)
            print(col)
            print(dataset_nums[col])
            continue
    activity_col = activity_col.apply(_round_col)

    return activity_col

def make_fullness_column(dataset: pd.DataFrame, exists_features: pd.Index, name: str=NewFeaturesNames.FULLNESS.value) -> pd.Series:
    fullness_col = dataset[exists_features].map(lambda x: _is_value_exists(x, num_false=0)).sum(axis=1)
    fullness_col = fullness_col / len(exists_features)
    fullness_col = fullness_col.apply(_round_col)
    fullness_col.name = name

    return fullness_col

def make_content_activity_column(dataset: pd.DataFrame, nums_features: pd.Index, name: str=NewFeaturesNames.CONTENT_ACTIVITY.value) -> pd.Series:
    activity_col = _nums_activity(dataset, nums_features)
    activity_col.name = name

    return activity_col

def make_mean_column(dataset: pd.DataFrame, can_features: pd.Index, name: str="privacy_score") -> pd.Series:
    result_col = (dataset[can_features].sum(axis=1) / len(can_features)).apply(_round_col)
    result_col.name = name

    return result_col

def make_education_column(dataset: pd.DataFrame, name: str="education_level") -> pd.Series:
    def check_schools(x):
        if x == None or x == []:
            return 0
        else:
            return 1
        
    def check_unis(x):
        if isinstance(x, list):
            return len(x)
        else:
            return 0
        
    def check_education(x):
        if x["universities"] > 0:
            if x["universities"] > 1:
                return 3
            else:
                return 2
        elif x["schools"] == 1:
            return 1
        else:
            return 0
        
    schools_exists = dataset["schools"].apply(check_schools)
    universities_count = dataset["universities"].apply(check_unis)
        
    education_col = pd.concat([schools_exists, universities_count], axis=1)
    education_col.columns = ["schools", "universities"]
    education_col = education_col.apply(check_education, axis=1)
    education_col.name = name
    return education_col

def make_relation_column(dataset: pd.DataFrame, name: str="relation") -> pd.Series:
    in_relation = [2, 3, 8]
    idk_relation = [5, 7]
    def change_relation(x):
        if pd.isna(x):
            return 0
        elif x in in_relation:
            return 10
        elif x in idk_relation:
            return 11
        else:
            return x
        
    education_col = dataset[name].apply(change_relation)
    education_col.name = name
    return education_col

def make_uniqueness_column(dataset: pd.DataFrame, clusters_list: list[list[int]], name: str="uniqueness") -> pd.Series:
    data = np.zeros(dataset.shape[0])

    for i in range(len(clusters_list)):
        data[clusters_list[i]] = i

    uniqueness_col = pd.Series(data, name=name)
    return uniqueness_col
def make_full_uniqueness_column(dataset: pd.DataFrame, flews_cols: pd.Index, name: str="uniqueness", max_clusters: int=5, verbose: bool=True) -> pd.Series:
    flews_ids = dataset.index.tolist()
    clusters_list = [flews_ids]
    
    for i in range(max_clusters-1):
        flews_ids, flews_dataset_info = get_flews_ids(dataset.iloc[flews_ids], flews_cols)
        clusters_list.append(flews_ids)

        if verbose:
            print(f"Найдено аномалий на {i+1} шаге: {len(flews_ids)}")
            print(flews_dataset_info)
            print()

    # Кластеризация аномалий
    uniqueness_col = make_uniqueness_column(dataset, clusters_list, name)
    uniqueness_col.name = name

    return uniqueness_col

def make_age_column(dataset: pd.DataFrame, name: str="age", now_year: int=datetime.datetime.now().year) -> pd.Series:
    def convert_to_age(x):
        if not isinstance(x, str):
            return 0
        parts = x.strip().split('.')
        if len(parts) == 3:
            try:
                year = int(parts[-1])
                return now_year - year
            except ValueError:
                return 0
        return 0
    
    result_col = dataset["bdate"].apply(convert_to_age)
    result_col.name = name
    
    return result_col

def make_choice_fields_ratio_column(dataset: pd.DataFrame, name: str="choice_fields_ratio") -> pd.Series:
    def convert_dict(x):
        if x == None:
            return 0
        elif isinstance(x, dict):
            if "langs_full" in x and "langs" in x:
                return len(x) - 1
            else:
                return len(x)
        else:
            return 0
    
    openness_col = dataset["personal"].apply(convert_dict)
    openness_col += dataset["city"].apply(convert_dict)

    openness_col = (openness_col / openness_col.max()).apply(_round_col)

    openness_col.name = name
    
    return openness_col

def make_count_features(dataset: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    for col in features:
        if col in dataset.columns:
            dataset[col] = dataset[col].apply(_value_count)
            dataset[col].name = col

    return dataset[features]
        
def modify_site_column(dataset: pd.DataFrame, name: str="site"):
    exist_values = [
        "http://",
        "https://",
        "@",
        "www",
        ".ru",
        ".ру",
        ".com",
        ".net",
        "gmail.",
        "mail.",
    ]

    def checker(x):
        if x != None and isinstance(x, str):
            x = x.strip()
            if len(x) > 1:
                for inner in exist_values:
                    if inner in x:
                        return x
                return ""
            else:
                return ""
        return ""
    
    dataset[name] = dataset[name].apply(checker)

    return dataset[name]

def modify_phone_column(dataset: pd.DataFrame, name: str="home_phone"):
    def checker(x):
        if x != None and isinstance(x, str):
            x = x.strip()
            if 9 <= len(x) <= 17:
                digits_count = 0
                for ch in x:
                    if ch.isalpha():
                        return ""
                    digits_count += 1 if ch.isdigit() else 0
                if digits_count > 12 or digits_count < 3:
                    return ""
                return x
            else:
                return ""
        return ""
    
    dataset[name] = dataset[name].apply(checker)

    return dataset[name]

def modify_bdate_column(dataset: pd.DataFrame, name: str="bdate"):
    def checker(x):
        if x != None and isinstance(x, str):
            x = x.strip()
            if x == "":
                return 0
            parts = x.split('.')
            if len(parts) == 3:
                return 2
            return 1
            
        return 0
    
    dataset[name] = dataset[name].apply(checker)

    return dataset[name]

def make_exists_column(dataset: pd.DataFrame, column: str, num_false: int = None, str_false: str = None, name: str="exists") -> pd.Series:
    result_col = dataset[column].apply(lambda x: _is_value_exists(x, num_false, str_false))
    result_col.name = name

    return result_col

def make_exists_features(dataset: pd.DataFrame, features: pd.Index, num_false: int = None, str_false: str = None) -> pd.DataFrame:
    for col in features:
        if col in dataset.columns:
            dataset[col] = dataset[col].apply(lambda x: _is_value_exists(x, num_false, str_false))
            dataset[col].name = col

    return dataset[features]

def make_new_features(dataset: pd.DataFrame, counter_cols: pd.Index) -> pd.DataFrame:
    uniqueness_col = make_full_uniqueness_column(dataset, counter_cols)
    age_col = make_age_column(dataset, now_year=NOW_YEAR)
    fullness_col = make_fullness_column(dataset, list(set(BulkFeatures.ACTIVITY_INFOS.value + BulkFeatures.PERSONAL_INFOS.value)))
    content_activity_col = make_content_activity_column(dataset, counter_cols)
    # content_activity_new_col = make_content_activity_column(dataset, content_activity_new_features_nums)
    privacy_score_col = make_mean_column(dataset, BulkFeatures.PRIVACY_SCORE.value, name="privacy_score")
    communication_accessibility_col = make_mean_column(dataset, BulkFeatures.COMMUNICATION_ACCESSIBILITY.value, name="communication_accessibility")
    education_col = make_education_column(dataset)

    dataset["relation"] = make_relation_column(dataset)

    dataset = pd.concat([
        dataset, 
        uniqueness_col, 
        age_col, 
        fullness_col, 
        content_activity_col, 
        # content_activity_new_col, 
        privacy_score_col, 
        communication_accessibility_col, 
        education_col
    ], axis=1)

    return dataset

def replace_some_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    1. Признаки has_mobile, home_phone, mobile_phone были объединены в один has_mobile по логической операции ИЛИ

    """
    features_to_exists = [
        "activities",
        "books",
        "games",
        "quotes",
        "tv",
        "interests",
        "movies",
        "music",

        "about",
        "status",
        "crop_photo",
        "home_town",
        "city",

        "nickname",

    ]
    features_to_modify = [
        "site",
        "home_phone",
        "mobile_phone",
    ]
    features_complex = [
        "personal",
        "relatives",
        "schools",
        "universities",
        "career",
    ]

    dataset["friends_status"] = dataset["friends_status"].map({"This profile is private": 0, "success": 1})
    dataset["bdate"] = modify_bdate_column(dataset, name="bdate")
    dataset["walls_status"] = (make_exists_column(dataset, "walls_status") == False).astype(int)

    dataset["site"] = modify_site_column(dataset, name="site")
    dataset["home_phone"] = modify_phone_column(dataset, name="home_phone")
    dataset["mobile_phone"] = modify_phone_column(dataset, name="mobile_phone")
    dataset[features_to_modify] = make_exists_features(dataset, features=features_to_modify).astype(int)

    dataset[features_to_exists] = make_exists_features(dataset, features=features_to_exists).astype(int)
    dataset[features_complex] = make_count_features(dataset, features=features_complex)

    dataset["has_mobile"] = dataset[["has_mobile", "home_phone", "mobile_phone"]].max(axis=1)
    dataset = prepare_drop(dataset, columns=["home_phone", "mobile_phone"])


    return dataset
