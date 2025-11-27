import datetime
import time
import numpy as np
import pandas as pd

from core.vars import NOW_YEAR, ExplainVars, NewFeaturesNames, BulkFeatures
from core.analysis import get_flews_ids
from core.base import extract_features_by_start, prepare_complex_cols, prepare_drop, prepare_null

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
        is_missing = pd.isna(x)
        if not is_missing and x != num_false:
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

class FeaturesMaderBase:
    @staticmethod
    def make_mean_column(dataset: pd.DataFrame, features: pd.Index, name: str="privacy_score") -> pd.Series:
        result_col = (dataset[features].sum(axis=1) / len(features)).apply(_round_col)
        result_col.name = name

        return result_col
    
    @staticmethod
    def make_count_features(dataset: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
        for col in features:
            if col in dataset.columns:
                dataset[col] = dataset[col].apply(_value_count)
                dataset[col].name = col

        return dataset[features]

    @staticmethod
    def make_exists_features(dataset: pd.DataFrame, features: pd.Index, num_false: int = None, str_false: str = None) -> pd.DataFrame:
        for col in features:
            if col in dataset.columns:
                dataset[col] = dataset[col].apply(lambda x: _is_value_exists(x, num_false, str_false))
                dataset[col].name = col

        return dataset[features]

class FeaturesMaderCustom:
    @staticmethod
    def make_fullness_column(dataset: pd.DataFrame, exists_features: pd.Index, name: str=NewFeaturesNames.FULLNESS.value) -> pd.Series:
        fullness_col = dataset[exists_features].map(lambda x: _is_value_exists(x, num_false=0)).sum(axis=1)
        fullness_col = fullness_col / len(exists_features)
        fullness_col = fullness_col.apply(_round_col)
        fullness_col.name = name

        return fullness_col

    @staticmethod
    def make_content_activity_column(dataset: pd.DataFrame, nums_features: pd.Index, name: str=NewFeaturesNames.CONTENT_ACTIVITY.value) -> pd.Series:
        activity_col = _nums_activity(dataset, nums_features)
        activity_col.name = name

        return activity_col

    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
    def make_clips_popularity_column(dataset: pd.DataFrame, name: str="clips_popularity") -> pd.Series:
        dataset = dataset.copy()
        used_cols = ["counters_clips_views", "counters_clips_likes", "counters_clips"]
            
        dataset[name] = (dataset["counters_clips_views"] / 10 + dataset["counters_clips_likes"])
        dataset[name] = dataset[name] / dataset["counters_clips"].apply(lambda x: max(np.log(x), 1))
        dataset[name] = dataset[name].round(2)
        
        return dataset[name], used_cols

    @staticmethod
    def make_counters_media_storage_column(dataset: pd.DataFrame, name: str="counters_media_storage") -> pd.Series:
        dataset = dataset.copy()
        used_cols = ["counters_video_playlists", "counters_albums"]
            
        dataset[name] = dataset[used_cols].sum(axis=1)
        
        return dataset[name], used_cols

    @staticmethod
    def make_is_ghost_column(dataset: pd.DataFrame, name: str="is_ghost") -> pd.Series:
        dataset = dataset.copy()
        used_cols = ["last_seen", "has_photo", "has_mobile", "deactivated"]
            
        deactivated = dataset[["has_mobile", "deactivated"]].max(axis=1)
        available = dataset[["last_seen", "has_photo"]].max(axis=1).apply(lambda x: x == 0)
        dataset[name] = pd.concat([deactivated, available], axis=1).max(axis=1).astype(int)
        
        return dataset[name], used_cols

    @staticmethod
    def make_photos_posts_ratio_column(dataset: pd.DataFrame, name: str="photos_posts_ratio") -> pd.Series:
        dataset = dataset.copy()
        used_cols = ["counters_posts", "counters_photos"]
            
        dataset[used_cols] = dataset[used_cols].map(lambda x: max(x, 1))
        dataset[name] = (dataset["counters_photos"] / dataset["counters_posts"]).apply(lambda x: max(np.log(x), 1)).round(2)
        
        return dataset[name], used_cols

    @staticmethod
    def make_counters_features(dataset: pd.DataFrame, name: str="counters") -> pd.DataFrame:
        dataset = dataset.copy()
        dataset, counter_cols = prepare_complex_cols(dataset, to="json", columns=[name])
        dataset = prepare_drop(dataset, columns=[name + "_mutual_friends"])

        return dataset

    @staticmethod
    def make_personal_features(dataset: pd.DataFrame, name: str="personal") -> pd.DataFrame:
        dataset = dataset.copy()
        dataset, personal_cols = prepare_complex_cols(dataset, to="json", columns=[name])
        dataset = prepare_drop(dataset, columns=["personal_religion_id", "personal_langs_full"])
        personal_cols = extract_features_by_start(dataset, start="personal_")
        dataset[personal_cols] = prepare_null(dataset[personal_cols])

        personal_count_cols = ["personal_langs"]
        dataset[personal_count_cols] = FeaturesMaderBase.make_count_features(dataset, features=personal_count_cols).astype(int)

        personal_exists_cols = prepare_drop(dataset[personal_cols], columns=personal_count_cols).columns.tolist()
        dataset[personal_exists_cols] = FeaturesMaderBase.make_exists_features(dataset, features=personal_exists_cols, num_false=0).astype(int)
        dataset["personal_fullness"] = (dataset[personal_exists_cols].sum(axis=1) / len(personal_exists_cols)).round(2)
        dataset = prepare_drop(dataset, columns=personal_exists_cols)

        return dataset

    @staticmethod
    def make_uniqueness_column(dataset: pd.DataFrame, features: pd.Index, name: str="uniqueness", max_clusters: int=5) -> pd.Series:
        uniqueness_col = pd.Series(np.zeros(dataset.shape[0]), name=name)

        flews_ids = dataset.index.tolist()
        
        for i in range(max_clusters-1):
            flews_ids, flews_dataset_info = get_flews_ids(dataset.iloc[flews_ids], features)
            uniqueness_col.iloc[flews_ids] = i + 1
            
        uniqueness_col.name = name

        return uniqueness_col

    @staticmethod
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
        
class FeaturesModifier:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
    def modify_occupation_column(dataset: pd.DataFrame, name: str="occupation"):
        def checker(x):
            if x != None and isinstance(x, dict):
                x = x.get("type", ExplainVars.UNKNOWN.value)
                return x
                
            return ExplainVars.UNKNOWN.value
        
        dataset[name] = dataset[name].apply(checker)

        return dataset[name]
    
    @staticmethod
    def modify_screen_name_column(dataset: pd.DataFrame, name: str="screen_name"):
        def checker(x):
            if pd.isna(x) or isinstance(x, str) and len(x) == 0:
                return 0
            elif x.startswith("id") and x[2:].isdigit():
                return 0
            else:
                return 1
        
        dataset[name] = dataset[name].apply(checker)

        return dataset[name]
    
    @staticmethod
    def modify_last_seen_column(dataset: pd.DataFrame, name: str="last_seen") -> pd.Series:
        def pre_last_seen(x):
            if pd.isna(x) or isinstance(x, dict) and x == {}:
                return 0
            return x["time"]
        
        def post_last_seen(x, max_seconds):
            if pd.isna(x) or isinstance(x, int) and x == 0:
                return 0
            offline_time = max_seconds - x
            if offline_time >= datetime.timedelta(days=31).total_seconds():
                return 0
            else:
                return 1
        
        dataset[name] = dataset[name].apply(pre_last_seen)
        max_seconds = dataset[name].max()
        dataset[name] = dataset[name].apply(lambda x: post_last_seen(x, max_seconds))
        
        return dataset[name]


def make_new_features(dataset: pd.DataFrame) -> pd.DataFrame:
    counter_cols = extract_features_by_start(dataset, start="counters_")

    uniqueness_col = FeaturesMaderCustom.make_uniqueness_column(dataset, features=counter_cols)
    age_col = FeaturesMaderCustom.make_age_column(dataset, now_year=NOW_YEAR)
    fullness_col = FeaturesMaderCustom.make_fullness_column(dataset, list(set(BulkFeatures.ACTIVITY_INFOS.value + BulkFeatures.PERSONAL_INFOS.value)))
    content_activity_col = FeaturesMaderCustom.make_content_activity_column(dataset, counter_cols)
    privacy_score_col = FeaturesMaderBase.make_mean_column(dataset, BulkFeatures.PRIVACY_SCORE.value, name="privacy_score")
    communication_accessibility_col = FeaturesMaderBase.make_mean_column(dataset, BulkFeatures.COMMUNICATION_ACCESSIBILITY.value, name="communication_accessibility")
    education_col = FeaturesMaderCustom.make_education_column(dataset)

    drop_cols = BulkFeatures.PRIVACY_SCORE.value + BulkFeatures.ACTIVITY_INFOS.value + BulkFeatures.PERSONAL_INFOS.value

    new_features_df = pd.concat([
        uniqueness_col, 
        age_col, 
        fullness_col, 
        content_activity_col, 
        privacy_score_col, 
        communication_accessibility_col, 
        education_col
    ], axis=1)

    dataset = pd.concat([
        dataset, 
        new_features_df
    ], axis=1)

    return dataset, new_features_df.columns.tolist(), drop_cols

def replace_some_features(dataset: pd.DataFrame) -> pd.DataFrame:
    unwanted_cols = []
    if "home_phone" in dataset.columns:
        if dataset["home_phone"].dtype == "object":
            dataset["home_phone"] = FeaturesMaderBase.make_exists_features(dataset, features=["home_phone"])

        dataset["has_mobile"] = dataset[["has_mobile", "home_phone"]].max(axis=1)
        unwanted_cols.append("home_phone")

    if "site" in dataset.columns:
        if dataset["site"].dtype == "object":
            dataset["site"] = FeaturesMaderBase.make_exists_features(dataset, features=["site"])

    for categ_col in dataset.select_dtypes("object"):
        print(categ_col)
        dataset[categ_col] = dataset[categ_col].factorize()[0]

    return dataset, unwanted_cols
