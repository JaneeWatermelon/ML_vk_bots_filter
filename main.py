import time
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from core.for_ml import before_all, feature_engeneering
from core.feature_engineering import make_new_features
from core.prepare_datasets import merge_datasets, get_both_cols
from core.base import COLS_TO_RENAME, override_tcl_tk, prepare_drop, split_cols, save_dataset
from core.vars import DATASETS_ROOT, OVERRIDE_TCL_TK

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

if OVERRIDE_TCL_TK:
    override_tcl_tk()

drop_cols = [
    "index",
    "_id",
    "id",
    "blacklisted",
    "blacklisted_by_me",

    "education_form",
    "education_status",
    "faculty",
    "faculty_name",
    "graduation",
    "university",
    "university_name",

    "friend_status",
    "is_friend",

    "photo_50",
    "photo_100",
    "photo_200",
    "photo_200_orig",
    "photo_400_orig",
    "photo_max",
    "photo_max_orig",
    "photo_id",

    "online_app",
    "online_mobile",
    "country",
    "education",
    "connections",
    "contacts",
    "exports",
    "is_favorite",
    "is_service",
    "lists",
    "timezone",
    "last_updated_x",
    "last_updated_y",
    "friends_status",
    "downloaded_at",
    "status_audio",
    "names_status",
    "walls_status",
    "can_access_closed",
]

one_hot_cols = [
    "relation",
    "sex",
    # "walls_status",
    "uniqueness",
    "education_level",
]

exists_cols = [
    "about",
    "activities",
    "books",
    "crop_photo",
    "games",
    "home_phone",
    "home_town",
    "interests",
    "mobile_phone",
    "movies",
    "music",
    "occupation",
    "quotes",
    "relatives",
    "site",
    "status",
    "tv",
    "city",
    "skype",
    "relation_partner",
    "first_name",
    "last_name",
    "maiden_name",
    "nickname",
]

complex_cols = [
    "career",
    "military",
    "schools",
    "universities",

    "bdate",
    "last_seen",
    "personal",
    "screen_name",
    "wall_posts_date",
]

zero_fillna_cols = [
    'can_be_invited_group',
    'can_post', 'can_see_all_posts', 'can_see_audio',
    'can_send_friend_request', 'can_write_private_message', 'has_mobile',
    'has_photo', 'is_closed', 'is_hidden_from_feed', 'is_no_index',
    'online', 'trending', 'verified', 'wall_comments', 'friends_count',
    'counters_albums', 'counters_audios', 'counters_followers',
    'counters_gifts', 'counters_groups', 'counters_pages',
    'counters_photos', 'counters_subscriptions', 'counters_user_photos',
    'counters_videos', 'counters_video_playlists',
    'counters_clips_followers', 'counters_clips_views',
    'counters_clips_likes', 'counters_posts', 'counters_articles',
    'counters_clips', 'wall_posts_likes', 'wall_posts_reposts',
    'wall_posts_views', 'activity'
] + one_hot_cols


def prepare_exists(dataset: pd.DataFrame, exists_cols: list):
    def convert(x):
        if (isinstance(x, list) and x == []) or (isinstance(x, str) and len(x) == 0) or x is None:
            return 0
        else:
            return 1
    
    dataset = dataset.copy()
    for col in exists_cols:
        if col in dataset:
            dataset[col] = dataset[col].apply(convert)
    return dataset

def prepare_complex(dataset: pd.DataFrame, complex_cols: list, now_year: int=datetime.datetime.now().year):
    def bdate(x):
        if pd.isna(x) or isinstance(x, str) and len(x) == 0:
            return 0
        # print(f"Type: {type(x)} | value: {x} | x is None: {x is None}")
        nums = x.split(".")
        if len(nums) == 3:
            year = int(nums[-1])
            age = now_year - year
            if age <= 75:
                return 1
            else:
                return -1
        else:
            return 1
        
    def last_seen(x):
        if pd.isna(x) or isinstance(x, dict) and x == {}:
            return 0
        offline_time = time.time() - x["time"]
        if offline_time >= datetime.timedelta(days=31).total_seconds():
            return -1
        else:
            return 1
        
    def wall_posts_date(x):
        # print(f"Type: {type(x)} | value: {x};")
        # time.sleep(1)
        if pd.isna(x) or x == 10**6:
            return 0
        elif x < datetime.timedelta(days=1).total_seconds():
            return -1
        else:
            return 1
        
    def personal(x):
        if isinstance(x, dict):
            return len(x)
        else:
            return 0
        
    def screen_name(x):
        if pd.isna(x) or isinstance(x, str) and len(x) == 0:
            return 0
        elif x.startswith("id") and x[2:].isdigit():
            return 0
        else:
            return 1
        
    for col in complex_cols:
        if col in dataset:
            if col in ["career", "military", "schools", "universities"]:
                dataset[col] = dataset[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
            if col == "bdate":
                dataset[col] = dataset[col].apply(bdate)
            if col == "last_seen":
                dataset[col] = dataset[col].apply(last_seen)
            if col == "personal":
                dataset[col] = dataset[col].apply(personal)
            if col == "screen_name":
                dataset[col] = dataset[col].apply(screen_name)
            if col == "wall_posts_date":
                dataset[col] = dataset[col].apply(wall_posts_date)

    return dataset

def make_fake_bot_col(dataset: pd.DataFrame):
    dataset["is_bot"] = pd.Series(data=np.random.randint(0, 2, dataset.shape[0]))
    return dataset

def prepare_dataset(dataset: pd.DataFrame, now_year: int=datetime.datetime.now().year):
    # dataset = dataset.drop_duplicates()
    dataset = prepare_drop(dataset, columns=drop_cols, reset_index=False)
    dataset = prepare_exists(dataset, exists_cols)
    dataset = prepare_complex(dataset, complex_cols, now_year)
    # dataset = make_fake_bot_col(dataset)

    # not_filled_cols = prepare_drop(dataset, columns=(one_hot_cols + exists_cols + complex_cols), reset_index=False).columns
    # print(not_filled_cols)

    return dataset.copy()

def standart_and_fill_data(dataset: pd.DataFrame):
    numeric_features, categ_features = split_cols(dataset)
    if categ_features.tolist():
        cat_imputer = SimpleImputer(strategy="most_frequent")
        dataset[categ_features] = cat_imputer.fit_transform(dataset[categ_features])

    if numeric_features.tolist():
        num_imputer = SimpleImputer(strategy="median")
        dataset[numeric_features] = num_imputer.fit_transform(dataset[numeric_features])

    for col in categ_features:
        dataset[col] = dataset[col].factorize()[0]

    return dataset

def get_mi_scores(X: pd.DataFrame, y: pd.Series):
    X = standart_and_fill_data(X)
    scores = mutual_info_regression(X, y)
    mi_frame = pd.Series(scores, index=X.columns, name="MI Scores").sort_values(ascending=False)
    
    return mi_frame

def prepare_mi_scores(dataset: pd.DataFrame, mi_frame: pd.Series):
    drop_cols = mi_frame[mi_frame == 0].index.tolist()
    mi_frame = prepare_drop(mi_frame, rows=drop_cols, reset_index=False)
    dataset = prepare_drop(dataset, columns=drop_cols)
    return dataset, mi_frame, drop_cols

def show_mi_scores(X: pd.DataFrame):
    print(X)
    X.sort_values(ascending=True, inplace=True)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.barh(
        X.index,
        width=X.values,
        # kwargs={
        #     "fontsize": 16,
        # }
    )
    plt.show()



def get_pca_scores(X: pd.DataFrame, y: pd.Series, features: list=None):
    if not features:
        features = X.columns
    else:
        X = X[features]
    X = standart_and_fill_data(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=features)
    pca = PCA(random_state=42)

    scores = pca.fit_transform(X)

    cols_pca = [f"PCA_{i+1}" for i in range(scores.shape[1])]

    X_pca = pd.DataFrame(
        scores,
        columns=cols_pca
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=cols_pca,
        index=features,
    )

    print(loadings)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("Explained variance:", explained_variance)
    print("Cumulative variance:", cumulative_variance)

    n_components = np.argmax(cumulative_variance >= 0.8) + 1
    print(f"Рекомендуемое число компонент: {n_components}")

    model = XGBRegressor()

    my_pipline = Pipeline(
        steps=[
            ("model", model),
        ]
    )

    print(X_pca.iloc[:, :n_components])

    scores = -1 * cross_val_score(
        estimator=my_pipline, X=X_pca.iloc[:, :n_components], y=y, cv=5, scoring="neg_mean_squared_error"
    )

    return scores.mean()

def ready_pipeline(X, y, model):
    preprocessing = ColumnTransformer(
        transformers=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0), X.columns.tolist()),
            ("one_hot_encoder", OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist"), list(set(one_hot_cols) & set(X.columns.tolist()))),
        ],
        remainder='passthrough',
    )

    my_pipline = Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    return my_pipline

if __name__ == "__main__":
    # Склеиваем датасеты людей и ботов
    merged_dataset = merge_datasets(
        os.path.join(DATASETS_ROOT, "bot_users.json"),
        os.path.join(DATASETS_ROOT, "human_users.json"),
    )
    # save_dataset(merged_dataset, "for_ml_users.json")
    # save_dataset(merged_dataset, "for_ml_users.xlsx")

    ml_dataset, counter_cols = before_all(merged_dataset)

    print(ml_dataset.columns)
    ml_dataset = make_new_features(ml_dataset, counter_cols)

    # save_dataset(ml_dataset, "for_ml_users_prepared.json")
    # save_dataset(ml_dataset, "for_ml_users_prepared.xlsx")

    main_dataset = pd.read_json(os.path.join(DATASETS_ROOT, "users_no_bots.json"))
    for from_col, to_col in COLS_TO_RENAME.items():
        main_dataset[to_col] = main_dataset[from_col]
    main_dataset = main_dataset.drop(columns=list(COLS_TO_RENAME.keys()))

    intersection_cols = get_both_cols([main_dataset, ml_dataset])
    print(intersection_cols)

    ml_dataset = ml_dataset.dropna(subset=["is_bot"])

    ml_dataset = ml_dataset[intersection_cols + ["is_bot"]]
    main_dataset = main_dataset[intersection_cols]

    ml_dataset = prepare_dataset(ml_dataset, 2025)
    main_dataset = prepare_dataset(main_dataset, 2024)

    # dataset.to_excel("users_droped.xlsx")

    # print(ml_dataset.info())
    # print(ml_dataset.nunique())
    # print(ml_dataset.head())
    # print(ml_dataset.isnull().sum())

    X_ml = ml_dataset
    y_ml = X_ml.pop("is_bot")
    X_main = main_dataset

    mi_scores = get_mi_scores(X_ml, y_ml)
    X_ml, mi_scores, drop_cols = prepare_mi_scores(X_ml, mi_scores)
    X_main = prepare_drop(X_main, columns=drop_cols)
    # show_mi_scores(mi_scores)
    # print(X_ml[mi_scores.index])

    imputer = SimpleImputer(strategy="constant", fill_value=0)

    # Обучаем импьютер на обучающих данных и преобразуем обе выборки
    X_ml_imputed = imputer.fit_transform(X_ml)
    X_main_imputed = imputer.transform(X_main)
    
    # Преобразуем обратно в DataFrame для сохранения названий колонок
    X_ml = pd.DataFrame(X_ml_imputed, columns=X_ml.columns, index=X_ml.index)
    X_main = pd.DataFrame(X_main_imputed, columns=X_main.columns, index=X_main.index)

    models_map = {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=0),
        "XGBR": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=0),
        "RandomForest": RandomForestClassifier(max_leaf_nodes=500, random_state=0),
        "DecisionTree": DecisionTreeClassifier(random_state=0),
        "KNeighbors": KNeighborsClassifier(),
    }

    scores_results = [None, 0]

    for name, model in models_map.items():
        
        done_pipeline = ready_pipeline(X_ml, y_ml, model)

        accuracy_scores = cross_val_score(
            estimator=done_pipeline, X=X_ml, y=y_ml, cv=5, scoring="accuracy", error_score="raise"
        )

        mean_accuracy = accuracy_scores.mean()
        if mean_accuracy > scores_results[1]:
            scores_results = [name, mean_accuracy]

        print(f"Model: {name} | Accuracy = {mean_accuracy}")

    model = models_map[scores_results[0]]
    model.fit(X_ml, y_ml)

    y_pred = model.predict(X_main)
    is_bot_col = pd.Series(data=y_pred)

    main_dataset = pd.read_json(os.path.join(DATASETS_ROOT, "users_no_bots.json"))
    main_dataset["is_bot"] = is_bot_col

    # main_dataset.to_json("users_with_bots.json")
    # main_dataset.to_excel("users_with_bots.xlsx")
