import time
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.feature_engineering import FeaturesMaderCustom
from core.prepare_datasets import merge_datasets, prepare_dataset
from core.base import get_high_corr_pairs, override_tcl_tk, prepare_complex_cols, prepare_drop, prepare_null, split_cols, save_dataset
from core.vars import DATASETS_ROOT, OVERRIDE_TCL_TK, FeatureGroups

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from core.visualisation import custom_heatmap
from core.default_visual import set_visual_default

def some_infos(dataset: pd.DataFrame):
    # Размерность
    print(dataset.shape)
    print()

    # Типы признаков
    print(dataset.iloc[:, :70].info())
    print()

    # Доля пропусков
    missing_ratio = dataset.isnull().mean() * 100
    print("Доля пропусков по признакам (%):")
    print(missing_ratio.sort_values(ascending=False))
    print()

    # Дубликаты
    stringed_users, stringed_cols = prepare_complex_cols(dataset, to="string")
    dups = stringed_users.duplicated()
    print(f"Кол-во дубликатов: {dups.sum()}")
    print()

    return dataset

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

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset, columns = numeric_features.tolist() + categ_features.tolist())

    return dataset

class MutalInfo:
    @staticmethod
    def get_mi_scores(X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        X = standart_and_fill_data(X)
        scores = mutual_info_classif(X, y, random_state=0)
        mi_frame = pd.Series(scores, index=X.columns, name="MI Scores").sort_values(ascending=False)
        
        return mi_frame

    @staticmethod
    def prepare_mi_scores(dataset: pd.DataFrame, mi_frame: pd.Series):
        drop_cols = mi_frame[mi_frame == 0].index.tolist()
        mi_frame = prepare_drop(mi_frame, rows=drop_cols, reset_index=False)
        dataset = prepare_drop(dataset, columns=drop_cols)
        return dataset, mi_frame, drop_cols

    @staticmethod
    def show_mi_scores(X: pd.DataFrame):
        set_visual_default()
        X.sort_values(ascending=True, inplace=True)
        plt.barh(
            X.index,
            width=X.values,
        )
        plt.show()

    @staticmethod
    def analysis_mi_scores(X: pd.DataFrame, y: pd.Series, drop_zeros_info: bool = False):
        mi_scores = MutalInfo.get_mi_scores(X, y)
        drop_cols = []
        if drop_zeros_info:
            X, mi_scores, drop_cols = MutalInfo.prepare_mi_scores(X, mi_scores)
        MutalInfo.show_mi_scores(mi_scores)

        print(X[mi_scores.index])

        return X, y, drop_cols

class PrincipalComponentAnalysis:
    @staticmethod
    def get_pca_scores(X: pd.DataFrame, features: list=None):
        if not features:
            features = X.columns
        else:
            X = X[features]
        X = standart_and_fill_data(X)
        
        pca = PCA(random_state=42)

        scores = pca.fit_transform(X)

        cols_pca = [f"PCA_{i+1}" for i in range(scores.shape[1])]

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=cols_pca,
            index=features,
        )

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        n_components = np.argmax(cumulative_variance >= 0.8) + 1
        print(f"Рекомендуемое число компонент: {n_components}")

        return loadings.iloc[:, :n_components]
    
    @staticmethod
    def make_full_analysis(dataset: pd.DataFrame, features: pd.Index=None):
        if features == None:
            features = dataset.columns

        pca_loadings = PrincipalComponentAnalysis.get_pca_scores(
            prepare_drop(
                dataset=dataset, 
                columns=features
            )
        )
        custom_heatmap(
            pca_loadings,
            title="Нагрузки главных компонент",
            cmap="coolwarm",
        )

def ready_pipeline(X: pd.DataFrame, y: pd.Series, model):
    X_nunique = X.convert_dtypes().select_dtypes(exclude="float").nunique().sort_values(ascending=True)
    max_values = 6

    one_hot_cols = set([col for col in X_nunique.index if X_nunique.loc[col] <= max_values])
    # Константа 6 была выбрана исходя из кол-ва категорий в признаке relation
    # print(one_hot_cols)

    preprocessing = ColumnTransformer(
        transformers=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0), X.columns.tolist()),
            ("one_hot_encoder", OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist"), list(set(one_hot_cols))),
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
    if OVERRIDE_TCL_TK:
        override_tcl_tk()

    np.random.seed(42)

    # Склеиваем датасеты людей и ботов
    ml_dataset = merge_datasets(
        os.path.join(DATASETS_ROOT, "bot_users.json"),
        os.path.join(DATASETS_ROOT, "human_users.json"),
    )

    ml_dataset, new_features_cols = prepare_dataset(ml_dataset)

    some_infos(ml_dataset)

    X_ml = ml_dataset
    y_ml = X_ml.pop("is_bot")

    # Анализируем матрицу корреляции признаков
    X_corr = prepare_drop(dataset=X_ml, columns=new_features_cols).corr()

    # Получаем пары признаков с корреляцией >= указанного значения
    high_corr_pairs = get_high_corr_pairs(X_corr, alpha=0.8, verbose=True)

    # Делаем выводы на основе полученных пар и удаляем "похожие" признаки
    corr_drop_cols = ["followers_count", "photo_id"]
    X_ml = prepare_drop(X_ml, columns=corr_drop_cols)


    # Проводим анализ главных компонент
    PrincipalComponentAnalysis.make_full_analysis(
        X_ml, 
        features=new_features_cols + FeatureGroups.CATEGORIAL_COLS.value
    )

    X_ml["clips_popularity"], used_cols = FeaturesMaderCustom.make_clips_popularity_column(X_ml)
    X_ml = prepare_drop(X_ml, used_cols)
    X_ml["counters_media_storage"], used_cols = FeaturesMaderCustom.make_counters_media_storage_column(X_ml)
    X_ml = prepare_drop(X_ml, used_cols)
    X_ml["is_ghost"], used_cols = FeaturesMaderCustom.make_is_ghost_column(X_ml)
    X_ml = prepare_drop(X_ml, used_cols)
    X_ml["photos_posts_ratio"], used_cols = FeaturesMaderCustom.make_photos_posts_ratio_column(X_ml)
    X_ml = prepare_drop(X_ml, used_cols)

    # Повторно проводим анализ главных компонент, после обработки новых признаков
    PrincipalComponentAnalysis.make_full_analysis(
        X_ml, 
        features=new_features_cols + ["clips_popularity", "counters_media_storage", "is_ghost", "photos_posts_ratio"] + FeatureGroups.CATEGORIAL_COLS.value
    )

    # Проводим анализ важной информации в признаках по отношению к признаку is_bot
    _, _, _ = MutalInfo.analysis_mi_scores(X_ml, y_ml, drop_zeros_info=False)

    X_ml, y_ml, droped_cols = MutalInfo.analysis_mi_scores(X_ml, y_ml, drop_zeros_info=True)

    print(droped_cols)

    X_ml = prepare_null(X_ml)

    save_dataset(X_ml, file_name="X_ml_proceeded.json")
    save_dataset(X_ml, file_name="X_ml_proceeded.xlsx")

    models_map = {
        "GradientBoosting": (GradientBoostingClassifier(),
            {
                "model__n_estimators": range(100, 500+1, 100),
                "model__random_state": [0],
                "model__max_depth": [None, 5, 10],
            }
        ),
        "XGBC": (XGBClassifier(),
            {
                "model__n_estimators": range(100, 500+1, 100),
                "model__random_state": [0],
                "model__learning_rate": [None, 0.05, 0.1],
                "model__max_depth": [None, 5, 10],
            }
        ),
        "RandomForest": (RandomForestClassifier(),
            {
                "model__n_estimators": range(100, 500+1, 100),
                "model__max_leaf_nodes": [None] + list(range(50, 200+1, 50)),
                "model__random_state": [0],
                "model__max_depth": [None, 5, 10],
            }
        ),
        "DecisionTree": (DecisionTreeClassifier(),
            {
                "model__max_leaf_nodes": [None] + list(range(50, 250+1, 50)),
                "model__random_state": [0],
                "model__max_depth": [None, 5, 10],
            }
        ),
        "KNeighbors": (KNeighborsClassifier(),
            {
                "model__n_neighbors": [2, 5, 10],
                "model__p": [2, 5, 10],
            }
        ),
    }

    scores_results = [None, 0]

    for name, data in models_map.items():
        model, param_grid = data
        
        done_pipeline = ready_pipeline(X_ml, y_ml, model)

        try:
            # Создаем GridSearchCV
            grid_search = GridSearchCV(
                estimator=done_pipeline,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Обучаем с поиском гиперпараметров
            grid_search.fit(X=X_ml, y=y_ml)
            
            # Теперь можно использовать best_score_ и best_params_
            print(f"Model: {name} | Accuracy = {round(grid_search.best_score_, 3)} | Params = {grid_search.best_params_}")

            if grid_search.best_score_ > scores_results[1]:
                scores_results = [name, grid_search.best_score_]
        except Exception as e:
            print(e)
