import os
import pandas as pd

from core.vars import DATASETS_ROOT

def override_tcl_tk():
    """Устанавливает окружение для Tcl/Tk библиотек"""
    os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

def merge_users(users_url: str, friends_url: str, user_names_url: str, new_walls_url: str):
    """Объединяет несколько JSON файлов в один датафрейм"""
    users_data = pd.read_json(users_url)
    friends_data = pd.read_json(friends_url)
    user_names_data = pd.read_json(user_names_url)
    new_walls_data = pd.read_json(new_walls_url)

    def prepare_to_merge(dataset: pd.DataFrame, cols_to_change: dict, name: str, cols_to_drop: list=[], base_dataset: pd.DataFrame=users_data):
        """Подготавливает данные для объединения, изменяя названия колонок"""
        dataset = dataset.rename(columns=cols_to_change)
        base_cols = base_dataset.drop(columns=cols_to_change.values(), errors="ignore").columns.tolist()

        new_cols_to_change = {}

        for col in dataset.columns:
            if col in base_cols:
                new_cols_to_change[col] = name + "_" + col

        if cols_to_drop:
            dataset = dataset.drop(columns=cols_to_drop)

        dataset = dataset.rename(columns=new_cols_to_change)

        return dataset
    
    friends_data = prepare_to_merge(friends_data, {"fake_id": "id", "count": "friends_count"}, "friends", ["_id"])
    user_names_data = prepare_to_merge(user_names_data, {"fake_id": "id"}, "names", ["_id"])
    new_walls_data = prepare_to_merge(new_walls_data, {"user_id": "id"}, "walls", ["_id"])

    users_data = users_data.merge(friends_data, on="id")
    users_data = users_data.merge(user_names_data, on="id")
    users_data = users_data.merge(new_walls_data, on="id")
    # users_data.to_excel("dataset.xlsx")
    # users_data.to_json("dataset.json")

    return users_data

def load_datasets(is_merge: bool=False):
    """Загружает наборы данных из файлов"""
    # Основные
    if is_merge:
        users_data = merge_users(
            os.path.join(DATASETS_ROOT, "users.json"),
            os.path.join(DATASETS_ROOT, "friends.json"),
            os.path.join(DATASETS_ROOT, "user_names.json"),
            os.path.join(DATASETS_ROOT, "new_walls.json"),
        )
    else:
        users_data = pd.read_json(os.path.join(DATASETS_ROOT, "dataset.json"))

    # Дополнительные
    guman_data = pd.read_excel(os.path.join(DATASETS_ROOT, "guman.xlsx")).iloc[:, 1:]
    sap_data = pd.read_excel(os.path.join(DATASETS_ROOT, "socials_and_psyco.xlsx")).iloc[:, 1:]

    return users_data, guman_data, sap_data

def split_cols(dataset: pd.DataFrame):
    """Разделяет колонки на категориальные и числовые"""
    categ_features = dataset.select_dtypes("object").columns
    numeric_features = dataset.select_dtypes(exclude=["object"]).columns
    return numeric_features, categ_features

def label_encoder(dataset: pd.DataFrame, sort: bool = True, reverse: bool = False) -> pd.DataFrame:
    """Применяет кодирование меток для категориальных признаков"""
    for col in dataset.columns:
        encoder_map = {}
        i = 0

        unique_list = dataset[col].unique().tolist()
        if sort:
            unique_list.sort(reverse=reverse)

        for val in unique_list:
            encoder_map[val] = i
            i += 1

        dataset[col] = dataset[col].map(encoder_map)

    return dataset

def prepare_drop(dataset: pd.DataFrame, columns: pd.Index=None, rows: pd.Index=None, reset_index: bool=True, drop_index: bool=True) -> pd.DataFrame:
    """Удаляет колонки и/или строки из датафрейма"""
    if columns is not None:
        dataset = dataset.drop(columns=columns, errors="ignore")
    if rows is not None:
        dataset = dataset.drop(rows, errors="ignore")
    if reset_index:
        dataset = dataset.reset_index(drop=drop_index)
    return dataset

def prepare_complex_cols(dataset: pd.DataFrame, to: str, columns: pd.Index=None) -> tuple[pd.DataFrame, list]:
    """Расширяет сложные колонки (списки, словари) в несколько колонок"""
    if columns == None:
        columns = dataset.columns

    result_df = dataset.copy()
    complex_cols = []

    if to == "json":
        for col in columns:
            # Создаем DataFrame из словарей
            expanded_df = pd.json_normalize(dataset[col])

            # Добавляем префикс к названиям колонок
            expanded_df = expanded_df.add_prefix(f'{col}_')

            complex_cols += expanded_df.columns.tolist()

            # Объединяем с исходным DataFrame
            result_df = pd.concat([result_df.drop(columns=[col]), expanded_df], axis=1)

    elif to == "string":
        for col in result_df.columns:
            if result_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                print(f"Внимание: колонка '{col}' содержит сложные данные | Преобразованно в строку")
                result_df[col] = result_df[col].apply(lambda x: str(x))
                complex_cols.append(col)

    return result_df, complex_cols

def prepare_null(dataset: pd.DataFrame):
    """Заполняет пропуски в датафрейме"""
    converted_dataset = dataset.convert_dtypes()

    categorial_cols = converted_dataset.select_dtypes("object").columns
    string_cols = dataset.drop(columns=categorial_cols).select_dtypes("object").columns
    numerical_cols = dataset.select_dtypes(exclude=["object"]).columns

    # save_dataset(dataset[categorial_cols], "categorial_data.xlsx")

    dataset[numerical_cols] = dataset[numerical_cols].fillna(0)
    dataset[string_cols] = dataset[string_cols].fillna("")
    for col in categorial_cols:
        # Пропускаем пустые колонки
        if dataset[col].isna().sum() == 0:
            continue
            
        # Проверяем тип данных в колонке
        non_null_values = dataset[col].dropna()
        if len(non_null_values) > 0:
            random_not_null_value = non_null_values.iloc[0]
            
            if isinstance(random_not_null_value, list):
                # Для списков заполняем пустыми списками
                dataset[col] = dataset[col].apply(lambda x: x if isinstance(x, list) else [])
            else:
                # Для словарей заполняем пустыми словарями
                dataset[col] = dataset[col].apply(lambda x: x if isinstance(x, dict) else {})

    return dataset

def save_dataset(dataset: pd.DataFrame, file_name: str, base_path: str=os.path.join(DATASETS_ROOT)):
    """Сохраняет датафрейм в файл"""
    format = file_name.split(".")[-1]

    path = os.path.join(base_path, file_name)

    if format == "json":
        dataset.to_json(path)
    elif format == "xlsx":
        dataset.to_excel(path)
    elif format == "csv":
        dataset.to_csv(path)
    else:
        print(f"Передан недоступный формат: {format}")

def extract_features_by_start(dataset: pd.DataFrame, start: str):
    columns = dataset.columns.tolist()
    result = []
    for col in columns:
        if col.startswith(start):
            result.append(col)
    return result

def get_low_disp_features(dataset: pd.DataFrame, alpha: float = 0.98) -> pd.DataFrame:
    low_disp_cols = []

    for col in dataset.columns:
        norm_value_counts = dataset[col].value_counts(normalize=True).sort_values(ascending=False)
        most_frequent = norm_value_counts.iloc[0]
        if most_frequent >= alpha:
            low_disp_cols.append(col)
    
    return dataset[low_disp_cols]

def get_high_corr_pairs(dataset: pd.DataFrame, alpha: float = 0.9, verbose: bool = False) -> list[tuple[str, str, float]]:
    n = len(dataset)
    corr_drop_cols = []

    for k, col in enumerate(dataset.columns):
        component = dataset[col]
        for i in range(k+1, n):
            corr_val = component.iloc[i]
            if abs(corr_val) >= alpha:
                corr_drop_cols.append((col, component.index[i], corr_val))
                if verbose:
                    print(f"{col} - {component.index[i]} | corr_val: {round(corr_val, 3)}")
    
    return corr_drop_cols

COLS_TO_RENAME = {
    "friends_count": "counters_friends"
}