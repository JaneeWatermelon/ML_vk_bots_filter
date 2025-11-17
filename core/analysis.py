import numpy as np
import pandas as pd

from core.base import split_cols, prepare_drop
from core.vars import NewFeaturesNames, BulkFeatures

def get_flews_ids(dataset: pd.DataFrame, features: pd.Index=None):
    if features:
        numeric_features, _ = split_cols(dataset[features])
    else:
        numeric_features, _ = split_cols(dataset)

    flews_ids = set()

    pd_cols = []
    pd_left_sides = []
    pd_right_sides = []
    pd_counts = []

    for col in numeric_features:
        data = dataset[col].dropna()
        Q_1 = dataset[col].quantile(0.25)
        Q_3 = dataset[col].quantile(0.75)
        IQR = Q_3 - Q_1
        left_side = Q_1 - 1.5 * IQR
        right_side = Q_3 + 1.5 * IQR
        curr_flews_ids = set(data[(data <= left_side) | (data >= right_side)].index.tolist())

        if len(curr_flews_ids) != len(data):
            flews_ids |= curr_flews_ids

            pd_cols.append(col)
            pd_left_sides.append(left_side)
            pd_right_sides.append(right_side)
            pd_counts.append(len(curr_flews_ids))

    flews_dataset = pd.DataFrame(data={"column": pd_cols, "left_side": pd_left_sides, "right_side": pd_right_sides, "count": pd_counts})

    return list(flews_ids), flews_dataset

def get_bots(dataset: pd.DataFrame, counter_cols: pd.Index, border: int = 70) -> tuple[pd.Index, pd.Index]:
    MAX_SCORE = 100
    SMALL_K = 5
    result_bot_scores = np.zeros(dataset.shape[0])

    def culc_score(bot_scores: np.ndarray, bot_mask, add_score: int):
        maybe_bot_ids = dataset[bot_mask].index
        bot_scores[maybe_bot_ids] += add_score

        return bot_scores.clip(0, MAX_SCORE)
    
    def anyway_bot_ids(dataset: pd.DataFrame):
        # 1. Аккаунт удалён или забанен
        bot_mask = (dataset["names_status"] == "deactivated")
        bot_mask |= (dataset["walls_status"] == "User was deleted or banned")
        bot_mask |= (dataset["first_name"] == "DELETED") & (dataset["last_name"].isna() | (dataset["last_name"] == ""))
        # 2. Средний промежуток выкладывания постов менее 1 дня
        bot_mask |= (dataset[NewFeaturesNames.POSTS_INTERVAL.value] < 3600*24) & (dataset[NewFeaturesNames.POSTS_INTERVAL.value] != -1)
        
        return dataset[bot_mask].index

    def anyway_empty_ids(dataset: pd.DataFrame):
        # 1. Counter_cols + friends_count = 0 
        bot_mask = ((dataset[counter_cols + ["friends_count"]].fillna(0).sum(axis=1) == 0) & (dataset["is_closed"].fillna(0) == 0))
        
        return dataset[bot_mask].index
    
    def anyway_human_ids(dataset: pd.DataFrame):
        # 5. Не базовый screen_name = Не бот
        def is_id_base(x: str):
            return x is None or (x.startswith("id") and x[2:].isdigit())

        bot_mask = (dataset["screen_name"].apply(is_id_base) == False)
        
        return dataset[bot_mask].index
    
    def print_100_percent_bots_count(bot_scores: np.ndarray):
        func = np.vectorize(lambda x: x >= border)
        count = func(bot_scores).sum()
        print(count)
    
    def maybe_bot_scores(dataset: pd.DataFrame, bot_scores: np.ndarray):
        # 6. Несоответствие медиа-активности и количества друзей
        bot_mask = dataset["friends_status"] == "success"
        masked_dataset = dataset[bot_mask]
        clips_activity = ((masked_dataset["counters_clips_likes"] + (masked_dataset["counters_clips_views"] / 5)) / (masked_dataset["counters_clips"] * 2).apply(lambda x: max(x, 1)))
        walls_activity = ((masked_dataset["wall_posts_likes"] + (masked_dataset["wall_posts_views"] / 5)) / (masked_dataset["counters_posts"] * 2).apply(lambda x: max(x, 1)))
        max_media_activity = pd.concat([clips_activity, walls_activity], axis=1).max(axis=1)

        score = (masked_dataset["friends_count"].fillna(0) / max_media_activity.apply(lambda x: max(x, 1)))
        bot_scores = culc_score(bot_scores, bot_mask, score * SMALL_K)
        print_100_percent_bots_count(bot_scores)

        score = (max_media_activity.fillna(0) / masked_dataset["friends_count"].apply(lambda x: max(x, 1)))
        bot_scores = culc_score(bot_scores, bot_mask, score * SMALL_K)
        print_100_percent_bots_count(bot_scores)

        # 7. Несоответствие количества подписок и количества друзей
        bot_mask = dataset["friends_status"] == "success"
        masked_dataset = dataset[bot_mask]
        score = (masked_dataset["counters_subscriptions"].fillna(0) / masked_dataset["friends_count"].apply(lambda x: max(x, 1)))
        bot_scores = culc_score(bot_scores, bot_mask, score * SMALL_K)
        print_100_percent_bots_count(bot_scores)

        # 8. Много отсутствующей персональной информации
        bot_mask = dataset["is_closed"] == 0
        
        score = dataset[bot_mask][BulkFeatures.PERSONAL_INFOS.value].count(axis=1)
        score = score.apply(lambda x: len(BulkFeatures.PERSONAL_INFOS.value) - x)
        score = score / len(BulkFeatures.PERSONAL_INFOS.value)
        bot_scores = culc_score(bot_scores, bot_mask, score * 70)
        print_100_percent_bots_count(bot_scores)

        # 9. Отсутствие аватара
        bot_mask = dataset["has_photo"] == 0
        bot_scores = culc_score(bot_scores, bot_mask, 20)
        print_100_percent_bots_count(bot_scores)
        
        return bot_scores

    empty_ids = anyway_empty_ids(dataset)
    print(f"Пустых: {len(empty_ids)}")
    bot_ids = anyway_bot_ids(dataset)
    print(f"Ботов: {len(bot_ids)}")
    human_ids = anyway_human_ids(dataset)
    print(f"Людей: {len(human_ids)}")
    maybe_bot_ids = dataset[maybe_bot_scores(dataset, result_bot_scores) >= border].index
    print(f"Возможных ботов: {len(maybe_bot_ids)}")

    result_empty = list(empty_ids)
    result_bots = list(set(bot_ids) | set(maybe_bot_ids) - set(human_ids) - set(empty_ids))

    return pd.Index(data=result_bots), pd.Index(data=result_empty)
