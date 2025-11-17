"""
Файл содержит "константы", которые используются в остальных файлах
При необходимости, вы можете изменять их для своего проекта
"""
import enum

DATASETS_ROOT = "datasets" # место, где находятся и сохраняются все датасеты
ASSETS_ROOT = "../assets" # место, где находятся все статические файлы (например фото графиков)
NOW_YEAR = 2024 # год сборки исходного датасета
AGE_LIMIT = 80 # максимальный адекватный возраст

# Эти переменные указывают пути к некоторым библиотекам
# Вам они скорее всего не пригодятся, поэтому можете установить OVERRIDE_TCL_TK = False
OVERRIDE_TCL_TK = True
TCL_LIBRARY = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
TK_LIBRARY = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

class ExplainVars(enum.Enum):
    UNKNOWN = "не указано"
    IN_RELATIONSHIP = "в отношениях"
    ALONE = "одинок(а)"

# Класс-перичислитель, хранящий перевод категорий в текст
class ExplainCategories(enum.Enum):
    RELATION_OLD = {
        1: "не женат/не замужем",
        2: "есть друг/есть подруга",
        3: "помолвлен/помолвлена",
        4: "женат/замужем",
        5: "всё сложно",
        6: "в активном поиске",
        7: "влюблён/влюблена",
        8: "в гражданском браке",
        0: ExplainVars.UNKNOWN.value
    }
    RELATION_NEW = {
        1: "не женат/не замужем",
        10: ExplainVars.IN_RELATIONSHIP.value,
        4: "женат/замужем",
        11: "всё сложно",
        6: "в активном поиске",
        0: ExplainVars.UNKNOWN.value
    }
    RELATION_COMPRESSED = {
        1: ExplainVars.ALONE.value,
        6: ExplainVars.ALONE.value,
        11: ExplainVars.ALONE.value,
        10: ExplainVars.IN_RELATIONSHIP.value,
        4: ExplainVars.IN_RELATIONSHIP.value,
        0: ExplainVars.UNKNOWN.value
    }
    EDUCATION_LEVEL = {
        0: ExplainVars.UNKNOWN.value,
        1: "школа",
        2: "высшее",
        3: "несколько высших",
    }
    UNIQUENESS = {
        0: "НПС",
        1: "пассивные",
        2: "активные",
        3: "очень активные",
        4: "медиа-личности",
    }
    SEX = {
        0: ExplainVars.UNKNOWN.value,
        1: "женщины",
        2: "мужчины",
    }
    WALLS_STATUS = {
        0: "не доступен",
        1: "доступен",
    }
    FRIENDS_STATUS = {
        0: "This profile is private",
        1: "success",
    }
    BDATE = {
        0: ExplainVars.UNKNOWN.value,
        1: "день и месяц",
        2: "полностью",
    }

# Класс-перичислитель, хранящий названия новых признаков
class NewFeaturesNames(enum.Enum):
    FULLNESS = "fullness"
    CONTENT_ACTIVITY = "content_activity"
    POSTS_INTERVAL = "posts_interval"

class BulkFeatures(enum.Enum):
    ACTIVITY_INFOS = [
        "activities",
        "books",
        "games",
        "quotes",
        "tv",
        "interests",
        "movies",
        "music",
    ]
    PERSONAL_INFOS = [
        "about",
        "status",

        "personal",
        "relatives",
        "crop_photo",

        "home_town",
        "city",

        "home_phone",
        # "mobile_phone",

        "schools",
        "universities",
        "career",
    ]
    PRIVACY_SCORE = [
        "can_send_friend_request",
        "can_access_closed",
        "can_be_invited_group",
        "can_post",
        "can_see_all_posts",
        "can_see_audio",
        "can_write_private_message",
    ]
    COMMUNICATION_ACCESSIBILITY = [
        "can_send_friend_request",
        "can_be_invited_group",
        "can_write_private_message",
    ]

# Список колонок на удаление (часть 1, пункт 5)
FEATURES_TO_REMOVE = [
    "_id",
    'blacklisted', 'blacklisted_by_me', 'friend_status', 'is_friend',

    'is_no_index', 'online', 'trending', 
    # 'wall_comments',

    'occupation', 'military', 'maiden_name',

    'screen_name', 'names_status',

    'photo_id', 'crop_photo', 'photo_100', 'photo_200', 'photo_200_orig',
    'photo_400_orig', 'photo_50', 'photo_max', 'photo_max_orig',

    'verified', 'online_app', 'online_mobile', 'relation_partner', 'status_audio', 'country',
    'education', 'connections', 'contacts', 'exports', 'is_favorite', 'is_service', 'lists', 'timezone',

    'schools',

    'universities', 'university', 'faculty', 'university_name',
    'education_form', 'education_status', 'faculty_name', 'graduation',

    'last_updated_x', 'last_updated_y', 'downloaded_at',

    #,'counters_albums', 'counters_audios', 'counters_followers', 'counters_gifts',
    #'counters_groups', 'counters_pages', 'counters_photos', 'counters_subscriptions',
    #'counters_user_photos', 'counters_videos', 'counters_video_playlists',
    #'counters_clips_followers', 'counters_clips_views', 'counters_clips_likes',
    #'counters_posts', 'counters_articles', 'counters_clips',

    'can_send_friend_request', 'can_access_closed', 'can_be_invited_group', 'can_post',
    'can_see_all_posts', 'can_see_audio', 'can_write_private_message',

    #'has_mobile','home_phone', 
    'skype', 'is_hidden_from_feed',
]

if __name__ == "__main__":
    print(ExplainCategories.RELATION_NEW.value[1])