"""
Файл содержит "константы", которые используются в остальных файлах
При необходимости, вы можете изменять их для своего проекта
"""
import enum

DATASETS_ROOT = "datasets" # место, где находятся и сохраняются все датасеты
ASSETS_ROOT = "../assets" # место, где находятся все статические файлы (например фото графиков)
NOW_YEAR = 2025 # год сборки исходного датасета
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

        # "personal",
        "relatives",
        "crop_photo",

        "home_town",
        "city",

        "home_phone",

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

class FeatureGroups(enum.Enum):
    DROP_COLS = [
        # Поля, связанные непосредственно с аккаунтом, который собирал информацию
        'common_count', 
        'is_favorite',
        'friend_status', 
        'is_friend',
        'is_hidden_from_feed',

        'id', 
        'online', 
        'online_app',
        'skype', 
        'status_audio',

        'university',
        'university_name', 
        'faculty', 
        'faculty_name', 
        'graduation',
        'education_form', 
        'education_status', 
    ]

    EXISTS_COLS = [
        'nickname', 
        'maiden_name', 
        'domain', 
        'status', 
        'first_name',
        'last_name', 
        'city', 
        'photo_id', 
        'interests', 
        'books', 
        'tv', 
        'quotes', 
        'about', 
        'games',
        'movies', 
        'activities', 
        'music',
        'crop_photo',
        'home_town', 
        'relation_partner',

        'deactivated',
    ]

    COMPLEX_COLS = [
        'counters', 
        'personal', 

        'home_phone', 
        'site',
        'occupation',
        'bdate',
        'last_seen', 
        'screen_name', 
        'relation', 
        
        'career', 
        'military',
        'universities', 
        'schools', 
        'relatives', 
    ]

    DONE_COLS = [
        'has_photo',
        'has_mobile',
        'can_post', 
        'can_see_all_posts',
        'can_see_audio', 
        'can_write_private_message', 
        'can_send_friend_request',
        'can_be_invited_group',
        'is_verified', 
        'is_closed',
        'sex',
        'verified', 
        'can_access_closed', 
        'followers_count', 
        'is_bot', 
    ]

    CATEGORIAL_COLS = [
        'sex',
        'bdate',
        'occupation',
    ]

if __name__ == "__main__":
    print(ExplainCategories.RELATION_NEW.value[1])