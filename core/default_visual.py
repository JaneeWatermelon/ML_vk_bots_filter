import os
import warnings
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from core.base import override_tcl_tk
from core.vars import OVERRIDE_TCL_TK, ExplainCategories

if OVERRIDE_TCL_TK:
    override_tcl_tk()
    
# ==============================================
# ЕДИНЫЙ СТИЛЬ ДЛЯ ВСЕХ ГРАФИКОВ ПРОЕКТА
# ==============================================

DEFAULT_FIGSIZE = (15, 8)
BLUE_COLOR = "#4882FF"
MINT_COLOR = "#8EFFCA"
RED_COLOR = "#FF4747"
# --- Цветовая палитра ---
TRIPLE_PALETTE = [BLUE_COLOR, MINT_COLOR, RED_COLOR]
DOUBLE_PALETTE = [BLUE_COLOR, MINT_COLOR]
TRIPLE_CMAP = LinearSegmentedColormap.from_list("custom", TRIPLE_PALETTE)
DOUBLE_CMAP = LinearSegmentedColormap.from_list("custom", DOUBLE_PALETTE)
GENDER_PALETTE = {0: MINT_COLOR, 1: RED_COLOR, 2: BLUE_COLOR}
GENDER_PALETTE_EXPLAINED = {
    ExplainCategories.SEX.value[0]: MINT_COLOR, 
    ExplainCategories.SEX.value[1]: RED_COLOR, 
    ExplainCategories.SEX.value[2]: BLUE_COLOR
}

def get_discrete_palette(cmap, n_colors=6, reverse=False):
    """
    Создает дискретную палитру с заданным количеством цветов
    """
    new_cmap = []
    doles = np.linspace(0, 1, n_colors)
    for dole in doles:
        new_cmap.append(cmap(dole))
    if reverse:
        new_cmap.reverse()
    return new_cmap

def set_visual_default():
    warnings.filterwarnings("ignore")
    # --- Установка стилей ---
    plt.style.use('seaborn-v0_8-whitegrid') # Чистый стиль с сеткой
    sns.set_style("whitegrid")

    # print(plt.rcParams.keys())

    # --- Настройка параметров matplotlib (rcParams) ---
    plt.rcParams['figure.figsize'] = DEFAULT_FIGSIZE # Размер графиков по умолчанию
    plt.rcParams['font.size'] = 12 # Базовый размер шрифта
    plt.rcParams['axes.titlesize'] = 16 # Размер шрифта заголовка
    plt.rcParams['axes.labelsize'] = 14 # Размер шрифта подписей осей
    plt.rcParams['xtick.labelsize'] = 12 # Размер шрифта меток на оси X
    plt.rcParams['ytick.labelsize'] = 12 # Размер шрифта меток на оси Y
    plt.rcParams['legend.fontsize'] = 12 # Размер шрифта легенды
    plt.rcParams['figure.dpi'] = 100 # Качество изображения

if __name__ == "__main__":
    set_visual_default()

    # Пример 1: Простой линейный график
    dates = pd.date_range(start='2023-01-01', periods=6, freq='M')
    values = [120, 145, 160, 155, 180, 210]

    plt.figure() # Создаем новый график
    plt.plot(dates, values, marker='o', linewidth=2)
    plt.title('Динамика роста подписчиков')
    plt.xlabel('Дата')
    plt.ylabel('Количество подписчиков')
    plt.xticks(rotation=45)
    plt.tight_layout() # Убираем пересечения подписей
    plt.show()

    # Пример 2: Столбчатая диаграмма с Seaborn
    categories = ['Фото', 'Видео', 'Текст', 'Опрос', 'Репост']
    counts = [45, 30, 15, 5, 10]
    heatmap_vals = [
        [45, 30, 8, 5, 10],
        [52, 0, 15, 5, 10],
        [45, 30, 4, 5, 3],
        [45, 30, 15, 5, 10],
        [34, 30, 15, 43, 15],
    ]

    sns.barplot(x=counts, y=categories, hue=["1", "2", "3", "4", "5"], palette=get_discrete_palette(cmap=DOUBLE_CMAP, n_colors=5, reverse=True))
    plt.title('Типы публикаций в аккаунте')
    plt.xlabel('Количество постов')
    plt.tight_layout()
    plt.show()

    sns.heatmap(heatmap_vals, cmap=TRIPLE_CMAP, annot=True)
    plt.title('heatmap')
    plt.tight_layout()
    plt.show()