import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core.default_visual import (set_visual_default, get_discrete_palette, 
                                 DOUBLE_PALETTE, TRIPLE_CMAP, TRIPLE_PALETTE, DOUBLE_CMAP,
                                 BLUE_COLOR, RED_COLOR, MINT_COLOR, DEFAULT_FIGSIZE
                                )
from core.vars import ASSETS_ROOT, ExplainCategories
from core.base import prepare_drop

def visualize_gender_education(data: pd.DataFrame, activity_col: pd.Index, sex_col: pd.Index, education_level_col: pd.Index):
    set_visual_default()
    print("=== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ 'Гендерные различия в самораскрытии' ===")
    
    fig, axes = plt.subplots(2, 2, figsize=DEFAULT_FIGSIZE)

    categorised_data = data.copy()
    categorised_data[sex_col] = categorised_data[sex_col].map(ExplainCategories.SEX.value)
    categorised_data[education_level_col] = categorised_data[education_level_col].map(ExplainCategories.EDUCATION_LEVEL.value)

    gender_palette = {
        ExplainCategories.SEX.value[0]: MINT_COLOR, 
        ExplainCategories.SEX.value[1]: RED_COLOR, 
        ExplainCategories.SEX.value[2]: BLUE_COLOR, 
    }
    
    # 1. Barplot средних значений активности
    curr_ax = axes[0, 0]
    sns.barplot(
        data=categorised_data, 
        x=education_level_col, 
        y=activity_col, 
        hue=sex_col, 
        estimator="mean", 
        palette=gender_palette, 
        ax=curr_ax,
        errorbar=None
    )
    curr_ax.set_title('Средняя активность по полу и уровню образования')
    curr_ax.set_xlabel('Уровень образования')
    curr_ax.set_ylabel('Средняя активность')
    curr_ax.legend(title="Пол")
    for container in curr_ax.containers:
        curr_ax.bar_label(container, fmt='%.2f')

    # 2. Boxplot по активности
    curr_ax = axes[0, 1]
    sns.boxplot(
        data=data[activity_col], 
        color=MINT_COLOR, 
        ax=curr_ax
    )
    curr_ax.set_title('Распределение активности')
    curr_ax.set_ylabel('Уровень активности')

    # Получаем статистики из boxplot
    stats = data[activity_col].describe()
    q1 = stats['25%']
    median = stats['50%']
    q3 = stats['75%']

    handles = [
        curr_ax.axhline(y=median, color=RED_COLOR, linestyle='--', label=f'Медиана: {median:.2f}'),
        curr_ax.axhline(y=q1, color=BLUE_COLOR, linestyle='--', label=f'Q1: {q1:.2f}'),
        curr_ax.axhline(y=q3, color=BLUE_COLOR, linestyle='--', label=f'Q3: {q3:.2f}'),
    ]

    curr_ax.legend(
        handles = handles,
        # labels = labels, 
        title="Статистика", 
        loc="upper right"
    )
    
    # 3. Heatmap корреляция
    curr_ax = axes[1, 0]
    ticks = ["активность", "пол", "ур.образования"]
    sns.heatmap(
        data[[activity_col, sex_col, education_level_col]].corr(), 
        annot=True, 
        cmap=TRIPLE_CMAP, 
        ax=curr_ax
    )
    curr_ax.set_xticklabels(ticks, rotation=0)
    curr_ax.set_yticklabels(ticks, rotation=90)
    curr_ax.set_title('Корреляция Пола, Образования и Активности')

    curr_ax = axes[1, 1]
    curr_ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_bots_and_peoples(dataset: pd.DataFrame, bots_ids: pd.Index, empty_ids: pd.Index, counter_cols: pd.Index):
    set_visual_default()
    # alive_ids = dataset.drop(bots_ids).index
    alive_ids = prepare_drop(dataset, rows=(list(bots_ids) + list(empty_ids)))
    bots_count = len(bots_ids)
    empty_count = len(empty_ids)

    plt.pie(
        [len(alive_ids), bots_count, empty_count],
        labels=["Люди", "Боты", "Пустые аккаунты"],
        colors=[BLUE_COLOR, RED_COLOR, MINT_COLOR],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Доля Ботов и Живых профилей")
    plt.show()
    plt.close()

def save_plot_png(save_dir: str, name: str, dpi: int=300):
    os.makedirs(save_dir, exist_ok=True)
    file_name = name + ".png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Сохранён график: {save_path}")

def base_plot(
        plot: plt.axes, 
        title: str, 
        xlabel: str = None, 
        ylabel: str = None,
        ticks: list = None,

        is_save: bool = False,
        save_dir: str = "custom_barplots", 
        save_name: str = "default_plot",
        
        is_show: bool = True,
        ax = None,

        legend_handles: list = None
    ):

    if ax != None:
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if ticks:
            ax.set_xticklabels(ticks, rotation=15)
        if legend_handles:
            ax.legend(handles=legend_handles)
    else:
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if ticks:
            plt.xticks(ticks=np.array(range(len(ticks))) + 0.5, labels=ticks, rotation=15)
        if legend_handles:
            plt.legend(handles=legend_handles)

    plt.tight_layout()

    if is_save:
        save_plot_png(save_dir=os.path.join(ASSETS_ROOT, save_dir), name=save_name)
    if is_show:
        plt.show()

def custom_barplot(
        dataset: pd.DataFrame, 
        x: pd.Index,
        y: pd.Index,
        title: str, 
        xlabel: str, 
        ylabel: str, 
        palette: dict,
        save_dir: str = "custom_barplots", 
        is_save: bool = False,
        save_name: str = "default_plot",
        sorting: bool = False,
        is_show=True,
        ax = None,
    ):
    dataset = dataset.copy()

    if sorting:
        dataset = dataset.sort_values(by=y, ascending=False)

    ticks = dataset[x].values.tolist()

    plot = sns.barplot(
        data=dataset, 
        x=x, 
        y=y, 
        palette=palette,
        ax=ax,
        order=ticks,
    )

    if ax != None:
        for container in plot.containers:
            ax.bar_label(container, fmt="%.2f")
    else:
        for container in plot.containers:
            plt.bar_label(container, fmt="%.2f")

    base_plot(
        plot=plot,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ticks=ticks,
        is_save=is_save,
        save_dir=save_dir,
        save_name=save_name,
        is_show=is_show,
        ax=ax,
    )

def custom_heatmap(
        dataset: pd.DataFrame, 
        title: str, 
        cmap: dict,
        xlabel: str = None, 
        ylabel: str = None, 
        save_dir: str = "custom_barplots", 
        is_save: bool = False,
        save_name: str = "default_plot",
        annot=True,
        is_show=True,
        fmt='.2f',
        vmin=None, 
        vmax=None,
        ax = None,
    ):

    dataset = dataset.copy()

    plot = sns.heatmap(
        dataset, 
        annot=annot, 
        cmap=cmap, 
        fmt=fmt,
        center=0,
        vmin=vmin, 
        vmax=vmax,
        ax=ax
    )

    yticks = dataset.index.tolist()

    if ax != None:
        ax.set_yticklabels(yticks, rotation=0, va="center")
    else:
        plt.yticks(ticks=np.array(range(len(yticks))) + 0.5, labels=yticks, rotation=0, va="center")


    base_plot(
        plot=plot,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ticks=dataset.columns.tolist(),
        is_save=is_save,
        save_dir=save_dir,
        save_name=save_name,
        is_show=is_show,
        ax=ax,
    )

