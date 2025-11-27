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
        ax=ax,
        # annot_kws={"size":6}
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

