from typing import Any

import pandas as pd
from scipy.stats import wilcoxon

from attention_lapse_detection.types import GridCell

threshold = 0.05
CELL = ["model", "fps", "window_seconds"]


def scores_by_seed(runs: pd.DataFrame, cell: GridCell) -> pd.Series:
    model, fps, window = cell
    rows = runs.loc[
        (runs.model == model) & (runs.fps == fps) & (runs.window_seconds == window)
    ]
    
    return rows.set_index("seed")["val_clip_ap"].sort_index()


def selection_table(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.loc[runs.features == "all_features"]

    cells = runs.groupby(CELL).agg(
        val_ap=("val_clip_ap", "mean"),
        sd=("val_clip_ap", "std"),
        params=("params", "first"),
    )

    cell_list: list[GridCell] = cells.index.tolist()
    best_cell = cell_list[cells["val_ap"].argmax()]
    best_scores = scores_by_seed(runs, best_cell)

    p_values = []
    eligible = []
    for cell in cell_list:
        if cell == best_cell:
            p_values.append(float("nan"))
            eligible.append(True)
        else:
            result: Any = wilcoxon(scores_by_seed(runs, cell), best_scores)
            p_values.append(result.pvalue)
            eligible.append(result.pvalue > threshold)

    cells["diff_vs_largest"] = cells["val_ap"] - cells["val_ap"].max()
    cells["p_vs_largest"] = p_values
    cells["eligible"] = eligible

    smallest_eligible = cells.loc[cells["eligible"], "params"].idxmin()
    cells["retained"] = cells.index == smallest_eligible

    return cells


def select(runs: pd.DataFrame) -> GridCell:
    cells = selection_table(runs)
    retained: list[GridCell] = cells.loc[cells["retained"]].index.tolist()
    return retained[0]


def select_per_model(runs: pd.DataFrame) -> dict[str, GridCell]:
    winners = {}
    for model_name in sorted(runs["model"].unique()):
        winners[model_name] = select(runs.loc[runs["model"] == model_name])

    return winners
