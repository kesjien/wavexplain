"""
examples/favorita_sales_forecast.py

Demonstrates wavexplain's real API on a genuine, substantial dataset:
the Corporacion Favorita Grocery Sales Forecasting dataset (125M+ rows,
174,685 store/item series).

This script is intentionally split from the core wavexplain package:
data loading and panel-building are domain-specific glue code, exactly
the kind of thing the README says wavexplain does NOT do for you. What
this demonstrates is wiring your own data into wavexplain's model,
explainer, and card-rendering pieces.

Requires: the Favorita dataset extracted locally (train.csv, items.csv),
available from the original Kaggle competition:
https://www.kaggle.com/c/favorita-grocery-sales-forecasting

Usage:
    python favorita_sales_forecast.py --train-csv path/to/train.csv \\
        --checkpoint path/to/trained_model.pt
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from wavexplain import MultiSeriesWaveNet, CounterfactualExplainer, render_card_html

INPUT_LENGTH = 90
HORIZON = 16


# ---------------------------------------------------------------------------
# Domain-specific data loading. This is exactly the kind of thing wavexplain
# deliberately does not include -- your own panel-building code lives here,
# not in the library.
# ---------------------------------------------------------------------------

def load_favorita_train(train_csv_path: str) -> pd.DataFrame:
    dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32",
              "onpromotion": "object"}
    df = pd.read_csv(
        train_csv_path,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype=dtypes, parse_dates=["date"], low_memory=False,
    )
    df["unit_sales"] = df["unit_sales"].clip(lower=0)
    df["onpromotion"] = (
        df["onpromotion"].astype(str).map({"True": 1.0, "False": 0.0}).fillna(0.0).astype("float32")
    )
    return df


def build_series_panels(df: pd.DataFrame):
    df = df.copy()
    df["series_id"] = df.groupby(["store_nbr", "item_nbr"], sort=True).ngroup().astype("int32")
    series_index = (
        df[["series_id", "store_nbr", "item_nbr"]].drop_duplicates()
        .sort_values("series_id").reset_index(drop=True)
    )
    num_series = len(series_index)

    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    date_id_map = {d: i for i, d in enumerate(date_range)}
    df["date_id"] = df["date"].map(date_id_map).astype("int32")

    num_days = len(date_range)
    sales_panel = np.zeros((num_series, num_days), dtype="float32")
    promo_panel = np.zeros((num_series, num_days), dtype="float32")

    grouped = df.groupby(["series_id", "date_id"]).agg(
        unit_sales=("unit_sales", "sum"), onpromotion=("onpromotion", "max"),
    )
    idx_series = grouped.index.get_level_values(0).to_numpy()
    idx_date = grouped.index.get_level_values(1).to_numpy()
    sales_panel[idx_series, idx_date] = grouped["unit_sales"].to_numpy(dtype="float32")
    promo_panel[idx_series, idx_date] = grouped["onpromotion"].to_numpy(dtype="float32")

    return sales_panel, promo_panel, series_index, date_range


def load_or_build_panels(train_csv_path: str, cache_dir: str = "favorita_cache"):
    if os.path.exists(f"{cache_dir}/sales_panel.npy"):
        print(f"Loading cached panels from '{cache_dir}/'...")
        sales_panel = np.load(f"{cache_dir}/sales_panel.npy")
        promo_panel = np.load(f"{cache_dir}/promo_panel.npy")
        series_index = pd.read_csv(f"{cache_dir}/series_index.csv")
        date_range = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f"{cache_dir}/date_range.csv")["date"]))
        return sales_panel, promo_panel, series_index, date_range

    print("No cache found, building from raw CSV (this can take a few minutes)...")
    df = load_favorita_train(train_csv_path)
    sales_panel, promo_panel, series_index, date_range = build_series_panels(df)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(f"{cache_dir}/sales_panel.npy", sales_panel)
    np.save(f"{cache_dir}/promo_panel.npy", promo_panel)
    series_index.to_csv(f"{cache_dir}/series_index.csv", index=False)
    pd.Series(date_range).to_csv(f"{cache_dir}/date_range.csv", index=False, header=["date"])
    return sales_panel, promo_panel, series_index, date_range


# ---------------------------------------------------------------------------
# wavexplain usage starts here -- this is the part the library actually provides.
# ---------------------------------------------------------------------------

def explain_one_series(model, sales_panel, promo_panel, series_id, device):
    """
    Builds the input window for one series' held-out forecast, then uses
    wavexplain's CounterfactualExplainer to decompose it into three named
    groups: seasonal_pattern, recent_trend, and promotion_effect.
    """
    num_days = sales_panel.shape[1]
    start = num_days - INPUT_LENGTH - HORIZON

    sales_w = sales_panel[series_id, start:start + INPUT_LENGTH]
    promo_w = promo_panel[series_id, start:start + INPUT_LENGTH]

    log_sales = np.log1p(np.clip(sales_w, 0, None))
    full_input = np.stack([log_sales, promo_w])  # (2, input_length), model's input space

    sales_history = sales_panel[series_id, :num_days - HORIZON]
    sales_baseline = float(np.log1p(np.clip(sales_history, 0, None)).mean())
    promo_baseline = 0.0

    promo_mask = promo_w > 0.5
    recent_mask = np.zeros(INPUT_LENGTH, dtype=bool)
    recent_mask[-14:] = True
    recent_trend_mask = recent_mask & (~promo_mask)
    seasonal_pattern_mask = (~promo_mask) & (~recent_mask)

    reveal_groups = OrderedDict([
        ("seasonal_pattern", seasonal_pattern_mask),
        ("recent_trend", recent_trend_mask),
        ("promotion_effect", promo_mask),
    ])

    explainer = CounterfactualExplainer(
        model, series_id=series_id, device=device,
        output_transform=lambda x: torch.expm1(x.clamp(min=0)),
    )
    contributions, baseline_pred, full_pred = explainer.explain(
        full_input, baseline_values=[sales_baseline, promo_baseline],
        reveal_groups=reveal_groups,
    )
    return contributions, baseline_pred, full_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="favorita-grocery-sales-forecasting/train.csv")
    parser.add_argument("--checkpoint", default="wavenet_multiseries.pt")
    parser.add_argument("--series-id", type=int, default=None,
                         help="Series to explain; if omitted, picks one with a genuine promo/non-promo mix.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sales_panel, promo_panel, series_index, date_range = load_or_build_panels(args.train_csv)
    num_series = sales_panel.shape[0]
    print(f"Panel: {num_series} series x {sales_panel.shape[1]} days")

    model = MultiSeriesWaveNet(num_series=num_series, horizon=HORIZON, num_covariates=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    series_id = args.series_id
    if series_id is None:
        window_promo = promo_panel[:, -(INPUT_LENGTH + HORIZON):-HORIZON]
        promo_fraction = window_promo.mean(axis=1)
        mixed = np.where((promo_fraction > 0.15) & (promo_fraction < 0.85))[0]
        series_id = int(np.random.default_rng(0).choice(mixed))
        print(f"No --series-id given, auto-selected series {series_id} (has a genuine promo/non-promo mix)")

    contributions, baseline_pred, full_pred = explain_one_series(
        model, sales_panel, promo_panel, series_id, device
    )

    print(f"\nSeries {series_id}:")
    print(f"  Baseline prediction (nothing revealed): {baseline_pred:.1f}")
    for name, val in contributions.items():
        print(f"  {name}: {val:+.1f}")
    print(f"  Full prediction (real forecast):        {full_pred:.1f}")
    reconstructed = baseline_pred + sum(contributions.values())
    print(f"  Check -- baseline + contributions:      {reconstructed:.1f} "
          f"(should equal full prediction exactly)")

    row = series_index[series_index["series_id"] == series_id].iloc[0]
    output_path = f"favorita_card_{series_id}.html"
    render_card_html(
        title=f"Store {int(row['store_nbr'])} - Item {int(row['item_nbr'])}",
        total_forecast=full_pred,
        contributions=contributions,
        baseline_prediction=baseline_pred,
        highlight_group="promotion_effect",
        unit_label="units",
        output_path=output_path,
    )
    print(f"\nSaved card to {output_path}")


if __name__ == "__main__":
    main()
