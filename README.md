# wavexplain

<img width="472" height="461" alt="image" src="https://github.com/user-attachments/assets/7c878817-cc5f-4370-a56c-b9d2b1f9c6fe" />


Causal, counterfactual attribution for multi-series time-series forecasters.

Most forecasting models give you a number. `wavexplain` gives you a number
plus an honest answer to *why*: which parts of a series' recent history
actually drove this specific prediction, measured directly rather than
approximated.

## Why counterfactual, not just attribution-value allocation

A common approach to "explaining" a forecast is to compute a per-timestep
attribution score (e.g. via SHAP) and allocate shares of that score into
named buckets. That approach has a real failure mode: if a bucket has very
few data points, or its attribution values happen to have mixed signs, the
allocated share can collapse toward zero even when the underlying driver is
real and substantial. This showed up during development: a product with a
completely normal 20-90 unit baseline produced a card claiming its "typical
pattern" contribution was zero, purely because there weren't enough
non-promoted days in that specific window to sum over.

`wavexplain` instead measures real model predictions. Starting from a
fully-baselined input, it reveals named groups of the input in sequence and
records the actual prediction at each stage. The named contributions are
guaranteed to sum **exactly** to the true forecast, because every number is
a directly measured prediction, not an estimated allocation.

## Install

```bash
pip install wavexplain
```

(Or, until published: `pip install -e .` from a local clone.)

## Quickstart

```python
import numpy as np
import torch
from collections import OrderedDict
from wavexplain import MultiSeriesWaveNet, CounterfactualExplainer, render_card_html

# 1. Train (or load) a MultiSeriesWaveNet on your own panel data.
#    Input convention: (batch, 1 + num_covariates, time), channel 0 is
#    your target series, any other channels are covariates you define.
model = MultiSeriesWaveNet(num_series=1000, horizon=7, num_covariates=1)
model.load_state_dict(torch.load("your_checkpoint.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Build the input window you want explained: shape (channels, timesteps).
#    Apply whatever transform your model expects (e.g. log1p) yourself --
#    this library doesn't assume a specific transform.
full_input = np.stack([your_log_target_history, your_covariate_history])

# 3. Define named groups of timesteps to reveal, in the order you want
#    them attributed -- put the most "baseline" group first.
groups = OrderedDict([
    ("seasonal_pattern", your_typical_days_mask),   # bool array, shape (timesteps,)
    ("recent_trend", your_recent_days_mask),
    ("promotion_effect", your_promo_days_mask),
])

explainer = CounterfactualExplainer(
    model, series_id=42, device=device,
    output_transform=torch.expm1,  # e.g. if your model outputs log1p-space
)
contributions, baseline_pred, full_pred = explainer.explain(
    full_input, baseline_values=[0.0, 0.0], reveal_groups=groups
)

# contributions["seasonal_pattern"] + contributions["recent_trend"]
#   + contributions["promotion_effect"] + baseline_pred == full_pred  (exact)

# 4. Render a plain-language card.
render_card_html(
    title="Series 42",
    total_forecast=full_pred,
    contributions=contributions,
    baseline_prediction=baseline_pred,
    highlight_group="promotion_effect",
    output_path="forecast_card.html",
)
```

## What this library does not do

- It doesn't load or preprocess your data. Bring your own panel-building
  pipeline; `MultiSeriesWaveNet` only cares about tensor shapes.
- It doesn't claim causal discovery in the formal sense (no causal graph
  recovery). "Counterfactual" here means measuring the model's own response
  to a controlled input change, not identifying true causal structure in the
  underlying data-generating process.
- It doesn't validate that your groups are a sensible decomposition of the
  input -- that's a domain judgment only you can make.

## Development origin

This library grew out of extending a 2018 WaveNet-based sales forecasting
model with an interpretability layer. See the accompanying paper for the
full evaluation methodology, including a faithfulness test (deletion/
insertion, statistically significant at p < 0.0001 across 30 series) and an
honest analysis of when attribution to a specific covariate is and isn't
meaningful across a product panel.

## License

MIT
