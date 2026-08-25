"""
wavexplain.attribution

Generic sequential counterfactual explainer for multi-channel,
multi-series forecasters (built for MultiSeriesWaveNet, but works with
any model matching the same (series_values, series_id) -> prediction
signature).

Unlike attribution methods that allocate shares of a raw attribution
sum (which can collapse to near-zero for a group with very few or
sign-cancelling data points -- a real failure mode found during
development, see CHANGELOG), this measures actual model predictions at
each reveal stage. Named contributions are guaranteed to sum EXACTLY
to the true forecast, because they are differences between real,
directly-measured predictions, not estimated allocations.
"""

from collections import OrderedDict

import numpy as np
import torch

from .model import FixedSeriesWrapper


class CounterfactualExplainer:
    """
    Decomposes one forecast into named, user-defined contribution
    groups, by revealing groups of input positions in a specified
    order, starting from a fully-baselined input.

    Example:
        explainer = CounterfactualExplainer(model, series_id=42, device=device)
        contributions, baseline_pred, full_pred = explainer.explain(
            full_input=my_input_tensor,          # (channels, timesteps)
            baseline_values=[0.0, 0.0],           # one baseline per channel
            reveal_groups={
                "seasonal_pattern": typical_mask,  # (timesteps,) or (channels, timesteps) bool
                "recent_trend": recent_mask,
                "promotion_effect": promo_mask,
            },
        )
        # contributions["promotion_effect"] + contributions["recent_trend"]
        #   + contributions["seasonal_pattern"] + baseline_pred == full_pred (exactly)
    """

    def __init__(self, model, series_id: int, device,
                 output_transform=None, input_is_already_transformed=True):
        """
        model: a trained model with signature
            model(series_values, series_id_tensor) -> prediction_tensor
        series_id: integer ID of the series being explained
        device: torch device
        output_transform: optional callable applied to the raw model
            output before computing contributions (e.g.
            `lambda x: torch.expm1(x.clamp(min=0))` if your model was
            trained on log1p-transformed targets and outputs shouldn't
            go negative before the inverse transform). Bake any
            clamping or clipping you need into this callable -- this
            class applies it as-is and doesn't clamp anything itself,
            to stay domain-agnostic. Defaults to identity (no transform).
        input_is_already_transformed: if your model expects transformed
            inputs (e.g. log1p-sales), pass full_input already in that
            space -- this class does not transform inputs for you, to
            stay domain-agnostic. Kept as a flag purely as a reminder;
            has no effect on computation.
        """
        self.wrapper = FixedSeriesWrapper(model, series_id, device).to(device)
        self.device = device
        self.output_transform = output_transform or (lambda x: x)

    def explain(self, full_input, baseline_values, reveal_groups: "OrderedDict[str, np.ndarray]",
                output_index: int = 0):
        """
        full_input: (channels, timesteps) array or tensor -- the real,
            fully-observed input.
        baseline_values: sequence of length `channels`, the reference
            value for each channel when "not revealed."
        reveal_groups: an OrderedDict mapping group_name -> boolean mask
            (shape (timesteps,), broadcast across channels, or shape
            (channels, timesteps) for per-channel masks). Groups are
            revealed cumulatively in the given order -- put the group
            you consider most "baseline-like" first.
        output_index: which element of the model's output vector to
            explain (e.g. which forecast horizon day, if the model
            outputs a multi-step forecast).

        Returns:
            contributions: dict {group_name: float}, in the same order
                as reveal_groups, the incremental prediction change from
                revealing that group.
            baseline_prediction: float, prediction with nothing revealed.
            full_prediction: float, the real, fully-revealed prediction.

        Guarantee: baseline_prediction + sum(contributions.values())
        equals full_prediction exactly (up to floating point precision),
        since every value is a real measured prediction, not an
        estimated allocation.
        """
        full_input_t = torch.as_tensor(full_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        channels, timesteps = full_input.shape[-2], full_input.shape[-1]

        def predict(revealed_mask):
            x = full_input_t.clone()
            for c in range(channels):
                x[0, c, :] = baseline_values[c]
            mask = np.asarray(revealed_mask)
            if mask.ndim == 1:
                mask = np.broadcast_to(mask, (channels, timesteps))
            idx_c, idx_t = np.where(mask)
            for c, t in zip(idx_c, idx_t):
                x[0, c, t] = full_input_t[0, c, t]
            with torch.no_grad():
                raw_pred = self.wrapper(x)
            transformed = self.output_transform(raw_pred)
            return float(transformed[0, output_index].cpu())

        cumulative_mask = np.zeros((channels, timesteps), dtype=bool)
        baseline_prediction = predict(cumulative_mask)

        contributions = OrderedDict()
        prev_prediction = baseline_prediction
        for name, mask in reveal_groups.items():
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 1:
                mask_arr = np.broadcast_to(mask_arr, (channels, timesteps))
            cumulative_mask = cumulative_mask | mask_arr
            pred = predict(cumulative_mask)
            contributions[name] = pred - prev_prediction
            prev_prediction = pred

        full_prediction = predict(np.ones((channels, timesteps), dtype=bool))

        return contributions, baseline_prediction, full_prediction
