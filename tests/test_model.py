"""
tests/test_model.py

Minimal smoke tests: confirm the model builds and trains on dummy data,
and that CounterfactualExplainer's contributions sum exactly to the
real prediction (the core correctness guarantee of this library).
"""

from collections import OrderedDict

import numpy as np
import torch

from wavexplain import MultiSeriesWaveNet, CounterfactualExplainer


def test_model_builds_and_trains():
    model = MultiSeriesWaveNet(num_series=10, horizon=4, num_covariates=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 2, 30)  # (batch, 1+num_covariates, time)
    series_id = torch.randint(0, 10, (8,))
    target = torch.randn(8, 4)

    pred = model(x, series_id)
    assert pred.shape == (8, 4)

    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()


def test_counterfactual_contributions_sum_exactly():
    torch.manual_seed(0)
    model = MultiSeriesWaveNet(num_series=5, horizon=3, num_covariates=1)
    model.eval()
    device = torch.device("cpu")

    input_length = 20
    full_input = np.random.randn(2, input_length).astype("float32")

    group_a = np.zeros(input_length, dtype=bool)
    group_a[:10] = True
    group_b = np.zeros(input_length, dtype=bool)
    group_b[10:] = True

    explainer = CounterfactualExplainer(model, series_id=2, device=device)
    contributions, baseline_pred, full_pred = explainer.explain(
        full_input,
        baseline_values=[0.0, 0.0],
        reveal_groups=OrderedDict([("group_a", group_a), ("group_b", group_b)]),
    )

    reconstructed = baseline_pred + sum(contributions.values())
    assert abs(reconstructed - full_pred) < 1e-4, (
        f"Contributions should sum exactly to the real prediction: "
        f"{reconstructed} != {full_pred}"
    )


if __name__ == "__main__":
    test_model_builds_and_trains()
    test_counterfactual_contributions_sum_exactly()
    print("All smoke tests passed.")
