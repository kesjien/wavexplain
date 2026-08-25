"""
wavexplain.cards

Renders plain-language "forecast driver cards" from a
CounterfactualExplainer result -- one number, one plain-English
sentence, and a short breakdown, instead of a per-timestep attribution
plot only a technical audience can read.

Domain-agnostic: pass in whatever named contributions you computed with
CounterfactualExplainer, and this renders them. It doesn't know or care
whether your groups are "promotion" and "seasonal trend" or something
else entirely.
"""

from collections import OrderedDict


def render_card_html(
    title: str,
    total_forecast: float,
    contributions: "OrderedDict[str, float]",
    baseline_prediction: float,
    baseline_label: str = "Typical pattern",
    highlight_group: str = None,
    unit_label: str = "units",
    output_path: str = None,
):
    """
    Renders a standalone HTML forecast card. Self-contained inline CSS,
    no external dependencies -- opens in any browser.

    title: short header text (e.g. "Store 47 - Item 1392260").
    total_forecast: the real, fully-revealed forecast value.
    contributions: OrderedDict {group_name: contribution_value} from
        CounterfactualExplainer.explain(). The baseline_prediction is
        folded into the first row (baseline_label) automatically.
    baseline_prediction: the explainer's baseline_prediction value.
    baseline_label: display label for the combined
        baseline + first-group row.
    highlight_group: name of a group (matching a key in `contributions`)
        to badge and call out in the explanation sentence, if it's the
        largest contributor. If None, no badge is shown.
    unit_label: unit suffix shown after numbers (e.g. "units", "$",
        "sessions").

    Returns the HTML string; also writes it to `output_path` if given.
    """
    names = list(contributions.keys())
    values = list(contributions.values())

    # Fold the baseline into the first contribution group for display,
    # since most users don't need "baseline" as a separate concept.
    display_rows = []
    if names:
        display_rows.append((baseline_label, baseline_prediction + values[0]))
        for name, val in zip(names[1:], values[1:]):
            display_rows.append((name.replace("_", " ").capitalize(), val))
    else:
        display_rows.append((baseline_label, baseline_prediction))

    max_abs = max(abs(v) for _, v in display_rows) or 1

    largest_group = None
    if len(names) > 0:
        largest_group = names[int(max(range(len(values)), key=lambda i: abs(values[i])))]

    show_badge = highlight_group is not None and largest_group == highlight_group
    badge_label = highlight_group.replace("_", " ").capitalize() + "-driven" if show_badge else "Standard pattern"
    badge_bg = "#FAEEDA" if show_badge else "#F1EFE8"
    badge_text = "#854F0B" if show_badge else "#5F5E5A"
    explanation = (
        f"Mostly explained by {highlight_group.replace('_', ' ')}."
        if show_badge
        else "Follows this series' typical pattern."
    )

    def bar_width(v):
        return max(2, round(abs(v) / max_abs * 100))

    rows_html = ""
    for i, (label, val) in enumerate(display_rows):
        color = "#EF9F27" if (show_badge and i > 0 and label.lower().startswith(
            highlight_group.replace("_", " "))) else "#888780"
        sign = "+" if val >= 0 and i > 0 else ""
        rows_html += f"""
  <div class="row">
    <div class="row-label"><span class="muted">{label}</span><span>{sign}{round(val)}</span></div>
    <div class="bar-track"><div class="bar-fill" style="width:{bar_width(val)}%; background:{color};"></div></div>
  </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background: #f1efe8; padding: 40px; }}
  .card {{ background: #ffffff; border-radius: 12px; border: 1px solid #e0ded4; padding: 24px 28px; max-width: 420px; margin: 0 auto; }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px; }}
  .muted {{ color: #5f5e5a; font-size: 13px; }}
  .badge {{ background: {badge_bg}; color: {badge_text}; font-size: 12px; padding: 3px 10px; border-radius: 6px; font-weight: 600; }}
  .big-number {{ font-size: 32px; font-weight: 600; margin: 4px 0; }}
  .explanation {{ font-size: 14px; color: #444441; margin-bottom: 20px; }}
  .section-title {{ font-size: 13px; color: #5f5e5a; font-weight: 600; margin-bottom: 12px; }}
  .row {{ margin-bottom: 10px; }}
  .row-label {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }}
  .bar-track {{ background: #f1efe8; border-radius: 4px; height: 8px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .total-row {{ display: flex; justify-content: space-between; align-items: center; margin-top: 16px; padding-top: 12px; border-top: 1px solid #e0ded4; }}
  .footnote {{ font-size: 12px; color: #888780; margin-top: 16px; }}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <p class="muted">{title}</p>
    <span class="badge">{badge_label}</span>
  </div>
  <p class="muted">Forecast</p>
  <p class="big-number">{round(total_forecast)} {unit_label}</p>
  <p class="explanation">{explanation}</p>
  <p class="section-title">What's driving this number</p>
  {rows_html}
  <div class="total-row">
    <span class="muted">Total forecast</span>
    <span style="font-weight:600; font-size:16px;">{round(total_forecast)} {unit_label}</span>
  </div>
  <p class="footnote">Breakdown is model-generated and may not capture every factor.</p>
</div>
</body>
</html>"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)

    return html
