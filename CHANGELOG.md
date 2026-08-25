# Changelog

## 0.1.0

Initial release.

The attribution approach in this release (`CounterfactualExplainer`) uses
sequential counterfactual masking, measuring real model predictions at each
reveal stage, rather than allocating shares of a raw attribution-value sum.

This design choice is not arbitrary: an earlier prototype allocated shares of
summed SHAP attribution values into named buckets, and it failed silently on
a real case during development. For a product promoted on nearly every day
in its input window, the "typical pattern" bucket -- built from summing
signed attribution over the handful of remaining non-promoted days -- landed
at exactly zero, despite the product having a perfectly normal 20-90 unit
baseline sales level. The zero was an artifact of too little data to sum
over and sign cancellation, not a real finding.

The counterfactual approach in this release cannot produce that failure
mode: every reported number is a real, directly measured model prediction,
and the reported contributions are guaranteed, by construction, to sum
exactly to the true forecast.
