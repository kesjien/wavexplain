"""
wavexplain.model

A shared, multi-series WaveNet-style forecaster: dilated causal
convolutions with gated activation (matching the original WaveNet
paper's gating scheme), conditioned on a learned per-series embedding
so one model can be trained across an entire panel of related time
series (e.g. many stores, sensors, or accounts) in ordinary mini-batches,
rather than fitting a separate model per series.

This is domain-agnostic: it doesn't know or care whether the series
represent retail sales, sensor readings, or anything else. Domain-
specific data loading belongs in your own code or in examples/, not
in this module.
"""

import torch
import torch.nn as nn


def _init_conv_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class GatedDilatedCausalConv1d(nn.Module):
    """
    One dilated causal residual block with gated activation (tanh *
    sigmoid). Genuinely causal: left-only zero-padding by
    dilation * (kernel_size - 1) guarantees no leakage from future
    timesteps, and preserves sequence length through the block.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)

        self.tanh_conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.sigm_conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.tanh_conv.apply(_init_conv_weights)
        self.sigm_conv.apply(_init_conv_weights)

        self.skip_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.skip_conv.apply(_init_conv_weights)

        self.res_proj = None
        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.res_proj.apply(_init_conv_weights)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x_padded = nn.functional.pad(x, (self.padding, 0))
        tanh_out = self.tanh(self.tanh_conv(x_padded))
        sigm_out = self.sigmoid(self.sigm_conv(x_padded))
        gated = tanh_out * sigm_out
        skip = self.skip_conv(gated)

        residual_input = x if self.res_proj is None else self.res_proj(x)
        return residual_input + skip, skip


class MultiSeriesWaveNet(nn.Module):
    """
    Single shared WaveNet-style forecaster over a panel of `num_series`
    related time series. A learned per-series embedding conditions the
    shared convolutional stack, so one model trains across the full
    panel in ordinary mini-batches.

    Input convention: (batch, 1 + num_covariates, input_length), where
    channel 0 is the target series and any additional channels are
    covariates you define (promotions, external signals, anything you
    have). What each channel means, and any transform (e.g. log1p) you
    apply before/after the model, is entirely up to your own code --
    this class only knows about tensor shapes.
    """

    def __init__(
        self,
        num_series: int,
        horizon: int,
        num_covariates: int = 0,
        series_embedding_dim: int = 8,
        filters: int = 32,
        kernel_size: int = 3,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.horizon = horizon
        self.series_embedding = nn.Embedding(num_series, series_embedding_dim)

        in_channels = 1 + num_covariates + series_embedding_dim
        self.blocks = nn.ModuleList()
        channels = in_channels
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(
                GatedDilatedCausalConv1d(channels, filters, kernel_size, dilation)
            )
            channels = filters

        self.output_conv1 = nn.Conv1d(filters, filters, kernel_size=1)
        self.output_conv2 = nn.Conv1d(filters, filters, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(filters, horizon)

    def forward(self, series_values: torch.Tensor, series_id: torch.Tensor):
        """
        series_values: (batch, 1 + num_covariates, time)
        series_id: (batch,) long tensor of series indices

        Returns: (batch, horizon) forecast tensor.
        """
        batch, _, T = series_values.shape

        embed = self.series_embedding(series_id)
        embed_tiled = embed.unsqueeze(-1).repeat(1, 1, T)
        x = torch.cat([series_values, embed_tiled], dim=1)

        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        out = self.relu(skip_sum)
        out = self.relu(self.output_conv1(out))
        out = self.output_conv2(out)
        out = out.mean(dim=-1)
        out = self.fc(out)
        return out


class FixedSeriesWrapper(nn.Module):
    """
    Wraps any (series_values, series_id) -> prediction model with a
    fixed series_id, so it can be called as a plain single-input
    function -- used by CounterfactualExplainer, and generally useful
    any time you want to treat "which series" as fixed context rather
    than a per-call argument.
    """

    def __init__(self, model: nn.Module, series_id: int, device: torch.device):
        super().__init__()
        self.model = model
        self.series_id = series_id
        self.device = device

    def forward(self, series_values: torch.Tensor) -> torch.Tensor:
        batch = series_values.shape[0]
        series_id_batch = torch.full(
            (batch,), self.series_id, dtype=torch.long, device=self.device
        )
        return self.model(series_values, series_id_batch)
