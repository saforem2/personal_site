# AERIS: Argonne Earth Systems Model
Sam Foreman
2025-10-08

- [🌎 AERIS](#earth_americas-aeris)
- [High-Level Overview of AERIS](#high-level-overview-of-aeris)
- [Model Overview](#model-overview)
- [Windowed Self-Attention](#windowed-self-attention)
- [Model Architecture: Details](#model-architecture-details)
- [Limitations of Deterministic
  Models](#limitations-of-deterministic-models)
- [Transitioning to a Probabilistic
  Model](#transitioning-to-a-probabilistic-model)
- [Training at Scale](#training-at-scale)
- [Sequence-Window-Pipeline Parallelism
  `SWiPe`](#sequence-window-pipeline-parallelism-swipe)
- [Aurora](#aurora)
- [AERIS: Scaling Results](#aeris-scaling-results)
- [Hurricane Laura](#hurricane-laura)
- [Seasonal Forecast Stability](#seasonal-forecast-stability)
- [References](#references)

## 🌎 AERIS

<div class="flex-container">

<div class="flex-child">

<div id="fig-arxiv">

![](./assets/team.png)

Figure 1: [arXiv:2509.13523](https://arxiv.org/abs/2509.13523)

</div>

</div>

<div class="flex-child" style="width:43.6%;">

<img src="./assets/cover2.svg" class="light-content" />

<img src="./assets/cover1.svg" class="dark-content" />

</div>

</div>

## High-Level Overview of AERIS

<div class="flex-container">

<div id="fig-rollout">

![](./assets/rollout.gif)

Figure 2: Rollout of AERIS model, specific humidity at 700m.

</div>

<div id="tbl-aeris">

Table 1: Overview of AERIS model and training setup

|           Property | Description      |
|-------------------:|:-----------------|
|             Domain | Global           |
|         Resolution | 0.25° & 1.4°     |
|      Training Data | ERA5 (1979–2018) |
| Model Architecture | Swin Transformer |
|        Speedup[^1] | O(10k–100k)      |

</div>

</div>

## Model Overview

<div class="flex-container" style="align-items: flex-start;">

<div id="tbl-data-vars">

Table 2: Variables used in AERIS training and prediction

|   Variable   | Description                   |
|:------------:|:------------------------------|
|    `t2m`     | 2m Temperature                |
| `X` `u`(`v`) | $u$ ($v$) wind component @ Xm |
|     `q`      | Specific Humidity             |
|     `z`      | Geopotential                  |
|    `msl`     | Mean Sea Level Pressure       |
|    `sst`     | Sea Surface Temperature       |
|    `lsm`     | Land-sea mask                 |

</div>

<div class="flex-child">

- **Dataset**: ECMWF Reanalysis v5 (ERA5)
- **Variables**: Surface and pressure levels
- **Usage**: Medium-range weather forecasting
- **Partition**:
  - Train: 1979–2018[^2]
  - Val: 2019
  - Test: 2020
- **Data Size**: 100GB at 5.6° to 31TB at 0.25°

</div>

</div>

## Windowed Self-Attention

<div class="flex-container">

<div class="flex-child" style="width:33%;">

- **Benefits for weather modeling**:
  - Shifted windows capture both local patterns and long-range context
  - Constant scale, windowed self-attention provides high-resolution
    forecasts
  - Designed (currently) for fixed, 2D grids
- **Inspiration from SOTA LLMs**:
  - `RMSNorm`, `SwiGLU`, 2D `RoPE`

</div>

<div id="fig-windowed-self-attention">

![](./assets/swin-transformer.png)

Figure 3: Windowed Self-Attention

</div>

</div>

## Model Architecture: Details

<div id="fig-model-arch-details">

![](./assets/model_architecture.svg)

Figure 4: Model Architecture

</div>

## Limitations of Deterministic Models

<div class="flex-container">

<div class="flex-child">

- <span class="red-text"></span>
  <span class="highlight-red">**Transformers**</span>:
  - *Deterministic*
  - Single input → single forecast

</div>

<div class="flex-child">

- <span class="green-text"></span>
  <span class="highlight-green">**Diffusion**</span>:
  - *Probabilistic*
  - Single input → ***ensemble of forecasts***
  - Captures uncertainty and variability in weather predictions
  - Enables ensemble forecasting for better risk assessment

</div>

</div>

## Transitioning to a Probabilistic Model

<div id="fig-forward-pass">

![](./assets/diffusion/light.svg)

Figure 5: Reverse diffusion with the
<span style="color:#228be6">input</span> condition, individual sampling
steps $t_{0} \rightarrow t_{64}$, the next time step
<span style="color:#40c057">estimate</span> and the
<span style="color:#fa5252">target</span> output.

</div>

<div class="flex-container">

![](./assets/diffusion.gif)

![](./assets/diffusion_forward.png)

</div>

## Training at Scale

- 

## Sequence-Window-Pipeline Parallelism `SWiPe`

<div class="flex-container">

<div class="flex-child" style="width:33%;">

- `SWiPe` is a **novel parallelism strategy** for Swin-based
  Transformers
- Hybrid 3D Parallelism strategy, combining:
  - Sequence parallelism (`SP`)
  - Window parallelism (`WP`)
  - Pipeline parallelism (`PP`)

</div>

<div id="fig-swipe-layer">

![](./assets/wpsp.svg)

Figure 6

</div>

</div>

<div id="fig-comms">

![](./assets/comms1.svg)

Figure 7: `SWiPe` Communication Patterns

</div>

## Aurora

<div class="flex-container" style="align-items: center; gap:10pt;">

<div id="tbl-aurora">

Table 3: Aurora[^3] Specs

| Property | Value   |
|---------:|:--------|
|    Racks | 166     |
|    Nodes | 10,624  |
| XPUs[^4] | 127,488 |
|     CPUs | 21,248  |
|     NICs | 84,992  |
|      HBM | 8 PB    |
|    DDR5c | 10 PB   |

</div>

<div id="fig-aurora">

![](./assets/aurora1.png)

Figure 8: Aurora: [Fact
Sheet](https://www.alcf.anl.gov/sites/default/files/2024-07/Aurora_FactSheet_2024.pdf).

</div>

</div>

## AERIS: Scaling Results

<div class="flex-container">

<div id="fig-aeris-scaling">

![](./assets/aeris-scaling.svg)

Figure 9: AERIS: Scaling Results

</div>

<div class="column" style="width:30%;">

- <span class="highlight-blue">**10 EFLOPs**</span> (sustained) @
  **120,960 GPUs**
- See (Hatanpää et al. (2025)) for additional details
- [arXiv:2509.13523](https://arxiv.org/abs/2509.13523)

</div>

</div>

## Hurricane Laura

<div id="fig-hurricane-laura">

![](./assets/science/hurricane.png)

Figure 10: Hurricane Laura tracks (top) and intensity (bottom).
Initialized 7(a), 5(b) and 3(c) days prior to 2020-08-28T00z.

</div>

## Seasonal Forecast Stability

<div id="fig-seasonal-forecast-stability">

<img src="./assets/science/s2s.png" style="width:90.0%" />

Figure 11: S2S Stability: (a) Spring barrier El Niño with realistic
ensemble spread in the ocean; (b) qualitatively sharp fields of SST and
Q700 predicted 90 days in the future from the
<span style="color:#65B8EE;">closest</span> ensemble member to the ERA5
in (a); and (c) stable Hovmöller diagrams of U850 anomalies (climatology
removed; m/s), averaged between 10°S and 10°N, for a 90-day rollout.

</div>

## References

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-stock2025aeris" class="csl-entry">

Hatanpää, Väinö, Eugene Ku, Jason Stock, Murali Emani, Sam Foreman,
Chunyong Jung, Sandeep Madireddy, et al. 2025. “AERIS: Argonne Earth
Systems Model for Reliable and Skillful Predictions.”
<https://arxiv.org/abs/2509.13523>.

</div>

</div>

[^1]: Relative to PDE-based models, e.g.:
    [GFS](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs)

[^2]: ~ 14,000 days of data

[^3]: 🏆 [Aurora Supercomputer Ranks Fastest for
    AI](https://www.intel.com/content/www/us/en/newsroom/news/intel-powered-aurora-supercomputer-breaks-exascale-barrier.html)

[^4]: Each node has 6 Intel Data Center GPU Max 1550 (code-named “Ponte
    Vecchio”) tiles, with 2 XPUs per tile.
