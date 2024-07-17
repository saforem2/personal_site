# 🎰 Deterministic `flash-attn`
<span class="dim-text">Sam Foreman</span>
[<span class="orcid-green"></span>](https://orcid.org/0000-0002-9981-0876)
2024-06-17

<!-- ::: {.callout-tip icon="false" title="📒 [W\&B Report](https://api.wandb.ai/links/aurora_gpt/nqjjhzd5)" collapse="true" style="width:100%;background-color: rgba(0,0,0,0.0);border:1px solid rgba(131,131,131,0.2)!important;"} -->

> \[**NOTE**\]: For additional details, refer to the [W&B
> Report](https://api.wandb.ai/links/aurora_gpt/nqjjhzd5).

<!-- ::: -->

Simple tests to confirm the loss is exactly reproducible across
independent runs (when launched with the same seed).

- In particular, we set:

  ``` python
  output = flash_attn_func(q, k, v, None, self.causal, deterministic=True)
  ```

  in all the `flash_attn_func(...)` calls from
  [`megatron/model/transformer.py`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/megatron/model/transformer.py)

- All experiments ran on Polaris @ ALCF, using:

  ``` yaml
  machine: Polaris
  args.zero_stage: 1
  args.num_layers: 32
  args.micro_batch_size: 1
  args.optimizer: "adamw"
  args.use_flash_attn: true
  env.DFL_STEM: "books"
  env.GRAD_ACC_STEPS: 8
  env.WORLD_SIZE: 8
  ```

<div id="fig-loss">

![](./assets/deterministic-flash-attn-loss.svg)


Figure 1: Plot of the loss curve for 3 independent runs with
`deterministic=True`

</div>
<!-- <iframe src="https://wandb.ai/aurora_gpt/AuroraGPT/workspace?nw=nwuserforemans" style="border:none;width:100%"> -->
<!-- <iframe src="https://wandb.ai/aurora_gpt/AuroraGPT/reports/Deterministic-Flash-Attention--Vmlldzo4MzU0OTQ0?accessToken=y1r4ftxjkxhhgpwhttag4g0rgqhxvqpi0hc1fbribk97brozbtjhrkbtc8wqkh5r" style="border:none;height:1024px;width:100%"> -->
