# 🎰 Deterministic `flash-attn`
Sam Foreman
2024-06-17

<link rel="preconnect" href="https://fonts.googleapis.com">

> \[**NOTE**\]: For additional details, refer to the [W&B
> Report](https://api.wandb.ai/links/aurora_gpt/nqjjhzd5).

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
