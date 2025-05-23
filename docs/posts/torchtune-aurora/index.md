# 🪛 Torchtune on Aurora
Sam Foreman
2025-03-23

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Patch on Aurora](#patch-on-aurora)
- [Using PyTorch 2.5](#using-pytorch-25)
- [Using PyTorch 2.3](#using-pytorch-23)

## Patch on Aurora

``` diff
diff --git a/torchtune/training/_distributed.py b/torchtune/training/_distributed.py
index ff959c5f..c3966290 100644
--- a/torchtune/training/_distributed.py
+++ b/torchtune/training/_distributed.py
@@ -14,7 +14,11 @@ import torch
 import torch.distributed as dist
 from torch import nn

-from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
+try:
+    from torch.distributed._composable.fsdp import fully_shard
+except (ImportError, ModuleNotFoundError):
+    from torch.distributed._composable.fsdp.fully_shard import fully_shard
+
 from torch.distributed._tensor import distribute_tensor, DTensor
 from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
 from torch.distributed.checkpoint.state_dict import (
@@ -532,6 +536,11 @@ def shard_model(
     """
     fsdp_kwargs = {"reshard_after_forward": reshard_after_forward}
     if cpu_offload:
+        try:
+            from torch.distributed._composable.fsdp import CPUOffloadPolicy
+        except (ImportError, ModuleNotFoundError):
+            from torch.distributed._composable.fsdp._fsdp_api import MixedPrecisionPolicy, CPUOffloadPolicy
+            # from torch.distributed._composable import CPUOffloadPolicy
         fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

     # Shard the model with FSDP, iterating in reverse to start with
```

## Using PyTorch 2.5

Using the new enviroment with `pytorch==2.5` and after applying the
above patch, I am almost able to get things working before ultimately
crashing with:

``` bash
[rank0]: RuntimeError: Tried to instantiate dummy base class Stream
```

The full command (and output) are included below:

- Command:

  ``` bash
  #[🐍 anl_2024_12_release_2](👻 anl_2024_12_release_2)
  #[08:47:46 AM][x4204c5s6b0n0][/f/A/f/p/p/torchtune][🌱 main][!?][⏱️ 2s]
  $ tune run full_finetune_distributed --config llama3_1/8B_full optimizer.fused=False
  ```

- Output:

  <details closed>

  <summary>

  Output:

  </summary>

  ``` bash
  [W128 08:49:19.162113524 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
  registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
  new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
  [2025-01-28 08:49:37][I][ezpz/dist:823] Using device='xpu' with backend='DDP' + 'gloo' for distributed training.
  [2025-01-28 08:49:37][I][ezpz/dist:869] ['x4204c5s6b0n0'][0/0]
  [2025-01-28 08:49:37][I][config/_utils:28:torchtune.utils._logging] Running FullFinetuneRecipeDistributed with resolved config:

  batch_size: 2
  checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: Meta-Llama-3.1-8B-Instruct/
  checkpoint_files:
  - model-00001-of-00004.safetensors
  - model-00002-of-00004.safetensors
  - model-00003-of-00004.safetensors
  - model-00004-of-00004.safetensors
  model_type: LLAMA3
  output_dir: /tmp/torchtune/llama3_1_8B/full
  recipe_checkpoint: null
  clip_grad_norm: null
  compile: false
  custom_sharded_layers:
  - tok_embeddings
  - output
  dataset:
  _component_: torchtune.datasets.alpaca_dataset
  packed: false
  device: xpu
  dtype: bf16
  enable_activation_checkpointing: false
  enable_activation_offloading: false
  epochs: 1
  gradient_accumulation_steps: 1
  log_every_n_steps: 1
  log_peak_memory_stats: true
  loss:
  _component_: torchtune.modules.loss.CEWithChunkedOutputLoss
  max_steps_per_epoch: null
  metric_logger:
  _component_: torchtune.training.metric_logging.DiskLogger
  log_dir: /tmp/torchtune/llama3_1_8B/full/logs
  model:
  _component_: torchtune.models.llama3_1.llama3_1_8b
  optimizer:
  _component_: torch.optim.AdamW
  fused: false
  lr: 2.0e-05
  optimizer_in_bwd: false
  output_dir: /tmp/torchtune/llama3_1_8B/full
  profiler:
  _component_: torchtune.training.setup_torch_profiler
  active_steps: 2
  cpu: true
  cuda: true
  enabled: false
  num_cycles: 1
  output_dir: /tmp/torchtune/llama3_1_8B/full/profiling_outputs
  profile_memory: false
  record_shapes: true
  wait_steps: 5
  warmup_steps: 3
  with_flops: false
  with_stack: false
  resume_from_checkpoint: false
  seed: null
  shuffle: true
  tokenizer:
  _component_: torchtune.models.llama3.llama3_tokenizer
  max_seq_len: null
  path: Meta-Llama-3.1-8B-Instruct/original/tokenizer.model

  [2025-01-28 08:49:37][I][recipes/full_finetune_distributed:141:__main__] log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False.
  /lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/training/checkpointing/_checkpoint_client.py:75: FutureWarning: get_world_size_and_rank is deprecated and will be removed in future versions. `get_world_size_and_rank` will move to `torchtune.utils._device` in future releases. Please use `torchtune.utils.get_world_size_and_rank` instead.
  _, self._rank = training.get_world_size_and_rank()
  [2025-01-28 08:49:37][D][training/seed:60:torchtune.utils._logging] Setting manual seed to local seed 4028640460. Local seed is seed + rank = 4028640460 + 0
  Writing logs to /tmp/torchtune/llama3_1_8B/full/logs/log_1738075777.txt
  [2025-01-28 08:49:39][I][recipes/full_finetune_distributed:499:__main__] FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
  /lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/tensor/_random.py:45: UserWarning: DTensor random operators may not have complete support on cpu device mesh
  warnings.warn(
  [2025-01-28 08:49:53][I][recipes/full_finetune_distributed:568:__main__] Instantiating model and loading checkpoint took 13.79 secs
  [2025-01-28 08:49:53][I][training/memory:301:torchtune.utils._logging] Memory stats after model init:
  XPU peak memory allocation: 1.04 GiB
  XPU peak memory reserved: 1.14 GiB
  XPU peak memory active: 1.04 GiB
  [2025-01-28 08:49:53][I][recipes/full_finetune_distributed:632:__main__] Optimizer is initialized.
  [2025-01-28 08:49:53][I][recipes/full_finetune_distributed:317:__main__] Loss is initialized.
  [2025-01-28 08:49:55][I][recipes/full_finetune_distributed:685:__main__] Dataset and Sampler are initialized.
  [2025-01-28 08:49:55][I][recipes/full_finetune_distributed:382:__main__] No learning rate scheduler configured. Using constant learning rate.
  [2025-01-28 08:49:55][W][training/_profiler:53:torchtune.utils._logging]  Profiling disabled.
  [2025-01-28 08:49:55][I][recipes/full_finetune_distributed:467:__main__]  Profiler config after instantiation: {'enabled': False}
  0%|                                                                                                                                                          | 0/26001 [00:00<?, ?it/s][rank0]: Traceback (most recent call last):
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/venvs/anl_2024_12_release_2/bin/tune", line 8, in <module>
  [rank0]:     sys.exit(main())
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/tune.py", line 49, in main
  [rank0]:     parser.run(args)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/tune.py", line 43, in run
  [rank0]:     args.func(args)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/run.py", line 214, in _run_cmd
  [rank0]:     self._run_single_device(args, is_builtin=is_builtin)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/run.py", line 108, in _run_single_device
  [rank0]:     runpy.run_path(str(args.recipe), run_name="__main__")
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/runpy.py", line 289, in run_path
  [rank0]:     return _run_module_code(code, init_globals, run_name,
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/runpy.py", line 96, in _run_module_code
  [rank0]:     _run_code(code, mod_globals, init_globals,
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/runpy.py", line 86, in _run_code
  [rank0]:     exec(code, run_globals)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/recipes/full_finetune_distributed.py", line 928, in <module>
  [rank0]:     sys.exit(recipe_main())
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/config/_parse.py", line 99, in wrapper
  [rank0]:     sys.exit(recipe_main(conf))
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/recipes/full_finetune_distributed.py", line 923, in recipe_main
  [rank0]:     recipe.train()
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/recipes/full_finetune_distributed.py", line 749, in train
  [rank0]:     logits = self._model(**batch)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
  [rank0]:     return self._call_impl(*args, **kwargs)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1844, in _call_impl
  [rank0]:     return inner()
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1769, in inner
  [rank0]:     args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 66, in fsdp_hook_wrapper
  [rank0]:     return torch._dynamo.disable(func, recursive=True)(*args, **kwargs)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 632, in _fn
  [rank0]:     return fn(*args, **kwargs)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 224, in _pre_forward
  [rank0]:     args, kwargs = self._root_pre_forward(module, args, kwargs)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 117, in _root_pre_forward
  [rank0]:     self._lazy_init()
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 175, in _lazy_init
  [rank0]:     self._init_shared_state()
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 183, in _init_shared_state
  [rank0]:     self._comm_ctx.lazy_init()
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/micromamba/envs/anl_2024_12_release_2/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_param_group.py", line 48, in lazy_init
  [rank0]:     raise RuntimeError("FSDP requires CUDA for streams")
  [rank0]: RuntimeError: FSDP requires CUDA for streams
  0%|                                                                                                                                                          | 0/26001 [00:00<?, ?it/s]
  [1]    135144 exit 1     tune run full_finetune_distributed --config llama3_1/8B_full
  took: 0h:00m:55s
  ```

  </details>

## Using PyTorch 2.3

- Command:

  ``` bash
  #[🐍 aurora_nre_models_frameworks-2024.2.1_u1](:ghost: aurora_nre_models_frameworks-2024.2.1_u1)
  #[08:07:29 AM][x4204c5s6b0n0][/f/A/f/p/p/torchtune][:seedling: main][!?][:stopwatch: 13s]
  $ tune run full_finetune_distributed --config llama4_1/8B_full optimizer.fused=False
  ```

- Output:

  <details closed>

  <summary>

  Output:

  </summary>

  ``` bash
  [2025-01-28 08:07:38][I][ezpz/dist:823] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
  [2025-01-28 08:07:38][I][ezpz/dist:869] ['x4204c5s6b0n0'][0/0]
  2025:01:28-08:07:38:(115657) |CCL_WARN| did not find MPI-launcher specific variables, switch to ATL/OFI, to force enable ATL/MPI set CCL_ATL_TRANSPORT=mpi
  2025:01:28-08:07:38:(115657) |CCL_WARN| value of CCL_LOCAL_RANK changed to be 0 (default:-1)
  2025:01:28-08:07:38:(115657) |CCL_WARN| value of CCL_LOCAL_SIZE changed to be 1 (default:-1)
  2025:01:28-08:07:38:(115657) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be none (default:hydra)
  [2025-01-28 08:07:38][I][config/_utils:28:torchtune.utils._logging] Running FullFinetuneRecipeDistributed with resolved config:

  batch_size: 2
  checkpointer:
    _component_: torchtune.training.FullModelHFCheckpointer
    checkpoint_dir: Meta-Llama-3.1-8B-Instruct/
    checkpoint_files:
    - model-00001-of-00004.safetensors
    - model-00002-of-00004.safetensors
    - model-00003-of-00004.safetensors
    - model-00004-of-00004.safetensors
    model_type: LLAMA3
    output_dir: /tmp/torchtune/llama3_1_8B/full
    recipe_checkpoint: null
  clip_grad_norm: null
  compile: false
  custom_sharded_layers:
  - tok_embeddings
  - output
  dataset:
    _component_: torchtune.datasets.alpaca_dataset
    packed: false
  device: xpu
  dtype: bf16
  enable_activation_checkpointing: false
  enable_activation_offloading: false
  epochs: 1
  gradient_accumulation_steps: 1
  log_every_n_steps: 1
  log_peak_memory_stats: true
  loss:
    _component_: torchtune.modules.loss.CEWithChunkedOutputLoss
  max_steps_per_epoch: null
  metric_logger:
    _component_: torchtune.training.metric_logging.DiskLogger
    log_dir: /tmp/torchtune/llama3_1_8B/full/logs
  model:
    _component_: torchtune.models.llama3_1.llama3_1_8b
  optimizer:
    _component_: torch.optim.AdamW
    fused: false
    lr: 2.0e-05
  optimizer_in_bwd: false
  output_dir: /tmp/torchtune/llama3_1_8B/full
  profiler:
    _component_: torchtune.training.setup_torch_profiler
    active_steps: 2
    cpu: true
    cuda: true
    enabled: false
    num_cycles: 1
    output_dir: /tmp/torchtune/llama3_1_8B/full/profiling_outputs
    profile_memory: false
    record_shapes: true
    wait_steps: 5
    warmup_steps: 3
    with_flops: false
    with_stack: false
  resume_from_checkpoint: false
  seed: null
  shuffle: true
  tokenizer:
    _component_: torchtune.models.llama3.llama3_tokenizer
    max_seq_len: null
    path: Meta-Llama-3.1-8B-Instruct/original/tokenizer.model

  [2025-01-28 08:07:38][I][recipes/full_finetune_distributed:141:__main__] log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False.
  /lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/training/checkpointing/_checkpoint_client.py:75: FutureWarning: get_world_size_and_rank is deprecated and will be removed in future versions. `get_world_size_and_rank` will move to `torchtune.utils._device` in future releases. Please use `torchtune.utils.get_world_size_and_rank` instead.
    _, self._rank = training.get_world_size_and_rank()
  [2025-01-28 08:07:38][D][training/seed:60:torchtune.utils._logging] Setting manual seed to local seed 2201534845. Local seed is seed + rank = 2201534845 + 0
  Writing logs to /tmp/torchtune/llama3_1_8B/full/logs/log_1738073258.txt
  [2025-01-28 08:07:39][I][recipes/full_finetune_distributed:499:__main__] FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
  /opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/distributed/_tensor/random.py:36: UserWarning: DTensor random operators may not have complete support on cpu device mesh
    warnings.warn(
  [2025-01-28 08:07:45][I][recipes/full_finetune_distributed:568:__main__] Instantiating model and loading checkpoint took 5.74 secs
  [2025-01-28 08:07:45][I][training/memory:301:torchtune.utils._logging] Memory stats after model init:
          XPU peak memory allocation: 1.04 GiB
          XPU peak memory reserved: 1.14 GiB
          XPU peak memory active: 1.04 GiB
  [2025-01-28 08:07:45][I][recipes/full_finetune_distributed:632:__main__] Optimizer is initialized.
  [2025-01-28 08:07:45][I][recipes/full_finetune_distributed:317:__main__] Loss is initialized.
  README.md: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.47k/7.47k [00:00<00:00, 85.6MB/s]
  (…)-00000-of-00001-a09b74b3ef9c3b56.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 24.2M/24.2M [00:00<00:00, 105MB/s]
  Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52002/52002 [00:01<00:00, 45116.86 examples/s]
  [2025-01-28 08:07:49][I][recipes/full_finetune_distributed:685:__main__] Dataset and Sampler are initialized.
  [2025-01-28 08:07:49][I][recipes/full_finetune_distributed:382:__main__] No learning rate scheduler configured. Using constant learning rate.
  [2025-01-28 08:07:49][W][training/_profiler:53:torchtune.utils._logging]  Profiling disabled.
  [2025-01-28 08:07:49][I][recipes/full_finetune_distributed:467:__main__]  Profiler config after instantiation: {'enabled': False}
    0%|                                                                                                                                                                                                                                                       | 0/26001 [00:00<?, ?it/s][rank0]: Traceback (most recent call last):
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/tune", line 10, in <module>
  [rank0]:     sys.exit(main())
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/tune.py", line 49, in main
  [rank0]:     parser.run(args)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/tune.py", line 43, in run
  [rank0]:     args.func(args)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/run.py", line 214, in _run_cmd
  [rank0]:     self._run_single_device(args, is_builtin=is_builtin)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/_cli/run.py", line 108, in _run_single_device
  [rank0]:     runpy.run_path(str(args.recipe), run_name="__main__")
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/runpy.py", line 289, in run_path
  [rank0]:     return _run_module_code(code, init_globals, run_name,
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/runpy.py", line 96, in _run_module_code
  [rank0]:     _run_code(code, mod_globals, init_globals,
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/runpy.py", line 86, in _run_code
  [rank0]:     exec(code, run_globals)
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/recipes/full_finetune_distributed.py", line 928, in <module>
  [rank0]:     sys.exit(recipe_main())
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/torchtune/config/_parse.py", line 99, in wrapper
  [rank0]:     sys.exit(recipe_main(conf))
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/recipes/full_finetune_distributed.py", line 923, in recipe_main
  [rank0]:     recipe.train()
  [rank0]:   File "/lus/flare/projects/Aurora_deployment/foremans/projects/pytorch/torchtune/recipes/full_finetune_distributed.py", line 749, in train
  [rank0]:     logits = self._model(**batch)
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
  [rank0]:     return self._call_impl(*args, **kwargs)
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1561, in _call_impl
  [rank0]:     args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 153, in _pre_forward
  [rank0]:     args, kwargs = self._root_pre_forward(module, args, kwargs)
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 67, in _root_pre_forward
  [rank0]:     self._lazy_init()
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 112, in _lazy_init
  [rank0]:     self._init_shared_state()
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_state.py", line 120, in _init_shared_state
  [rank0]:     self._comm_ctx.init()
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/distributed/_composable/fsdp/_fsdp_param_group.py", line 49, in init
  [rank0]:     self.all_gather_copy_in_stream = torch.cuda.Stream(priority=high_priority)
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/cuda/streams.py", line 34, in __new__
  [rank0]:     return super().__new__(cls, priority=priority, **kwargs)
  [rank0]:   File "/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/_utils.py", line 907, in err_fn
  [rank0]:     raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")
  [rank0]: RuntimeError: Tried to instantiate dummy base class Stream
    0%|                                                                                                                                                                                                                                                       | 0/26001 [00:00<?, ?it/s]
  [1]    115657 exit 1     tune run full_finetune_distributed --config llama3_1/8B_full
  ```

  </details>
