# 🚧 Frameworks Issue with <code>numpy \> 2</code>
Sam Foreman
2025-05-03

<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://iosevka-webfonts.github.io/iosevka/iosevka.css" rel="stylesheet">

- [Reinstall Modules built with
  `numpy < 2`](#reinstall-modules-built-with-numpy--2)
- [Profiles and Timing Comparisons](#profiles-and-timing-comparisons)
  - [DEFAULT BEHAVIOR](#default-behavior)
  - [WITH `USE_TORCH=1` and
    `numpy==2.2.5`](#with-use_torch1-and-numpy225)
- [Stack Trace from `numpy > 2` issue](#stack-trace-from-numpy--2-issue)

Something I just learned

The TensorFlow that is included in the new frameworks module
(`aurora_nre_models_frameworks-2025.0.0`) was built with
`numpy==1.26.4 (< 2)`.

Unfortunately, if you then (for whatever reason) then tries to install /
upgrade a package[^1] that has `numpy` in its dependencies, e.g.:

``` bash
python3 -m pip install --upgrade transformers
```

This will pull in `numpy > 2`, effectively breaking the frameworks
module.

In particular, any application that uses
[`intel/extension_for_pytorch`](https://github.com/intel/intel-extension-for-pytorch):

``` python
import intel_extension_for_pytorch as ipex
```

Will crash with:

``` bash
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.5 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
which is obviously frustrating for an application that is only using PyTorch.
```

Following through the stack trace, I see that the error is actually
coming from
[huggingface/transformers/image_transforms.py#L47](https://github.com/huggingface/transformers/image_transforms.py#L47).

Digging around a bit more I found there is a flag in `transformers` that
allows you to bypass the entire `import tensorflow as tf` logic:

``` bash
USE_TORCH=1
```

which not only prevents things from crashing with `numpy > 2`, but is
also noticeably quicker.

## Reinstall Modules built with `numpy < 2`

In addition to `tensorflow`, it seems that: { `jax`, `jaxlib`,
`ml-dtypes`, `opt-einsum`, `scipy` } were *all* built with `numpy < 2`,
and so need to be rebuilt after upgrading numpy.

To do so:

``` bash
python3 -m pip install --upgrade  numpy jax jaxlib ml-dtypes opt-einsum scipy transformers
```

✅ Now, we’re able to successfully:

``` bash
#[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
#[05/03/25 @ 12:12:29][x4515c7s4b0n0][/f/d/f/p/s/ezpz][🌱 update-utils][📦🤷✓] [⏱️ 25s]
; USE_TORCH=1 python3 -c 'import numpy as np; print(np.__version__) ; import ezpz '
2.2.5
[W503 12:12:34.201673197 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
    registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
[2025-05-03 12:13:18][W][utils/_logger:68:ezpz] Unable to import deepspeed. Please install it to use DeepSpeed features.
took: 0h:00m:48s
```

## Profiles and Timing Comparisons

Below we present timing profiles obtained from

- [`hyperfine`](https://github.com/sharkdp/hyperfine): A command-line
  benchmarking tool
- [`pyinstrument`](https://github.com/joerick/pyinstrument): A call
  stack profiler for Python

Below we present the profiles obtained by the usual mechanism, i.e.

``` bash
module load frameworks
python3 -m venv --system-site-packages "venvs/$(basename ${CONDA_PREFIX})"
source venvs/$(basename ${CONDA_PREFIX})/bin/activate
python3 -m pip install -e "git+https://github.com/saforem2/ezpz"
```

### DEFAULT BEHAVIOR

##### Bash Profile (`hyperfine`)

``` bash
$ hyperfine --max-runs=10 --shell=zsh --show-output 'ezpz_init() { $(which python3) -c "import ezpz" }; ezpz_init'
# ...[clipped]...
Time (mean ± σ):     10.754 s ±  0.122 s    [User: 14.574 s, System: 12.795 s]
Range (min … max):   10.467 s … 10.910 s    10 runs
```

#### Python Profile (`pyinstrument`)

``` bash
$ pyinstrument -c 'import ezpz'
# ...[clipped]...
12.021 <module>  ezpz/__init__.py:1
├─ 8.624 <module>  ezpz/dist.py:1
│  └─ 8.542 <module>  intel_extension_for_pytorch/__init__.py:1
│        [206 frames hidden]  intel_extension_for_pytorch, transfor...
│           0.247 <module>  torch/utils/_sympy/functions.py:1
│           └─ 0.241 <module>  sympy/__init__.py:1
│              └─ 0.126 <module>  sympy/polys/__init__.py:1
```

### WITH `USE_TORCH=1` and `numpy==2.2.5`

Below we present the profiles and timing measurements obtained:

1.  After upgrading `numpy==2.25`

2.  Skipping the `import tensorflow as tf` logic in `transformers` by
    specifying `USE_TORCH=1`, explicitly.

#### Bash Profile (`hyperfine`)

``` bash
$ hyperfine --max-runs=10 --shell=zsh --show-output 'ezpz_init() { USE_TORCH=1 $(which python3) -c "import ezpz" }; ezpz_init'
# ...[clipped]...
Time (mean ± σ):      7.491 s ±  0.162 s    [User: 12.130 s, System: 11.940 s]
Range (min … max):    7.311 s …  7.883 s    10 runs
```

#### Python Profile (`pyinstrument`)

``` bash
$ USE_TORCH=1 pyinstrument -c 'import ezpz'
# ...[clipped]...
8.478 <module>  ezpz/__init__.py:1
├─ 5.109 <module>  ezpz/dist.py:1
│  └─ 5.016 <module>  intel_extension_for_pytorch/__init__.py:1
│        [174 frames hidden]  intel_extension_for_pytorch, transfor...
│           0.249 <module>  torch/utils/_sympy/functions.py:1
│           └─ 0.241 <module>  sympy/__init__.py:1
│              └─ 0.124 <module>  sympy/polys/__init__.py:1
```

## Stack Trace from `numpy > 2` issue

<details closed>

<summary>

Stack Trace
</summary>

``` bash
#[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
#[05/02/25 @ 15:46:43][x4005c2s6b0n0][/f/d/f/p/s/t/2/ezpz][🌱 update-utils][✓] [⏱️ 19s]
; ezpz-test --profile --tp 2 --pp 4
[W502 16:00:18.739960487 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
    registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.5 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/lus/flare/projects/datascience/foremans/projects/saforem2/tmp/2025-05-02-150121/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/ezpz-test", line 6, in <module>
    from ezpz.test import main
  File "/lus/flare/projects/datascience/foremans/projects/saforem2/tmp/2025-05-02-150121/ezpz/src/ezpz/__init__.py", line 102, in <module>
    from ezpz.dist import (
  File "/lus/flare/projects/datascience/foremans/projects/saforem2/tmp/2025-05-02-150121/ezpz/src/ezpz/dist.py", line 42, in <module>
    import intel_extension_for_pytorch as ipex  # type:ignore[missingTypeStubs]
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py", line 128, in <module>
    from . import xpu
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/xpu/__init__.py", line 20, in <module>
    from .utils import *
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/xpu/utils.py", line 7, in <module>
    from .. import frontend
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/frontend.py", line 9, in <module>
    from .nn import utils
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/nn/__init__.py", line 6, in <module>
    from .modules import FrozenBatchNorm2d
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/nn/modules/__init__.py", line 11, in <module>
    from ...cpu.nn.linear_fuse_eltwise import IPEXLinearEltwise
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/cpu/nn/linear_fuse_eltwise.py", line 3, in <module>
    from intel_extension_for_pytorch.nn.utils._weight_prepack import (
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/nn/utils/__init__.py", line 1, in <module>
    from intel_extension_for_pytorch.nn.utils import _weight_prepack
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/nn/utils/_weight_prepack.py", line 8, in <module>
    from intel_extension_for_pytorch.cpu.tpp.utils.blocked_layout import (
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/cpu/tpp/__init__.py", line 2, in <module>
    from . import fused_bert
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/intel_extension_for_pytorch/cpu/tpp/fused_bert.py", line 16, in <module>
    from transformers.modeling_utils import apply_chunking_to_forward
  File "/lus/flare/projects/datascience/foremans/projects/saforem2/tmp/2025-05-02-150121/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/transformers/modeling_utils.py", line 69, in <module>
    from .loss.loss_utils import LOSS_MAPPING
  File "/lus/flare/projects/datascience/foremans/projects/saforem2/tmp/2025-05-02-150121/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/transformers/loss/loss_utils.py", line 21, in <module>
    from .loss_deformable_detr import DeformableDetrForObjectDetectionLoss, DeformableDetrForSegmentationLoss
  File "/lus/flare/projects/datascience/foremans/projects/saforem2/tmp/2025-05-02-150121/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/transformers/loss/loss_deformable_detr.py", line 4, in <module>
    from ..image_transforms import center_to_corners_format
  File "/lus/flare/projects/datascience/foremans/projects/saforem2/tmp/2025-05-02-150121/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/transformers/image_transforms.py", line 47, in <module>
    import tensorflow as tf
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/__init__.py", line 48, in <module>
    from tensorflow._api.v2 import __internal__
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/_api/v2/__internal__/__init__.py", line 8, in <module>
    from tensorflow._api.v2.__internal__ import autograph
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py", line 8, in <module>
    from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/python/autograph/core/ag_ctx.py", line 21, in <module>
    from tensorflow.python.autograph.utils import ag_logging
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/python/autograph/utils/__init__.py", line 17, in <module>
    from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/python/autograph/utils/context_managers.py", line 19, in <module>
    from tensorflow.python.framework import ops
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/python/framework/ops.py", line 41, in <module>
    from tensorflow.python import pywrap_tfe
  File "/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/lib/python3.10/site-packages/tensorflow/python/pywrap_tfe.py", line 25, in <module>
    from tensorflow.python._pywrap_tfe import *
AttributeError: _ARRAY_API not found
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

# ...[clipped]...
```

</details>

[^1]: This is a lot of packages, including: { `torch`, `jax`,
    `tensorflow`, `scipy`, `jaxlib`, `numpy`, `ml-dtypes`, `opt-einsum`,
    …, }.
