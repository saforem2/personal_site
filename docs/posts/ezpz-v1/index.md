# ezpz-v1
Sam Foreman
2024-08-23

- [🍋 `ezpz` v1](#lemon-ezpz-v1)
  - [👀 Overview](#eyes-overview)
- [`test_dist.py`](#test_distpy)
- [ezpz](#ezpz)
  - [Setup](#setup)
  - [Tests](#tests)
  - [Helper Utilities](#helper-utilities)
- [Complete Example](#complete-example)
  - [Gist](#gist)
  - [📦 Clone Repo(s)](#package-clone-repos)
  - [🛜 Setup Job](#wireless-setup-job)
  - [🐍 Setup Python](#snake-setup-python)
  - [📝 Test Setup](#pencil-test-setup)

## 🍋 `ezpz` v1

[Sam Foreman](https://samforeman.me)  
*2024-05-14*

### 👀 Overview

**`ezpz` 🍋**

Launch, train and communicate across all your accelerators, `ezpz`.

*Full support for your favorite framework + backend combo ❤️*.

`ezpz` simplifies the process of:

- Setting up + launching distributed training:

  - `import ezpz as ez`
    - `RANK =`
      [`ez.setup_torch(backend=backend)`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L551)
      <span class="dim-text">for `backend` $\in$ {`DDP`, `deepspeed`,
      `horovod`}</span>

    - `RANK =`
      [`ez.get_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#396)

    - `LOCAL_RANK =`
      [`ez.get_local_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#448)

    - `WORLD_SIZE =`
      [`ez.get_world_size()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L417)

    <span class="dim-text"> (see
    [`ezpz/dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py)
    for more details). </span>

- Writing device agnostic code:

  - <a href="https://github.com/saforem3/ezpz/blob/main/src/ezpz/dist.py#L332"><code>ezpz.get_torch_device()</code></a>
    \> \> - **Full support** for any {`device` + `framework` +
    `backend`}: \> - device: {`GPU`, `XPU`, `MPS`, `CPU`} \> -
    framework: {`torch`, `deepspeed`, `horovod`, `tensorflow`} \> -
    backend: {`DDP`, `deepspeed`, `horovod`}

## [`test_dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)

``` python
import os
import logging
import time
from typing import Optional
import torch
import ezpz as ez

# backend can be any of DDP, deespepeed, horovod
RANK = ez.setup_torch(
  backend=(
      backend := os.environ.get('BACKEND', 'DDP')
  ),
  port=(
      port := os.environ.get("MASTER_PORT", "29500")
  )
)
# RANK = DIST_INIT['rank']
# WORLD_SIZE = DIST_INIT['world_size']
# LOCAL_RANK = DIST_INIT['local_rank']
# if DEVICE == "cuda" and torch.cuda.is_available():
#     torch.cuda.set_device(LOCAL_RANK)
DEVICE = ez.get_torch_device()
WORLD_SIZE = ez.get_world_size()
LOCAL_RANK = ez.get_local_rank()
DEVICE_ID = f"{DEVICE}:{LOCAL_RANK}"


# log only from RANK == 0
logger = logging.getLogger(__name__)
logger.setLevel("INFO") if RANK == 0 else logger.setLevel("CRITICAL")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))  # 64
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 128))  # 128
OUTPUT_SIZE = int(os.environ.get("OUTPUT_SIZE", 128))  # 128
DTYPE = os.environ.get("DTYPE", torch.get_default_dtype())
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 50))

# logger.info(f"{DIST_INIT=}")


class Network(torch.nn.Module):
  def __init__(
          self,
          input_dim: int = 128,
          output_dim: int = 128,
          sizes: Optional[list[int]] = None,
  ):
      super(Network, self).__init__()
      if sizes is None:
          self.layers = torch.nn.Linear(input_dim, output_dim)
      elif len(sizes) > 0:
          layers = [torch.nn.Linear(input_dim, sizes[0])]
          for idx, size in enumerate(sizes[1:]):
              layers.append(
                  torch.nn.Linear(sizes[idx], size)
              )
          layers.append(torch.nn.Linear(sizes[-1], output_dim))
          self.layers = torch.nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.layers(x)


def calc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return (y - x).pow(2).sum()


def plot_losses(losses: dict) -> None:
  import plotext as pltx
  # y = list(losses.values())
  pltx.theme('clear')
  pltx.scatter(list(losses.values()))
  pltx.show()
  pltx.save_fig("test_dist_losses.txt")
  pltx.ylabel("loss")
  pltx.xlabel("iteration")


def main():
  model = Network(
      input_dim=INPUT_SIZE,
      output_dim=OUTPUT_SIZE,
      sizes=[1024, 512, 256, 128]
  )
  model.to(DEVICE)
  model.to(DEVICE_ID)
  logger.info(f'{model=}')
  optimizer = torch.optim.Adam(model.parameters())
  if backend.lower() == 'ddp':
      if WORLD_SIZE > 1:
          from torch.nn.parallel import DistributedDataParallel as DDP
          model = DDP(
              model,
              device_ids=[]
          )
  elif backend.lower() in ('ds', 'deepspeed'):
      import deepspeed
      # config = ez.load_ds_config().update(
      #     {"train_micro_batch_size_per_gpu": BATCH_SIZE}
      # )
      import argparse
      parser = argparse.ArgumentParser(
          description='My training script.'
      )
      parser.add_argument(
          '--local_rank',
          required=False,
          type=int,
          default=-1,
          # default=ez.get_local_rank()),
          help='local rank passed from distributed launcher',
      )
      # Include DeepSpeed configuration arguments
      parser = deepspeed.add_config_arguments(parser)
      cmd_args = parser.parse_args()
      logger.info(f'{cmd_args=}')
      model, optimizer, *_ = deepspeed.initialize(
          args=cmd_args,
          model=model,
          optimizer=optimizer,
      )

  losses = {}
  for iter in range(TRAIN_ITERS):
      t0 = time.perf_counter()
      x = torch.rand((BATCH_SIZE, INPUT_SIZE), dtype=DTYPE).to(DEVICE)
      y = model(x)
      loss = calc_loss(x, y)
      losses[iter] = loss
      dtf = ((t1 := time.perf_counter()) - t0)
      if backend == 'deepspeed':
          model.backward(loss)
          model.step(loss)
      else:
          loss.backward()
          optimizer.step()
      optimizer.zero_grad()
      dtb = time.perf_counter() - t1
      logger.info(
          ', '.join([
              f'{iter=}',
              f'loss={loss.item():.5f}',
              f'dt={dtf+dtb:.3f}',
              f'{dtf=:.3f}',
              f'{dtb=:.3f}'
          ])
      )
  if RANK == 0:
      plot_losses(losses)


if __name__ == '__main__':
  main()
```

</details>

<details closed>

<summary>

<code>README</code> (grandparent)
</summary>

## ezpz

<img alt="pyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.tensorflow.org"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&logo=TensorFlow&logoColor=white"></a>

Simplifies the process of setting up distributed training for:

- `pytorch` + `{DDP, deepspeed, horovod}`

- `tensorflow` + `horovod`

### Setup

- Install:

  ``` bash
  git clone https://github.com/saforem2/ezpz
  python3 -m pip install -e ezpz
  ```

- Determine available resources:

  ``` bash
  [ "$(hostname)==theta*" ] && HOSTFILE="${COBALT_NODEFILE}"  # ThetaGPU @ ALCF
  [ "$(hostname)==x3*" ] && HOSTFILE="${PBS_NODEFILE}"        # Polaris @ ALCF
  [ "$(hostname)==nid*" ] && HOSTFILE="${SLURM_NODELIST}"     # Perlmutter @ NERSC
  NHOSTS=$(wc -l < "${HOSTFILE}")
  NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
  NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))";
  echo $NHOSTS $NGPU_PER_HOST $NGPUS
  2 4 8
  ```

- Example `python` script:

  ``` python
  """
  ezpz/test.py
  """
  from ezpz import setup_torch, setup_tensorflow


  def test(
          framework: str = 'pytorch',
          backend: str = 'deepspeed',
          port: int | str = '5432'
  ):
      if framework == 'pytorch':
          _ = setup_torch(
              backend=backend,
              port=port,
          )
      elif framework == 'tensorflow':
          _ = setup_tensorflow()
      else:
          raise ValueError  

  if __name__ == '__main__':
      import sys
      try:
          framework = sys.argv[1]
      except IndexError:
          framework = 'pytorch'
      try:
          backend = sys.argv[2]
      except IndexError:
          backend = 'deepspeed'
      try:
          port = sys.argv[3]
      except IndexError:
          port = '5432'
      test(framework=framework, backend=backend, port=port)
  ```

### Tests

You can test a `{framework, backend}` combination by:

``` bash
mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py framework backend
```

for `framework` $\in$ `{pytorch, tensorflow}` and `backend` $\in$
`{horovod, deepspeed, DDP}`[^1]

- ✅ PyTorch + DDP:

  ``` bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py pytorch DDP
  ```

- ✅ PyTorch + DeepSpeed:

  ``` bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py pytorch deepspeed
  ```

- ✅ PyTorch + Horovod:

  ``` bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py pytorch horovod
  ```

- ✅ TensorFlow + Horovod:

  ``` bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py tensorflow
  ```

### Helper Utilities

- [`src/ezpz/bin/savejobenv`](./src/ezpz/bin/savejobenv): Shell script
  to save relevant job related environment variables to a file which can
  be sourced from new login instances.

- [`src/ezpz/bin/getjobenv`](./src/ezpz/bin/getjobenv): Shell script
  that, when sourced, will populate the current environment with the
  necessary job-related variables.

1.  Launch a job, clone (or navigate into) `ezpz`, and run
    [`src/ezpz/bin/savejobenv`](./src/ezpz/bin/savejobenv):

    ``` bash
    (thetalogin5) $ qsub-gpu -A datascience -n 4 -q full-node --attrs="filesystems=home,grand,eagle,theta-fs0:ssds=required" -t 12:00 -I
    (thetagpu13) $ git clone https://github.com/saforem2/ezpz
    (thetagpu13) $ cd ezpz/src/ezpz
    (thetagpu13) $ ./bin/savejobenv
    ┌──────────────────────────────────────────────────────────────────┐
    │ [DIST INFO]:
    │   • Writing Job info to /home/foremans/.cobaltenv
    │       • NHOSTS: 4
    │       • NGPU_PER_HOST: 8
    │       • NGPUS = (NHOSTS * NGPU_PER_HOST) = 32
    └──────────────────────────────────────────────────────────────────┘
    ┌──────────────────────────────────────────────────────────────────┐
    │ Saving COBALT env to /home/foremans/.cobaltenv from thetagpu13
    │ Writing COBALT vars to /home/foremans/.cobaltenv                 │
    └──────────────────────────────────────────────────────────────────┘
    ┌──────────────────────────────────────────────────────────────────┐
    │ Copying COBALT_NODEFILE to clipboard...
    │ COBALT_NODEFILE: /var/tmp/cobalt.10154591
    │ [Hosts]:
    │   thetagpu13 thetagpu12 thetagpu19 thetagpu18
    └──────────────────────────────────────────────────────────────────┘
    ┌───────────────────────────────────────────────────────────────────────┐
    │ Run 'source getjobenv' in a NEW SHELL to automatically set env vars   │
    └───────────────────────────────────────────────────────────────────────┘
    ```

2.  now, in a **NEW SHELL**

    ``` bash
    (localhost) $ ssh foremans@theta
    (thetalogin5) $ ssh thetagpu18
    (thetagpu18) $ module load conda/2023-01-11; cond activate base
    (thetagpu18) $ cd ezpz
    (thetagpu18) $ mkdir -p venvs/thetaGPU/2023-01-11
    (thetagpu18) $ python3 -m venv venvs/thetaGPU/2023-01-11 --system-site-packages
    (thetagpu18) $ source venvs/thetaGPU/2023-01-11/bin/activate
    (thetagpu18) $ python3 -m pip install -e .
    (thetagpu18) $ cd ezpz/src/ezpz
    (thetagpu18) $ source bin/getjobenv
    RUNNING_JOB_FILE: /var/tmp/cobalt-running-job
    JOBID: 10154591
    Loading job env from: /home/foremans/.cobaltenv
    Defining alias mpilaunch: mpilaunch: aliased to mpirun -n 32 -N 8 --hostfile /var/tmp/cobalt.10154591 -x PATH -x LD_LIBRARY_PATH
    HOSTFILE: /var/tmp/cobalt.10154591
    NHOSTS: 4
    NGPU_PER_HOST: 8
    NGPUS (NHOSTS x NGPU_PER_HOST): 32
    HOSTS: thetagpu13 thetagpu12 thetagpu19 thetagpu18
    (thetagpu18) $ mpilaunch python3 -m ezpz pytorch DDP
    Using DDP for distributed training
    RANK: 0 / 31
    RANK: 25 / 31
    RANK: 24 / 31
    RANK: 15 / 31
    RANK: 26 / 31
    RANK: 31 / 31
    RANK: 2 / 31
    RANK: 12 / 31
    RANK: 1 / 31
    RANK: 28 / 31
    RANK: 3 / 31
    RANK: 14 / 31
    RANK: 4 / 31
    RANK: 10 / 31
    RANK: 27 / 31
    RANK: 5 / 31
    RANK: 30 / 31
    RANK: 29 / 31
    RANK: 9 / 31
    RANK: 7 / 31
    RANK: 6 / 31
    RANK: 13 / 31
    RANK: 8 / 31
    RANK: 11 / 31
    RANK: 18 / 31
    RANK: 16 / 31
    RANK: 21 / 31
    RANK: 20 / 31
    RANK: 22 / 31
    RANK: 19 / 31
    RANK: 17 / 31
    RANK: 23 / 31
    ```

    while this example looked at ThetaGPU, the exact same process will
    work on any of `{ThetaGPU, Polaris, Perlmutter}`.

    2ez

</details>

</details>

<details closed>

<summary>

Deprecated:
</summary>

## Complete Example

### Gist

> [!TIP]
>
> ### <span class="dim-text" style="font-weight: normal;"> <code>saforem2/ezpz.md</code></span>
>
> <script src="https://gist.github.com/saforem2/2f2549894d9c65ed2edcfe6b1dbe6a70.js"></script>

### 📦 Clone Repo(s)

- [`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed):

  ``` bash
  $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
  Cloning into 'Megatron-DeepSpeed'...
  remote: Enumerating objects: 15538, done.
  remote: Counting objects: 100% (21/21), done.
  remote: Compressing objects: 100% (11/11), done.
  remote: Total 15538 (delta 10), reused 18 (delta 10), pack-reused 15517
  Receiving objects: 100% (15538/15538), 6.25 MiB | 32.32 MiB/s, done.
  Resolving deltas: 100% (11482/11482), done.
  Updating files: 100% (596/596), done.
  ```

- [`saforem2/ezpz`](https://github.com/saforem2/ezpz):

  ``` bash
  $ cd Megatron-DeepSpeed
  $ git clone https://github.com/saforem2/ezpz deps/ezpz
  Cloning into 'deps/ezpz'...
  remote: Enumerating objects: 2161, done.
  remote: Counting objects: 100% (390/390), done.
  remote: Compressing objects: 100% (181/181), done.
  remote: Total 2161 (delta 214), reused 285 (delta 151), pack-reused 1771
  Receiving objects: 100% (2161/2161), 4.28 MiB | 25.35 MiB/s, done.
  Resolving deltas: 100% (1134/1134), done.
  ```

### 🛜 Setup Job

1.  Source
    [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):

    ``` bash
    $ PBS_O_WORKDIR=$(pwd) source deps/ezpz/src/ezpz/bin/utils.sh
    Using WORKING_DIR: /eagle/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed
    ```

2.  `ezpz_setup_alcf`:

    ``` bash
    $ ezpz_setup_alcf
    [ezpz/bin/utils.sh]

    [2024-07-23-221417]
        • USER=foremans
        • MACHINE=polaris
        • HOST=x3006c0s25b1n0

    [ezpz_get_pbs_env]: Caught 0 arguments
        • hostfile: /var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
        • jobenv_file: /home/foremans/.pbsenv

    [ezpz_setup_host]
        • Using hostfile: /var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
        • Found in environment:
            • HOSTFILE: /var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
            • Writing PBS vars to: /home/foremans/.pbsenv

    [ezpz_save_pbs_env]
        • Setting:
            • HOSTFILE: /var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
            • JOBENV_FILE: /home/foremans/.pbsenv

    [HOSTS]
        • [host:0] - x3006c0s25b1n0.hsn.cm.polaris.alcf.anl.gov

    [DIST INFO]
        • HOSTFILE=/var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
        • NHOSTS=1
        • NGPU_PER_HOST=4
        • NGPUS=4
        • DIST_LAUNCH=mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16

    [LAUNCH]:
        • To launch across all available GPUs, use: launch
          launch = mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16
    ```

### 🐍 Setup Python

- `ezpz_setup_python`:

  ``` bash
  $ ezpz_setup_python
  No conda_prefix OR virtual_env found in environment...
  Setting up conda...
  Lmod is automatically replacing "nvhpc/23.9" with "gcc-native/12.3".
  Lmod is automatically replacing "PrgEnv-nvhpc/8.5.0" with "PrgEnv-gnu/8.5.0".
  Due to MODULEPATH changes, the following have been reloaded:
    1) cray-mpich/8.1.28
  Found conda at: /soft/applications/conda/2024-04-29/mconda3
  No VIRTUAL_ENV found in environment!
      - Trying to setup from /soft/applications/conda/2024-04-29/mconda3
      - Using VENV_DIR=/eagle/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/venvs/2024-04-29
      - Creating a new virtual env on top of 2024-04-29 in /eagle/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/venvs/2024-04-29
  [python] Using /eagle/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/venvs/2024-04-29/bin/python3

  $ which python3
  /eagle/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/venvs/2024-04-29/bin/python3

  $ python3 -m pip install -e deps/ezpz --require-virtualenv
  Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
  Obtaining file:///lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/deps/ezpz
    Installing build dependencies ... done
    Checking if build backend supports build_editable ... done
    Getting requirements to build editable ... done
    Installing backend dependencies ... done
    Preparing editable metadata (pyproject.toml) ... done
  ```

### 📝 Test Setup

- [`ezpz/test_dist.py`](https://github.com/saforem2/ezpz/blob/main/ezpz/test_dist.py)

  ``` bash
  $ launch python3 -m ezpz.test_dist
  ```

  <details closed>

  <summary>

  Output

  </summary>

  ``` bash
  [2024-07-23 22:21:37.972869][INFO][__init__:156] - Setting logging level to 'INFO' on 'RANK == 0'
  [2024-07-23 22:21:37.975224][INFO][__init__:157] - Setting logging level to 'CRITICAL' on all others 'RANK != 0'
  [2024-07-23 22:21:37.975718][INFO][__init__:160] - To disable this behavior, and log from ALL ranks (not recommended), set: 'export LOG_FROM_ALL_RANKS=1'  in your environment, and re-run.
  [2024-07-23 22:21:39.790899][INFO][dist:358] - [device='cuda'][rank=1/3][local_rank=1/3][node=0/0]
  [2024-07-23 22:21:39.790850][INFO][dist:358] - [device='cuda'][rank=2/3][local_rank=2/3][node=0/0]
  [2024-07-23 22:21:39.791749][INFO][dist:358] - [device='cuda'][rank=3/3][local_rank=3/3][node=0/0]
  [2024-07-23 22:21:39.797666][INFO][dist:95] -

  [dist_info]:
    • DEVICE=cuda
    • DEVICE_ID=cuda:0
    • DISTRIBUTED_BACKEND=nccl
    • GPUS_PER_NODE=4
    • HOSTS=['x3006c0s25b1n0.hsn.cm.polaris.alcf.anl.gov']
    • HOSTFILE=/var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    • HOSTNAME=x3006c0s25b1n0.hsn.cm.polaris.alcf.anl.gov
    • LOCAL_RANK=0
    • MACHINE=Polaris
    • NUM_NODES=1
    • NGPUS=4
    • NGPUS_AVAILABLE=4
    • NODE_ID=0
    • RANK=0
    • SCHEDULER=PBS
    • WORLD_SIZE_TOTAL=4
    • WORLD_SIZE_IN_USE=4
    • LAUNCH_CMD=mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2036165.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16


  [2024-07-23 22:21:39.800519][INFO][dist:725] - [0/4] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
  [2024-07-23 22:21:39.805001][INFO][dist:358] - [device='cuda'][rank=0/3][local_rank=0/3][node=0/0]
  [2024-07-23 22:21:39.805513][WARNING][dist:364] - Using [4 / 4] available "cuda" devices !!
  [2024-07-23 22:21:39.806121][INFO][dist:95] -

  [timers_import]:
    • os=1.062639057636261e-06
    • logging=4.0046870708465576e-07
    • typing=2.7157366275787354e-06
    • pathlib=1.2516975402832031e-06
    • ezpz=6.30505383014679e-07
    • torch=2.555549144744873e-06
    • torch_ddp=2.4745240807533264e-06
    • wandb=6.44102692604065e-05
    • total=7.550138980150223e-05


  [2024-07-23 22:21:39.807221][INFO][dist:95] -

  [CONFIG]:
    • warmup=0
    • log_freq=1
    • batch_size=64
    • input_size=128
    • output_size=128
    • dtype=torch.float32
    • device=cuda
    • world_size=4
    • train_iters=100

  [2024-07-23 22:21:41.373173][INFO][test_dist:183] - model=Network(
    (layers): Sequential(
      (0): Linear(in_features=128, out_features=1024, bias=True)
      (1): Linear(in_features=1024, out_features=512, bias=True)
      (2): Linear(in_features=512, out_features=256, bias=True)
      (3): Linear(in_features=256, out_features=128, bias=True)
      (4): Linear(in_features=128, out_features=128, bias=True)
    )
  )
  [2024-07-23 22:21:43.625040][INFO][test_dist:274] - iter=1, loss=2039.91, sps=2.252e+04, dt=0.00284196, dtf=0.0009801, dtb=0.001862
  [2024-07-23 22:21:43.628643][INFO][test_dist:274] - iter=2, loss=1424.54, sps=3.272e+04, dt=0.00195628, dtf=0.0005183, dtb=0.001438
  [2024-07-23 22:21:43.631833][INFO][test_dist:274] - iter=3, loss=1159.38, sps=3.331e+04, dt=0.00192147, dtf=0.0006139, dtb=0.001308
  [2024-07-23 22:21:43.634991][INFO][test_dist:274] - iter=4, loss=935.58, sps=3.343e+04, dt=0.00191451, dtf=0.0006113, dtb=0.001303
  [2024-07-23 22:21:43.638092][INFO][test_dist:274] - iter=5, loss=851.468, sps=3.483e+04, dt=0.00183759, dtf=0.0005938, dtb=0.001244
  [2024-07-23 22:21:43.641232][INFO][test_dist:274] - iter=6, loss=785.109, sps=3.409e+04, dt=0.00187757, dtf=0.0005972, dtb=0.00128
  [2024-07-23 22:21:43.644367][INFO][test_dist:274] - iter=7, loss=772.966, sps=3.417e+04, dt=0.00187292, dtf=0.0005868, dtb=0.001286
  [2024-07-23 22:21:43.647507][INFO][test_dist:274] - iter=8, loss=727.854, sps=3.411e+04, dt=0.00187638, dtf=0.0005814, dtb=0.001295
  [2024-07-23 22:21:43.650561][INFO][test_dist:274] - iter=9, loss=725.773, sps=3.546e+04, dt=0.00180485, dtf=0.0005975, dtb=0.001207
  [2024-07-23 22:21:43.653520][INFO][test_dist:274] - iter=10, loss=720.374, sps=3.785e+04, dt=0.00169078, dtf=0.0006108, dtb=0.00108
  [2024-07-23 22:21:43.656564][INFO][test_dist:274] - iter=11, loss=717.926, sps=3.602e+04, dt=0.00177678, dtf=0.0005694, dtb=0.001207
  [2024-07-23 22:21:43.659555][INFO][test_dist:274] - iter=12, loss=692.535, sps=3.682e+04, dt=0.00173814, dtf=0.0005557, dtb=0.001182
  [2024-07-23 22:21:43.662596][INFO][test_dist:274] - iter=13, loss=679.509, sps=3.67e+04, dt=0.00174377, dtf=0.0005531, dtb=0.001191
  [2024-07-23 22:21:43.665598][INFO][test_dist:274] - iter=14, loss=674.778, sps=3.727e+04, dt=0.00171741, dtf=0.0005496, dtb=0.001168
  [2024-07-23 22:21:43.668593][INFO][test_dist:274] - iter=15, loss=673.873, sps=3.666e+04, dt=0.00174556, dtf=0.0005708, dtb=0.001175
  [2024-07-23 22:21:43.671589][INFO][test_dist:274] - iter=16, loss=667.283, sps=3.694e+04, dt=0.00173238, dtf=0.0005453, dtb=0.001187
  [2024-07-23 22:21:43.674599][INFO][test_dist:274] - iter=17, loss=660.292, sps=3.646e+04, dt=0.00175558, dtf=0.0005538, dtb=0.001202
  [2024-07-23 22:21:43.677592][INFO][test_dist:274] - iter=18, loss=660.664, sps=3.696e+04, dt=0.00173169, dtf=0.0005441, dtb=0.001188
  [2024-07-23 22:21:43.680559][INFO][test_dist:274] - iter=19, loss=676.161, sps=3.709e+04, dt=0.00172556, dtf=0.0005668, dtb=0.001159
  [2024-07-23 22:21:43.683539][INFO][test_dist:274] - iter=20, loss=665.099, sps=3.702e+04, dt=0.0017287, dtf=0.0005281, dtb=0.001201
  [2024-07-23 22:21:43.686527][INFO][test_dist:274] - iter=21, loss=626.671, sps=3.7e+04, dt=0.00172989, dtf=0.0005279, dtb=0.001202
  [2024-07-23 22:21:43.689518][INFO][test_dist:274] - iter=22, loss=632.127, sps=3.702e+04, dt=0.00172883, dtf=0.0005085, dtb=0.00122
  [2024-07-23 22:21:43.692469][INFO][test_dist:274] - iter=23, loss=657.324, sps=3.755e+04, dt=0.00170436, dtf=0.0005164, dtb=0.001188
  [2024-07-23 22:21:43.695563][INFO][test_dist:274] - iter=24, loss=617.646, sps=3.558e+04, dt=0.00179856, dtf=0.0005767, dtb=0.001222
  [2024-07-23 22:21:43.698537][INFO][test_dist:274] - iter=25, loss=618.284, sps=3.705e+04, dt=0.00172744, dtf=0.0005522, dtb=0.001175
  [2024-07-23 22:21:43.701410][INFO][test_dist:274] - iter=26, loss=615.418, sps=3.961e+04, dt=0.00161577, dtf=0.0005298, dtb=0.001086
  [2024-07-23 22:21:43.704427][INFO][test_dist:274] - iter=27, loss=599.058, sps=3.648e+04, dt=0.00175461, dtf=0.0005156, dtb=0.001239
  [2024-07-23 22:21:43.707374][INFO][test_dist:274] - iter=28, loss=621.717, sps=3.778e+04, dt=0.00169387, dtf=0.0004899, dtb=0.001204
  [2024-07-23 22:21:43.710390][INFO][test_dist:274] - iter=29, loss=597.588, sps=3.623e+04, dt=0.00176654, dtf=0.0005663, dtb=0.0012
  [2024-07-23 22:21:43.713386][INFO][test_dist:274] - iter=30, loss=598.102, sps=3.71e+04, dt=0.00172484, dtf=0.0005497, dtb=0.001175
  [2024-07-23 22:21:43.716530][INFO][test_dist:274] - iter=31, loss=586.188, sps=3.357e+04, dt=0.00190664, dtf=0.0005618, dtb=0.001345
  [2024-07-23 22:21:43.719525][INFO][test_dist:274] - iter=32, loss=591.646, sps=3.672e+04, dt=0.00174293, dtf=0.000561, dtb=0.001182
  [2024-07-23 22:21:43.722513][INFO][test_dist:274] - iter=33, loss=574.161, sps=3.668e+04, dt=0.00174487, dtf=0.0005502, dtb=0.001195
  [2024-07-23 22:21:43.725524][INFO][test_dist:274] - iter=34, loss=586.41, sps=3.707e+04, dt=0.00172628, dtf=0.0005552, dtb=0.001171
  [2024-07-23 22:21:43.728594][INFO][test_dist:274] - iter=35, loss=574.43, sps=3.605e+04, dt=0.00177526, dtf=0.000576, dtb=0.001199
  [2024-07-23 22:21:43.731615][INFO][test_dist:274] - iter=36, loss=552.77, sps=3.642e+04, dt=0.00175741, dtf=0.0005588, dtb=0.001199
  [2024-07-23 22:21:43.734574][INFO][test_dist:274] - iter=37, loss=567.612, sps=3.748e+04, dt=0.00170768, dtf=0.0005318, dtb=0.001176
  [2024-07-23 22:21:43.737564][INFO][test_dist:274] - iter=38, loss=561.004, sps=3.706e+04, dt=0.00172686, dtf=0.0005489, dtb=0.001178
  [2024-07-23 22:21:43.740578][INFO][test_dist:274] - iter=39, loss=555.718, sps=3.645e+04, dt=0.00175567, dtf=0.0005662, dtb=0.001189
  [2024-07-23 22:21:43.743565][INFO][test_dist:274] - iter=40, loss=543.661, sps=3.708e+04, dt=0.00172613, dtf=0.0005363, dtb=0.00119
  [2024-07-23 22:21:43.746561][INFO][test_dist:274] - iter=41, loss=537.186, sps=3.691e+04, dt=0.00173373, dtf=0.0005346, dtb=0.001199
  [2024-07-23 22:21:43.749446][INFO][test_dist:274] - iter=42, loss=545.877, sps=3.998e+04, dt=0.00160083, dtf=0.000533, dtb=0.001068
  [2024-07-23 22:21:43.752446][INFO][test_dist:274] - iter=43, loss=546.533, sps=3.681e+04, dt=0.00173875, dtf=0.0005124, dtb=0.001226
  [2024-07-23 22:21:43.755384][INFO][test_dist:274] - iter=44, loss=545.989, sps=3.796e+04, dt=0.001686, dtf=0.0005054, dtb=0.001181
  [2024-07-23 22:21:43.758389][INFO][test_dist:274] - iter=45, loss=531.344, sps=3.667e+04, dt=0.00174516, dtf=0.0005569, dtb=0.001188
  [2024-07-23 22:21:43.761470][INFO][test_dist:274] - iter=46, loss=515.415, sps=3.69e+04, dt=0.00173432, dtf=0.000551, dtb=0.001183
  [2024-07-23 22:21:43.764494][INFO][test_dist:274] - iter=47, loss=523.498, sps=3.634e+04, dt=0.00176121, dtf=0.0005524, dtb=0.001209
  [2024-07-23 22:21:43.767522][INFO][test_dist:274] - iter=48, loss=515.942, sps=3.625e+04, dt=0.00176562, dtf=0.0005655, dtb=0.0012
  [2024-07-23 22:21:43.770555][INFO][test_dist:274] - iter=49, loss=527.433, sps=3.62e+04, dt=0.00176783, dtf=0.0005579, dtb=0.00121
  [2024-07-23 22:21:43.773467][INFO][test_dist:274] - iter=50, loss=520.038, sps=3.938e+04, dt=0.00162521, dtf=0.0005579, dtb=0.001067
  [2024-07-23 22:21:43.776470][INFO][test_dist:274] - iter=51, loss=507.743, sps=3.68e+04, dt=0.00173934, dtf=0.0005378, dtb=0.001202
  [2024-07-23 22:21:43.779466][INFO][test_dist:274] - iter=52, loss=505.372, sps=3.694e+04, dt=0.00173268, dtf=0.0005321, dtb=0.001201
  [2024-07-23 22:21:43.782434][INFO][test_dist:274] - iter=53, loss=505.824, sps=3.736e+04, dt=0.00171324, dtf=0.0005403, dtb=0.001173
  [2024-07-23 22:21:43.785426][INFO][test_dist:274] - iter=54, loss=498.697, sps=3.751e+04, dt=0.00170619, dtf=0.0005259, dtb=0.00118
  [2024-07-23 22:21:43.788396][INFO][test_dist:274] - iter=55, loss=492.434, sps=3.719e+04, dt=0.00172085, dtf=0.0005036, dtb=0.001217
  [2024-07-23 22:21:43.791354][INFO][test_dist:274] - iter=56, loss=486.032, sps=3.754e+04, dt=0.00170497, dtf=0.0005077, dtb=0.001197
  [2024-07-23 22:21:43.794333][INFO][test_dist:274] - iter=57, loss=487.687, sps=3.803e+04, dt=0.00168299, dtf=0.0005009, dtb=0.001182
  [2024-07-23 22:21:43.797238][INFO][test_dist:274] - iter=58, loss=481.011, sps=3.929e+04, dt=0.00162898, dtf=0.0005554, dtb=0.001074
  [2024-07-23 22:21:43.800237][INFO][test_dist:274] - iter=59, loss=478.058, sps=3.692e+04, dt=0.00173365, dtf=0.0005374, dtb=0.001196
  [2024-07-23 22:21:43.803250][INFO][test_dist:274] - iter=60, loss=476.983, sps=3.666e+04, dt=0.00174587, dtf=0.0005318, dtb=0.001214
  [2024-07-23 22:21:43.806222][INFO][test_dist:274] - iter=61, loss=468.415, sps=3.716e+04, dt=0.00172234, dtf=0.0005256, dtb=0.001197
  [2024-07-23 22:21:43.809230][INFO][test_dist:274] - iter=62, loss=461.661, sps=3.727e+04, dt=0.00171737, dtf=0.0005219, dtb=0.001195
  [2024-07-23 22:21:43.812204][INFO][test_dist:274] - iter=63, loss=465.746, sps=3.688e+04, dt=0.00173519, dtf=0.0005067, dtb=0.001228
  [2024-07-23 22:21:43.815192][INFO][test_dist:274] - iter=64, loss=470.95, sps=3.724e+04, dt=0.00171855, dtf=0.0004994, dtb=0.001219
  [2024-07-23 22:21:43.818155][INFO][test_dist:274] - iter=65, loss=463.301, sps=3.774e+04, dt=0.00169586, dtf=0.0005053, dtb=0.001191
  [2024-07-23 22:21:43.821161][INFO][test_dist:274] - iter=66, loss=450.195, sps=3.68e+04, dt=0.00173904, dtf=0.0005626, dtb=0.001176
  [2024-07-23 22:21:43.824143][INFO][test_dist:274] - iter=67, loss=449.097, sps=3.662e+04, dt=0.00174746, dtf=0.0005578, dtb=0.00119
  [2024-07-23 22:21:43.827103][INFO][test_dist:274] - iter=68, loss=447.465, sps=3.778e+04, dt=0.00169412, dtf=0.0005488, dtb=0.001145
  [2024-07-23 22:21:43.830071][INFO][test_dist:274] - iter=69, loss=444.676, sps=3.835e+04, dt=0.00166873, dtf=0.0005467, dtb=0.001122
  [2024-07-23 22:21:43.833030][INFO][test_dist:274] - iter=70, loss=429.532, sps=3.83e+04, dt=0.00167122, dtf=0.0005362, dtb=0.001135
  [2024-07-23 22:21:43.836024][INFO][test_dist:274] - iter=71, loss=437.085, sps=3.711e+04, dt=0.00172438, dtf=0.0005086, dtb=0.001216
  [2024-07-23 22:21:43.839009][INFO][test_dist:274] - iter=72, loss=436.272, sps=3.71e+04, dt=0.00172525, dtf=0.0005177, dtb=0.001208
  [2024-07-23 22:21:43.841920][INFO][test_dist:274] - iter=73, loss=430.464, sps=3.893e+04, dt=0.00164403, dtf=0.0004874, dtb=0.001157
  [2024-07-23 22:21:43.844806][INFO][test_dist:274] - iter=74, loss=426.483, sps=3.904e+04, dt=0.0016393, dtf=0.000449, dtb=0.00119
  [2024-07-23 22:21:43.847771][INFO][test_dist:274] - iter=75, loss=413.371, sps=3.75e+04, dt=0.0017066, dtf=0.0005185, dtb=0.001188
  [2024-07-23 22:21:43.850712][INFO][test_dist:274] - iter=76, loss=421.381, sps=3.77e+04, dt=0.00169769, dtf=0.000506, dtb=0.001192
  [2024-07-23 22:21:43.853587][INFO][test_dist:274] - iter=77, loss=415.112, sps=3.988e+04, dt=0.0016047, dtf=0.000537, dtb=0.001068
  [2024-07-23 22:21:43.856557][INFO][test_dist:274] - iter=78, loss=413.084, sps=3.729e+04, dt=0.0017161, dtf=0.0005459, dtb=0.00117
  [2024-07-23 22:21:43.859518][INFO][test_dist:274] - iter=79, loss=412.671, sps=3.761e+04, dt=0.00170149, dtf=0.0005066, dtb=0.001195
  [2024-07-23 22:21:43.862469][INFO][test_dist:274] - iter=80, loss=408.688, sps=3.776e+04, dt=0.00169481, dtf=0.0005446, dtb=0.00115
  [2024-07-23 22:21:43.865521][INFO][test_dist:274] - iter=81, loss=400.914, sps=3.674e+04, dt=0.00174196, dtf=0.0005528, dtb=0.001189
  [2024-07-23 22:21:43.868536][INFO][test_dist:274] - iter=82, loss=389.823, sps=3.655e+04, dt=0.00175112, dtf=0.000574, dtb=0.001177
  [2024-07-23 22:21:43.871531][INFO][test_dist:274] - iter=83, loss=399.073, sps=3.686e+04, dt=0.00173618, dtf=0.0005504, dtb=0.001186
  [2024-07-23 22:21:43.874511][INFO][test_dist:274] - iter=84, loss=385.773, sps=3.725e+04, dt=0.00171814, dtf=0.0005499, dtb=0.001168
  [2024-07-23 22:21:43.877492][INFO][test_dist:274] - iter=85, loss=400.61, sps=3.739e+04, dt=0.00171182, dtf=0.000546, dtb=0.001166
  [2024-07-23 22:21:43.880505][INFO][test_dist:274] - iter=86, loss=389.813, sps=3.673e+04, dt=0.00174226, dtf=0.0005734, dtb=0.001169
  [2024-07-23 22:21:43.883515][INFO][test_dist:274] - iter=87, loss=385.995, sps=3.694e+04, dt=0.00173256, dtf=0.0005296, dtb=0.001203
  [2024-07-23 22:21:43.886470][INFO][test_dist:274] - iter=88, loss=379.115, sps=3.774e+04, dt=0.00169591, dtf=0.0005467, dtb=0.001149
  [2024-07-23 22:21:43.889422][INFO][test_dist:274] - iter=89, loss=378.738, sps=3.798e+04, dt=0.00168494, dtf=0.0005276, dtb=0.001157
  [2024-07-23 22:21:43.892414][INFO][test_dist:274] - iter=90, loss=365.054, sps=3.675e+04, dt=0.00174164, dtf=0.000513, dtb=0.001229
  [2024-07-23 22:21:43.895367][INFO][test_dist:274] - iter=91, loss=380.372, sps=3.772e+04, dt=0.00169654, dtf=0.000495, dtb=0.001201
  [2024-07-23 22:21:43.898322][INFO][test_dist:274] - iter=92, loss=377.233, sps=3.852e+04, dt=0.00166155, dtf=0.000539, dtb=0.001123
  [2024-07-23 22:21:43.901288][INFO][test_dist:274] - iter=93, loss=366.226, sps=3.788e+04, dt=0.00168959, dtf=0.0005446, dtb=0.001145
  [2024-07-23 22:21:43.904284][INFO][test_dist:274] - iter=94, loss=366.221, sps=3.69e+04, dt=0.00173462, dtf=0.0005535, dtb=0.001181
  [2024-07-23 22:21:43.907289][INFO][test_dist:274] - iter=95, loss=366.673, sps=3.662e+04, dt=0.00174759, dtf=0.0005328, dtb=0.001215
  [2024-07-23 22:21:43.910260][INFO][test_dist:274] - iter=96, loss=362.985, sps=3.716e+04, dt=0.00172234, dtf=0.0005436, dtb=0.001179
  [2024-07-23 22:21:43.913277][INFO][test_dist:274] - iter=97, loss=349.768, sps=3.668e+04, dt=0.00174469, dtf=0.000529, dtb=0.001216
  [2024-07-23 22:21:43.916293][INFO][test_dist:274] - iter=98, loss=363.521, sps=3.675e+04, dt=0.0017416, dtf=0.0005412, dtb=0.0012
  [2024-07-23 22:21:43.919280][INFO][test_dist:274] - iter=99, loss=345.533, sps=3.717e+04, dt=0.00172205, dtf=0.0005134, dtb=0.001209
                               train/dt [2024-07-23-222143]
         │
  0.00284┤▘
         │
         │
  0.00264┤
         │
         │
  0.00243┤
         │
  0.00222┤
         │
         │
  0.00201┤
         │▗
         │ ▝▘▗▗▖               ▝
  0.00181┤   ▘  ▖         ▗
         │       ▘▄ ▖▄▖▄▗▖ ▗▗ ▘ ▗▄▝▖▗▗▖▖▖▗▗▘▀ ▄    ▗▗ ▗  ▄   ▖     ▗▗▖ ▖▖ ▖  ▄ ▖▖
         │      ▝  ▝      ▘  ▝ ▘    ▘    ▖     ▝▘▀▗  ▘▘▝▘ ▘▄▝  ▘▘▝▘▘ ▝▝ ▝▗▝▗▘ ▝ ▝
  0.00160┤                  ▖          ▗     ▝     ▘          ▀ ▗
         └┬─────────────────┬────────────────┬─────────────────┬────────────────┬─
         1.0              25.5             50.0              74.5            99.0
  train/dt                                 iter
  [2024-07-23 22:21:43.943086][INFO][plot:156] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/dt.txt
  text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/dt.txt
                               train/dtf [2024-07-23-222143]
         ┌───────────────────────────────────────────────────────────────────────┐
  0.00098┤▘                                                                      │
         │                                                                       │
         │                                                                       │
  0.00089┤                                                                       │
         │                                                                       │
         │                                                                       │
  0.00080┤                                                                       │
         │                                                                       │
  0.00071┤                                                                       │
         │                                                                       │
         │                                                                       │
  0.00063┤                                                                       │
         │ ▝▘▄  ▞                                                                │
         │    ▝▘ ▖  ▖  ▖  ▗   ▖   ▗  ▗      ▖                       ▗  ▖         │
  0.00054┤        ▀▝ ▞▖    ▝   ▀▝▀ ▘▝ ▖▄ ▝▝▘▝▝▖▗   ▚     ▀▘▄    ▗▗ ▞ ▀▗ ▗  ▗▖▚▗ ▖│
         │▝            ▝▝▖▖ ▚       ▘   ▖▖    ▝ ▘▄  ▝▘▚ ▖   ▗▘ ▘▖ ▖     ▘▝▖    ▘▗│
         │                   ▝                    ▝    ▝      ▘           ▝      │
  0.00045┤                                                    ▗                  │
         └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
         1.0              25.5             50.0              74.5            99.0
  train/dtf                                iter
  [2024-07-23 22:21:43.952631][INFO][plot:156] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/dtf.txt
  text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/dtf.txt
                               train/dtb [2024-07-23-222143]
         ┌───────────────────────────────────────────────────────────────────────┐
  0.00186┤▘                                                                      │
         │                                                                       │
         │                                                                       │
  0.00173┤                                                                       │
         │                                                                       │
         │                                                                       │
  0.00160┤                                                                       │
         │                                                                       │
  0.00146┤                                                                       │
         │▗                                                                      │
         │                                                                       │
  0.00133┤                     ▗                                                 │
         │ ▝▖ ▗▖                                                                 │
         │   ▞              ▗                                                    │
  0.00120┤      ▖▖   ▗ ▗▗▘▝  ▗▖  ▖▗▖   ▖▘  ▖▄ ▄  ▚ ▗▗▖▞▝    ▝▖    ▖     ▖ ▚  ▗ ▘▄│
         │        ▀▗▘▘▘▖  ▘▝   ▘▝▝  ▀▝▘  ▀▝    ▝▘ ▝     ▘▀    ▞▘▘▝ ▞▝▚▗▖▗▗   ▘▝  │
         │                                                ▘▞               ▗▘    │
  0.00107┤      ▝           ▘          ▗     ▗     ▖            ▗                │
         └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
         1.0              25.5             50.0              74.5            99.0
  train/dtb                                iter
  [2024-07-23 22:21:43.962230][INFO][plot:156] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/dtb.txt
  text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/dtb.txt
                              train/loss [2024-07-23-222143]
         ┌────────────────────────────────────────────────────────────────────────┐
   2039.9┤▘                                                                       │
         │                                                                        │
         │                                                                        │
   1757.5┤                                                                        │
         │                                                                        │
         │                                                                        │
   1475.1┤▗                                                                       │
         │                                                                        │
   1192.7┤                                                                        │
         │ ▝                                                                      │
         │                                                                        │
    910.3┤  ▖                                                                     │
         │   ▖                                                                    │
         │   ▝▝▖▄▗                                                                │
    627.9┤        ▘▀▘▀▝▘▚▗▖▄▖▗                                                    │
         │                   ▘▝▘▀▝▘▚▝▄▗▖▄▗▖▄▗▖▖                                   │
         │                                    ▝▘▀▝▘▀▝▘▚▖▚▗▖▄▗▖▄▗▗                 │
    345.5┤                                                      ▘▝▘▀▝▘▀▝▀▝▘▞▝▖▄▗▖▄│
         └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
         1.0              25.5              50.0             74.5             99.0
  train/loss                               iter
  [2024-07-23 22:21:44.011096][INFO][plot:156] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/loss.txt
  text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/loss.txt
                             train/iter [2024-07-23-222144]
       ┌──────────────────────────────────────────────────────────────────────────┐
   99.0┤                                                                      ▗▗▖▀│
       │                                                                   ▄▝▘▘   │
       │                                                              ▗▖▞▝▘       │
   82.7┤                                                          ▄▗▘▀            │
       │                                                      ▖▄▝▘                │
       │                                                 ▗▗▖▀▝                    │
   66.3┤                                              ▄▝▘▘                        │
       │                                         ▗▖▞▝▘                            │
   50.0┤                                     ▄▗▘▀                                 │
       │                                 ▖▄▝▘                                     │
       │                            ▗▗▖▀▝                                         │
   33.7┤                         ▄▝▘▘                                             │
       │                    ▗▖▞▝▘                                                 │
       │                ▄▗▘▀                                                      │
   17.3┤            ▖▄▝▘                                                          │
       │       ▗▗▖▀▝                                                              │
       │    ▄▝▘▘                                                                  │
    1.0┤▖▞▝▘                                                                      │
       └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
       1.0              25.5               50.0              74.5             99.0
  train/iter                              iter
  [2024-07-23 22:21:44.021040][INFO][plot:156] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/iter.txt
  text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/iter.txt
                              train/sps [2024-07-23-222144]
         ┌───────────────────────────────────────────────────────────────────────┐
  39979.2┤                  ▖          ▝     ▗     ▖            ▝                │
         │                                                 ▄  ▀            ▗     │
         │      ▝  ▗      ▖  ▝      ▖    ▘     ▗▖▗▝   ▖▗▘ ▘    ▖▖▗▖▘ ▗▗ ▝▝▗ ▘    │
  37069.3┤        ▚ ▖▚▘▀▝▘ ▝▗  ▘▗▞ ▖▝▗▘▘▘▗▝▖▖ ▀  ▘ ▝▗▘▝  ▚  ▝▘     ▝▗▘ ▖▘ ▘  ▚▝▖▀│
         │      ▖▘        ▗   ▘   ▝         ▝                                    │
         │   ▘                                                                   │
  34159.4┤ ▗▖▝▝▘               ▗                                                 │
         │▗                                                                      │
  31249.4┤                                                                       │
         │                                                                       │
         │                                                                       │
  28339.5┤                                                                       │
         │                                                                       │
         │                                                                       │
  25429.6┤                                                                       │
         │                                                                       │
         │                                                                       │
  22519.7┤▖                                                                      │
         └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
         1.0              25.5             50.0              74.5            99.0
  train/sps                                iter
  [2024-07-23 22:21:44.030585][INFO][plot:156] - Appending plot to: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/sps.txt
  text saved in /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/test-dist-plots/train/sps.txt
  ```

  <details closed>

  <summary>

  PyInstrument Profile:

  </summary>

  ``` bash
  Recorded: 22:21:41  Samples:  2223
  Duration: 2.668     CPU time: 2.406
  v4.6.2

  Program: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/deps/ezpz/src/ezpz/test_dist.py

  2.668 <module>  ezpz/test_dist.py:1
  └─ 2.667 main  ezpz/test_dist.py:217
    ├─ 2.106 build_model_and_optimizer  ezpz/test_dist.py:171
    │  └─ 2.092 Adam.__init__  torch/optim/adam.py:15
    │        [147 frames hidden]  torch, transformers, jax, huggingface...
    ├─ 0.183 _forward_step  ezpz/test_dist.py:231
    │  ├─ 0.137 DistributedDataParallel._wrapped_call_impl  torch/nn/modules/module.py:1528
    │  │     [6 frames hidden]  torch
    │  │        0.123 Network._call_impl  torch/nn/modules/module.py:1534
    │  │        └─ 0.123 Network.forward  ezpz/test_dist.py:164
    │  │           └─ 0.123 Sequential._wrapped_call_impl  torch/nn/modules/module.py:1528
    │  │                 [7 frames hidden]  torch, <built-in>
    │  └─ 0.046 calc_loss  ezpz/test_dist.py:168
    ├─ 0.164 _backward_step  ezpz/test_dist.py:236
    │  ├─ 0.103 wrapper  torch/optim/optimizer.py:374
    │  │     [5 frames hidden]  torch
    │  └─ 0.060 Tensor.backward  torch/_tensor.py:466
    │        [4 frames hidden]  torch, <built-in>
    ├─ 0.113 tplot_dict  ezpz/plot.py:136
    │  └─ 0.082 show  plotext/_core.py:292
    │        [5 frames hidden]  plotext
    └─ 0.099 Logger.info  logging/__init__.py:1479
          [6 frames hidden]  logging, rich
              0.099 RichHandler.emit  rich/logging.py:126
              └─ 0.099 Console.print  ezpz/log/console.py:79
                └─ 0.099 Console.print  rich/console.py:1624
                      [5 frames hidden]  rich


  [2024-07-23 22:21:44.231519][INFO][profile:115] - Saving pyinstrument profile output to: /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/ezpz_pyinstrument_profiles
  [2024-07-23 22:21:44.232054][INFO][profile:123] - PyInstrument profile saved (as html) to:  /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/ezpz_pyinstrument_profiles/pyinstrument-profile-2024-07-23-222144.html
  [2024-07-23 22:21:44.232619][INFO][profile:131] - PyInstrument profile saved (as text) to:  /lus/eagle/projects/argonne_tpc/foremans/projects/argonne-lcf/tmp/2024-07-23-221253/Megatron-DeepSpeed/ezpz_pyinstrument_profiles/pyinstrument-profile-2024-07-23-222144.txt
  [2024-07-23 22:21:44.761876][INFO][profile:143] - Finished with pyinstrument profiler. Took: 2.66778s
  [2024-07-23 22:21:44.762534][INFO][test_dist:318] - [0] runtime=6.785542s
  ezpz-test-dist.log lines 216-359/359 (END)
  ```

  </details>

  </details>

------------------------------------------------------------------------

<details closed>

<summary>

Deprecated:
</summary>

1.  Download and source
    [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):

    ``` bash
    ezpz_utils() {
        fp=$(mktemp)
        curl -Ls https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/utils.sh > $fp
        source $fp
    }
    ezpz_utils
    ```

2.  Use `ezpz_setup_python` to:

    1.  Setup base `conda` environment
    2.  Create[^2] a virtual environment *on top of* the base `conda`
        environment[^3]

    ``` bash
    #[🌌][12:45:49 PM][foremans@x3006c0s13b0n0][~/tmp/foremans/2024-07-15-124441]
    $ PBS_O_WORKDIR=$(pwd) ezpz_utils && ezpz_setup_python && ezpz_setup_alcf 

    Using WORKING_DIR: /home/foremans/tmp/foremans/2024-07-15-124441
    No conda_prefix OR virtual_env found in environment...
    Setting up conda...

    Lmod is automatically replacing "nvhpc/23.9" with "gcc-native/12.3".


    Lmod is automatically replacing "PrgEnv-nvhpc/8.5.0" with "PrgEnv-gnu/8.5.0".


    Due to MODULEPATH changes, the following have been reloaded:
    1) cray-mpich/8.1.28

    Found conda at: /soft/applications/conda/2024-04-29/mconda3
    No VIRTUAL_ENV found in environment!
      - Trying to setup from /soft/applications/conda/2024-04-29/mconda3
      - Using VENV_DIR=/home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29

      - Creating a new virtual env on top of 2024-04-29 in /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29
    [python] Using /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29/bin/python3

    [ezpz/bin/utils.sh]

    [2024-07-15-124600]
      • USER=foremans
      • MACHINE=polaris
      • HOST=x3006c0s13b0n0

    [ezpz_setup_host]
      • Using hostfile: /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
      • Found in environment:
          • HOSTFILE: /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
          • Writing PBS vars to: /home/foremans/.pbsenv

    [ezpz_save_pbs_env]
      • Setting:
          • HOSTFILE: /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
          • JOBENV_FILE: /home/foremans/.pbsenv

    [HOSTS]
      • [host:0] - x3006c0s13b0n0.hsn.cm.polaris.alcf.anl.gov

    [DIST INFO]
      • HOSTFILE=/var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
      • NHOSTS=1
      • NGPU_PER_HOST=4
      • NGPUS=4
      • DIST_LAUNCH=mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16

    [LAUNCH]:
      • To launch across all available GPUs, use: launch
        launch = mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16
    ```

<!-- -->

3.  Install `ezpz` into the virtual environment from 2.

    ``` bash
    python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
    ```

    <details closed>

    <summary>

    output

    </summary>

    ``` bash
    Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
    Obtaining ezpz from git+https://github.com/saforem2/ezpz#egg=ezpz
    Cloning https://github.com/saforem2/ezpz to ./venvs/2024-04-29/src/ezpz
    Running command git clone --filter=blob:none --quiet https://github.com/saforem2/ezpz /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29/src/ezpz
    Resolved https://github.com/saforem2/ezpz to commit d8fabca03038db55a1dc490f801581e980f93a25
    Installing build dependencies ... done
    Checking if build backend supports build_editable ... done
    Getting requirements to build editable ... done
    Installing backend dependencies ... done
    Preparing editable metadata (pyproject.toml) ... done
    Requirement already satisfied: ambivalent in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (0.0.1)
    Requirement already satisfied: hydra-colorlog in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (1.2.0)
    Requirement already satisfied: hydra-core in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (1.3.2)
    Requirement already satisfied: ipython in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (8.24.0)
    Requirement already satisfied: jax in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.4.26)
    Requirement already satisfied: jaxlib in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.4.26+cuda12.cudnn89)
    Requirement already satisfied: jaxtyping in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.2.28)
    Requirement already satisfied: joblib in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (1.4.0)
    Requirement already satisfied: ml-dtypes in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.3.2)
    Requirement already satisfied: mpi4py in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (3.1.6)
    Requirement already satisfied: omegaconf in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2.3.0)
    Requirement already satisfied: plotext in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (5.2.8)
    Collecting pyinstrument (from ezpz)
    Downloading pyinstrument-4.6.2-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)
    Requirement already satisfied: rich in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (13.7.1)
    Requirement already satisfied: seaborn in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.13.2)
    Requirement already satisfied: sentencepiece in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (0.2.0)
    Requirement already satisfied: sh in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (2.0.6)
    Requirement already satisfied: tensorboard in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2.16.2)
    Requirement already satisfied: torch in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2.3.0)
    Requirement already satisfied: tqdm in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (4.65.0)
    Requirement already satisfied: wandb in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.16.6)
    Requirement already satisfied: xarray in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2024.3.0)
    Requirement already satisfied: colormaps in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ambivalent->ezpz) (0.4.1)
    Requirement already satisfied: matplotlib in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ambivalent->ezpz) (3.8.4)
    Requirement already satisfied: requests in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ambivalent->ezpz) (2.31.0)
    Requirement already satisfied: colorlog in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from hydra-colorlog->ezpz) (6.8.2)
    Requirement already satisfied: antlr4-python3-runtime==4.9.* in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from hydra-core->ezpz) (4.9.3)
    Requirement already satisfied: packaging in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from hydra-core->ezpz) (24.0)
    Requirement already satisfied: PyYAML>=5.1.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from omegaconf->ezpz) (6.0.1)
    Requirement already satisfied: decorator in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (0.1.7)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (2.17.2)
    Requirement already satisfied: stack-data in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (0.6.3)
    Requirement already satisfied: traitlets>=5.13.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (5.14.3)
    Requirement already satisfied: typing-extensions>=4.6 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (4.11.0)
    Requirement already satisfied: pexpect>4.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (4.9.0)
    Requirement already satisfied: numpy>=1.22 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jax->ezpz) (1.26.4)
    Requirement already satisfied: opt-einsum in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jax->ezpz) (3.3.0)
    Requirement already satisfied: scipy>=1.9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jax->ezpz) (1.13.0)
    Requirement already satisfied: typeguard==2.13.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jaxtyping->ezpz) (2.13.3)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from rich->ezpz) (3.0.0)
    Requirement already satisfied: pandas>=1.2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from seaborn->ezpz) (2.2.2)
    Requirement already satisfied: absl-py>=0.4 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (2.1.0)
    Requirement already satisfied: grpcio>=1.48.2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (1.62.2)
    Requirement already satisfied: markdown>=2.6.8 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (3.6)
    Requirement already satisfied: protobuf!=4.24.0,>=3.19.6 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (3.20.3)
    Requirement already satisfied: setuptools>=41.0.0 in ./venvs/2024-04-29/lib/python3.11/site-packages (from tensorboard->ezpz) (65.5.0)
    Requirement already satisfied: six>1.9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (1.16.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (3.0.2)
    Requirement already satisfied: filelock in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (3.13.1)
    Requirement already satisfied: sympy in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (1.12)
    Requirement already satisfied: networkx in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (3.3)
    Requirement already satisfied: jinja2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (3.0.3)
    Requirement already satisfied: fsspec in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (2024.3.1)
    Requirement already satisfied: Click!=8.0.0,>=7.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (8.1.7)
    Requirement already satisfied: GitPython!=3.1.29,>=1.0.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (3.1.43)
    Requirement already satisfied: psutil>=5.0.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (5.9.8)
    Requirement already satisfied: sentry-sdk>=1.0.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (2.0.1)
    Requirement already satisfied: docker-pycreds>=0.4.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (0.4.0)
    Requirement already satisfied: setproctitle in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (1.3.3)
    Requirement already satisfied: appdirs>=1.4.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (1.4.4)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from GitPython!=3.1.29,>=1.0.0->wandb->ezpz) (4.0.11)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jedi>=0.16->ipython->ezpz) (0.8.4)
    Requirement already satisfied: mdurl~=0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich->ezpz) (0.1.2)
    Requirement already satisfied: contourpy>=1.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (1.2.1)
    Requirement already satisfied: cycler>=0.10 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (1.4.5)
    Requirement already satisfied: pillow>=8 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (10.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from pandas>=1.2->seaborn->ezpz) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from pandas>=1.2->seaborn->ezpz) (2024.1)
    Requirement already satisfied: ptyprocess>=0.5 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from pexpect>4.3->ipython->ezpz) (0.7.0)
    Requirement already satisfied: wcwidth in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython->ezpz) (0.2.13)
    Requirement already satisfied: charset-normalizer<4,>=2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (2024.2.2)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from werkzeug>=1.0.1->tensorboard->ezpz) (2.1.3)
    Requirement already satisfied: executing>=1.2.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from stack-data->ipython->ezpz) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from stack-data->ipython->ezpz) (2.4.1)
    Requirement already satisfied: pure-eval in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from stack-data->ipython->ezpz) (0.2.2)
    Requirement already satisfied: mpmath>=0.19 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from sympy->torch->ezpz) (1.3.0)
    Requirement already satisfied: smmap<6,>=3.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from gitdb<5,>=4.0.1->GitPython!=3.1.29,>=1.0.0->wandb->ezpz) (5.0.1)
    Downloading pyinstrument-4.6.2-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (104 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.9/104.9 kB 3.1 MB/s eta 0:00:00
    Building wheels for collected packages: ezpz
    Building editable for ezpz (pyproject.toml) ... done
    Created wheel for ezpz: filename=ezpz-0.1-py3-none-any.whl size=10104 sha256=f73fbc552c6192f2d1575c08528267c1c70bd4ed2eebab011c692b4cf66fd9cb
    Stored in directory: /tmp/pip-ephem-wheel-cache-6xb0tqk8/wheels/b3/57/90/f3324177d75cbc607a034b5b8e66d5b3d35dcf087967430718
    Successfully built ezpz
    Installing collected packages: pyinstrument, ezpz
    Attempting uninstall: ezpz
      Found existing installation: ezpz 0.1
      Not uninstalling ezpz at /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages, outside environment /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29
      Cant uninstall 'ezpz'. No files were found to uninstall.
    Successfully installed ezpz pyinstrument-4.6.2

    [notice] A new release of pip is available: 24.0 -> 24.1.2
    [notice] To update, run: pip install --upgrade pip
    9.63s user 1.44s system 60% cpu 18.301s total
    ```

    </details>

``` bash
#[🌌][12:44:34 PM][foremans@x3006c0s13b0n0][~/tmp]
$ dname=$USER/$(tstamp) ; mkdir -p $dname && cd $dname


#[🌌][12:44:45 PM][foremans@x3006c0s13b0n0][~/tmp/foremans/2024-07-15-124441]
; ezpz_utils() { fp=$(mktemp) && curl -Ls https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/utils.sh > $fp && source $fp || exit }


#[🌌][12:44:47 PM][foremans@x3006c0s13b0n0][~/tmp/foremans/2024-07-15-124441]
$ PBS_O_WORKDIR=$(pwd) ezpz_utils
Using WORKING_DIR: /home/foremans/tmp/foremans/2024-07-15-124441


#[🌌][12:45:49 PM][foremans@x3006c0s13b0n0][~/tmp/foremans/2024-07-15-124441]
$ PBS_O_WORKDIR=$(pwd) ezpz_utils && ezpz_setup_python && ezpz_setup_alcf 

Using WORKING_DIR: /home/foremans/tmp/foremans/2024-07-15-124441
No conda_prefix OR virtual_env found in environment...
Setting up conda...

Lmod is automatically replacing "nvhpc/23.9" with "gcc-native/12.3".


Lmod is automatically replacing "PrgEnv-nvhpc/8.5.0" with "PrgEnv-gnu/8.5.0".


Due to MODULEPATH changes, the following have been reloaded:
  1) cray-mpich/8.1.28

Found conda at: /soft/applications/conda/2024-04-29/mconda3
No VIRTUAL_ENV found in environment!
    - Trying to setup from /soft/applications/conda/2024-04-29/mconda3
    - Using VENV_DIR=/home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29

    - Creating a new virtual env on top of 2024-04-29 in /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29
[python] Using /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29/bin/python3

[ezpz/bin/utils.sh]

[2024-07-15-124600]
    • USER=foremans
    • MACHINE=polaris
    • HOST=x3006c0s13b0n0

[ezpz_setup_host]
    • Using hostfile: /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    • Found in environment:
        • HOSTFILE: /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
        • Writing PBS vars to: /home/foremans/.pbsenv

[ezpz_save_pbs_env]
    • Setting:
        • HOSTFILE: /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
        • JOBENV_FILE: /home/foremans/.pbsenv

alias LAUNCH='mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16'
[HOSTS]
    • [host:0] - x3006c0s13b0n0.hsn.cm.polaris.alcf.anl.gov

[DIST INFO]
    • HOSTFILE=/var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    • NHOSTS=1
    • NGPU_PER_HOST=4
    • NGPUS=4
    • DIST_LAUNCH=mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16

[LAUNCH]:
    • To launch across all available GPUs, use: launch
      launch = mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Obtaining ezpz from git+https://github.com/saforem2/ezpz#egg=ezpz
  Cloning https://github.com/saforem2/ezpz to ./venvs/2024-04-29/src/ezpz
  Running command git clone --filter=blob:none --quiet https://github.com/saforem2/ezpz /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29/src/ezpz
  Resolved https://github.com/saforem2/ezpz to commit d8fabca03038db55a1dc490f801581e980f93a25
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Installing backend dependencies ... done
  Preparing editable metadata (pyproject.toml) ... done
Requirement already satisfied: ambivalent in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (0.0.1)
Requirement already satisfied: hydra-colorlog in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (1.2.0)
Requirement already satisfied: hydra-core in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (1.3.2)
Requirement already satisfied: ipython in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (8.24.0)
Requirement already satisfied: jax in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.4.26)
Requirement already satisfied: jaxlib in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.4.26+cuda12.cudnn89)
Requirement already satisfied: jaxtyping in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.2.28)
Requirement already satisfied: joblib in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (1.4.0)
Requirement already satisfied: ml-dtypes in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.3.2)
Requirement already satisfied: mpi4py in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (3.1.6)
Requirement already satisfied: omegaconf in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2.3.0)
Requirement already satisfied: plotext in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (5.2.8)
Collecting pyinstrument (from ezpz)
  Downloading pyinstrument-4.6.2-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)
Requirement already satisfied: rich in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (13.7.1)
Requirement already satisfied: seaborn in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.13.2)
Requirement already satisfied: sentencepiece in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (0.2.0)
Requirement already satisfied: sh in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ezpz) (2.0.6)
Requirement already satisfied: tensorboard in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2.16.2)
Requirement already satisfied: torch in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2.3.0)
Requirement already satisfied: tqdm in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (4.65.0)
Requirement already satisfied: wandb in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (0.16.6)
Requirement already satisfied: xarray in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ezpz) (2024.3.0)
Requirement already satisfied: colormaps in /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages (from ambivalent->ezpz) (0.4.1)
Requirement already satisfied: matplotlib in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ambivalent->ezpz) (3.8.4)
Requirement already satisfied: requests in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ambivalent->ezpz) (2.31.0)
Requirement already satisfied: colorlog in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from hydra-colorlog->ezpz) (6.8.2)
Requirement already satisfied: antlr4-python3-runtime==4.9.* in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from hydra-core->ezpz) (4.9.3)
Requirement already satisfied: packaging in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from hydra-core->ezpz) (24.0)
Requirement already satisfied: PyYAML>=5.1.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from omegaconf->ezpz) (6.0.1)
Requirement already satisfied: decorator in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (5.1.1)
Requirement already satisfied: jedi>=0.16 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (0.19.1)
Requirement already satisfied: matplotlib-inline in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (0.1.7)
Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (3.0.43)
Requirement already satisfied: pygments>=2.4.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (2.17.2)
Requirement already satisfied: stack-data in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (0.6.3)
Requirement already satisfied: traitlets>=5.13.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (5.14.3)
Requirement already satisfied: typing-extensions>=4.6 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (4.11.0)
Requirement already satisfied: pexpect>4.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from ipython->ezpz) (4.9.0)
Requirement already satisfied: numpy>=1.22 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jax->ezpz) (1.26.4)
Requirement already satisfied: opt-einsum in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jax->ezpz) (3.3.0)
Requirement already satisfied: scipy>=1.9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jax->ezpz) (1.13.0)
Requirement already satisfied: typeguard==2.13.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jaxtyping->ezpz) (2.13.3)
Requirement already satisfied: markdown-it-py>=2.2.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from rich->ezpz) (3.0.0)
Requirement already satisfied: pandas>=1.2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from seaborn->ezpz) (2.2.2)
Requirement already satisfied: absl-py>=0.4 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (2.1.0)
Requirement already satisfied: grpcio>=1.48.2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (1.62.2)
Requirement already satisfied: markdown>=2.6.8 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (3.6)
Requirement already satisfied: protobuf!=4.24.0,>=3.19.6 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (3.20.3)
Requirement already satisfied: setuptools>=41.0.0 in ./venvs/2024-04-29/lib/python3.11/site-packages (from tensorboard->ezpz) (65.5.0)
Requirement already satisfied: six>1.9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (1.16.0)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from tensorboard->ezpz) (3.0.2)
Requirement already satisfied: filelock in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (3.13.1)
Requirement already satisfied: sympy in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (1.12)
Requirement already satisfied: networkx in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (3.3)
Requirement already satisfied: jinja2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (3.0.3)
Requirement already satisfied: fsspec in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from torch->ezpz) (2024.3.1)
Requirement already satisfied: Click!=8.0.0,>=7.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (8.1.7)
Requirement already satisfied: GitPython!=3.1.29,>=1.0.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (3.1.43)
Requirement already satisfied: psutil>=5.0.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (5.9.8)
Requirement already satisfied: sentry-sdk>=1.0.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (2.0.1)
Requirement already satisfied: docker-pycreds>=0.4.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (0.4.0)
Requirement already satisfied: setproctitle in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (1.3.3)
Requirement already satisfied: appdirs>=1.4.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from wandb->ezpz) (1.4.4)
Requirement already satisfied: gitdb<5,>=4.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from GitPython!=3.1.29,>=1.0.0->wandb->ezpz) (4.0.11)
Requirement already satisfied: parso<0.9.0,>=0.8.3 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from jedi>=0.16->ipython->ezpz) (0.8.4)
Requirement already satisfied: mdurl~=0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich->ezpz) (0.1.2)
Requirement already satisfied: contourpy>=1.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (1.2.1)
Requirement already satisfied: cycler>=0.10 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (4.51.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (1.4.5)
Requirement already satisfied: pillow>=8 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (10.3.0)
Requirement already satisfied: pyparsing>=2.3.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (3.1.2)
Requirement already satisfied: python-dateutil>=2.7 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from matplotlib->ambivalent->ezpz) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from pandas>=1.2->seaborn->ezpz) (2024.1)
Requirement already satisfied: tzdata>=2022.7 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from pandas>=1.2->seaborn->ezpz) (2024.1)
Requirement already satisfied: ptyprocess>=0.5 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from pexpect>4.3->ipython->ezpz) (0.7.0)
Requirement already satisfied: wcwidth in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython->ezpz) (0.2.13)
Requirement already satisfied: charset-normalizer<4,>=2 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (2.1.0)
Requirement already satisfied: certifi>=2017.4.17 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from requests->ambivalent->ezpz) (2024.2.2)
Requirement already satisfied: MarkupSafe>=2.1.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from werkzeug>=1.0.1->tensorboard->ezpz) (2.1.3)
Requirement already satisfied: executing>=1.2.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from stack-data->ipython->ezpz) (2.0.1)
Requirement already satisfied: asttokens>=2.1.0 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from stack-data->ipython->ezpz) (2.4.1)
Requirement already satisfied: pure-eval in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from stack-data->ipython->ezpz) (0.2.2)
Requirement already satisfied: mpmath>=0.19 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from sympy->torch->ezpz) (1.3.0)
Requirement already satisfied: smmap<6,>=3.0.1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages (from gitdb<5,>=4.0.1->GitPython!=3.1.29,>=1.0.0->wandb->ezpz) (5.0.1)
Downloading pyinstrument-4.6.2-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (104 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.9/104.9 kB 3.1 MB/s eta 0:00:00
Building wheels for collected packages: ezpz
  Building editable for ezpz (pyproject.toml) ... done
  Created wheel for ezpz: filename=ezpz-0.1-py3-none-any.whl size=10104 sha256=f73fbc552c6192f2d1575c08528267c1c70bd4ed2eebab011c692b4cf66fd9cb
  Stored in directory: /tmp/pip-ephem-wheel-cache-6xb0tqk8/wheels/b3/57/90/f3324177d75cbc607a034b5b8e66d5b3d35dcf087967430718
Successfully built ezpz
Installing collected packages: pyinstrument, ezpz
  Attempting uninstall: ezpz
    Found existing installation: ezpz 0.1
    Not uninstalling ezpz at /home/foremans/.local/polaris/conda/2024-04-29/lib/python3.11/site-packages, outside environment /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29
    Cant uninstall 'ezpz'. No files were found to uninstall.
Successfully installed ezpz pyinstrument-4.6.2

[notice] A new release of pip is available: 24.0 -> 24.1.2
[notice] To update, run: pip install --upgrade pip
9.63s user 1.44s system 60% cpu 18.301s total
Connected to tcp://x3006c0s13b0n0.hsn.cm.polaris.alcf.anl.gov:7919
Found executable /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29/bin/python3
Launching application 5ccc89be-4289-49f5-8b4e-64104021b3c5
[2024-07-15 12:46:24.601903][INFO][__init__:156] - Setting logging level to 'INFO' on 'RANK == 0'
[2024-07-15 12:46:24.604160][INFO][__init__:157] - Setting logging level to 'CRITICAL' on all others 'RANK != 0'
[2024-07-15 12:46:24.604624][INFO][__init__:160] - To disable this behavior, and log from ALL ranks (not recommended), set: 'export LOG_FROM_ALL_RANKS=1'  in your environment, and re-run.
wandb: WARNING require() unsupported requirement: core
wandb: ERROR Supported wandb.require() features can be found at: https://wandb.me/library-require
wandb: WARNING require() unsupported requirement: core
wandb: ERROR Supported wandb.require() features can be found at: https://wandb.me/library-require
wandb: WARNING require() unsupported requirement: core
wandb: ERROR Supported wandb.require() features can be found at: https://wandb.me/library-require
wandb: WARNING require() unsupported requirement: core
wandb: ERROR Supported wandb.require() features can be found at: https://wandb.me/library-require
[2024-07-15 12:46:26.437667][INFO][dist:358] - [device='cuda'][rank=1/3][local_rank=1/3][node=0/0]
[2024-07-15 12:46:26.437716][INFO][dist:358] - [device='cuda'][rank=3/3][local_rank=3/3][node=0/0]
[2024-07-15 12:46:26.438619][INFO][dist:358] - [device='cuda'][rank=2/3][local_rank=2/3][node=0/0]
[2024-07-15 12:46:26.444402][INFO][dist:95] -

[dist_info]:
  • DEVICE=cuda
  • DEVICE_ID=cuda:0
  • DISTRIBUTED_BACKEND=nccl
  • GPUS_PER_NODE=4
  • HOSTS=['x3006c0s13b0n0.hsn.cm.polaris.alcf.anl.gov']
  • HOSTFILE=/var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
  • HOSTNAME=x3006c0s13b0n0.hsn.cm.polaris.alcf.anl.gov
  • LOCAL_RANK=0
  • MACHINE=Polaris
  • NUM_NODES=1
  • NGPUS=4
  • NGPUS_AVAILABLE=4
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=4
  • WORLD_SIZE_IN_USE=4
  • LAUNCH_CMD=mpiexec --verbose --envall -n 4 -ppn 4 --hostfile /var/spool/pbs/aux/2021158.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16


[2024-07-15 12:46:26.447254][INFO][dist:725] - [0/4] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
[2024-07-15 12:46:26.451604][INFO][dist:358] - [device='cuda'][rank=0/3][local_rank=0/3][node=0/0]
[2024-07-15 12:46:26.452120][WARNING][dist:364] - Using [4 / 4] available "cuda" devices !!
[2024-07-15 12:46:26.452676][INFO][dist:95] -

[timers_import]:
  • os=1.1026859283447266e-06
  • logging=4.507601261138916e-07
  • typing=2.9457733035087585e-06
  • pathlib=1.2619420886039734e-06
  • ezpz=6.109476089477539e-07
  • torch=3.5976991057395935e-06
  • torch_ddp=2.3636966943740845e-06
  • wandb=6.36400654911995e-05
  • total=7.597357034683228e-05


[2024-07-15 12:46:26.453718][INFO][dist:95] -

[CONFIG]:
  • warmup=0
  • log_freq=1
  • batch_size=64
  • input_size=128
  • output_size=128
  • dtype=torch.float32
  • device=cuda
  • world_size=4
  • train_iters=100


[2024-07-15 12:46:28.048558][INFO][test_dist:183] - model=Network(
  (layers): Sequential(
    (0): Linear(in_features=128, out_features=1024, bias=True)
    (1): Linear(in_features=1024, out_features=512, bias=True)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): Linear(in_features=128, out_features=128, bias=True)
  )
)
[2024-07-15 12:46:30.303706][INFO][test_dist:274] - iter=1, loss=1977.36, sps=1.829e+04, dt=0.00349948, dtf=0.00097, dtb=0.00253
[2024-07-15 12:46:30.307445][INFO][test_dist:274] - iter=2, loss=1521.61, sps=3.179e+04, dt=0.00201315, dtf=0.0005266, dtb=0.001487
[2024-07-15 12:46:30.310480][INFO][test_dist:274] - iter=3, loss=1101.45, sps=3.636e+04, dt=0.00176022, dtf=0.0005615, dtb=0.001199
[2024-07-15 12:46:30.313460][INFO][test_dist:274] - iter=4, loss=907.963, sps=3.725e+04, dt=0.00171805, dtf=0.0005335, dtb=0.001185
[2024-07-15 12:46:30.316456][INFO][test_dist:274] - iter=5, loss=842.827, sps=3.698e+04, dt=0.0017307, dtf=0.0005462, dtb=0.001184
[2024-07-15 12:46:30.319577][INFO][test_dist:274] - iter=6, loss=781.918, sps=3.443e+04, dt=0.00185873, dtf=0.0006062, dtb=0.001253
[2024-07-15 12:46:30.322847][INFO][test_dist:274] - iter=7, loss=778.317, sps=3.253e+04, dt=0.00196762, dtf=0.0005706, dtb=0.001397
[2024-07-15 12:46:30.325813][INFO][test_dist:274] - iter=8, loss=746.629, sps=3.746e+04, dt=0.00170848, dtf=0.0005085, dtb=0.0012
[2024-07-15 12:46:30.328695][INFO][test_dist:274] - iter=9, loss=736.385, sps=3.913e+04, dt=0.00163572, dtf=0.0004885, dtb=0.001147
[2024-07-15 12:46:30.331672][INFO][test_dist:274] - iter=10, loss=733.067, sps=3.713e+04, dt=0.00172381, dtf=0.0004742, dtb=0.00125
[2024-07-15 12:46:30.334841][INFO][test_dist:274] - iter=11, loss=729.301, sps=3.342e+04, dt=0.00191528, dtf=0.0003869, dtb=0.001528
[2024-07-15 12:46:30.338105][INFO][test_dist:274] - iter=12, loss=701.243, sps=3.198e+04, dt=0.00200144, dtf=0.0004697, dtb=0.001532
[2024-07-15 12:46:30.340953][INFO][test_dist:274] - iter=13, loss=681.563, sps=4.088e+04, dt=0.00156543, dtf=0.0004341, dtb=0.001131
[2024-07-15 12:46:30.343843][INFO][test_dist:274] - iter=14, loss=685.259, sps=4.053e+04, dt=0.00157891, dtf=0.0004456, dtb=0.001133
[2024-07-15 12:46:30.346744][INFO][test_dist:274] - iter=15, loss=678.99, sps=3.935e+04, dt=0.00162651, dtf=0.0004026, dtb=0.001224
[2024-07-15 12:46:30.349674][INFO][test_dist:274] - iter=16, loss=672.654, sps=3.801e+04, dt=0.00168398, dtf=0.0004565, dtb=0.001227
[2024-07-15 12:46:30.352400][INFO][test_dist:274] - iter=17, loss=660.552, sps=4.319e+04, dt=0.00148184, dtf=0.0003963, dtb=0.001085
[2024-07-15 12:46:30.355521][INFO][test_dist:274] - iter=18, loss=640.175, sps=3.408e+04, dt=0.00187814, dtf=0.0005312, dtb=0.001347
[2024-07-15 12:46:30.358530][INFO][test_dist:274] - iter=19, loss=645.41, sps=3.65e+04, dt=0.00175325, dtf=0.0004639, dtb=0.001289
[2024-07-15 12:46:30.361515][INFO][test_dist:274] - iter=20, loss=649.203, sps=3.675e+04, dt=0.00174155, dtf=0.0004799, dtb=0.001262
[2024-07-15 12:46:30.364256][INFO][test_dist:274] - iter=21, loss=633.891, sps=4.277e+04, dt=0.00149629, dtf=0.000416, dtb=0.00108
[2024-07-15 12:46:30.367135][INFO][test_dist:274] - iter=22, loss=623.202, sps=3.892e+04, dt=0.00164444, dtf=0.0004509, dtb=0.001194
[2024-07-15 12:46:30.370090][INFO][test_dist:274] - iter=23, loss=623.178, sps=3.771e+04, dt=0.00169695, dtf=0.0004252, dtb=0.001272
[2024-07-15 12:46:30.373119][INFO][test_dist:274] - iter=24, loss=626.489, sps=3.69e+04, dt=0.00173444, dtf=0.0004551, dtb=0.001279
[2024-07-15 12:46:30.375951][INFO][test_dist:274] - iter=25, loss=636.674, sps=4.089e+04, dt=0.0015651, dtf=0.0004223, dtb=0.001143
[2024-07-15 12:46:30.378913][INFO][test_dist:274] - iter=26, loss=639.64, sps=3.758e+04, dt=0.00170305, dtf=0.0004532, dtb=0.00125
[2024-07-15 12:46:30.381808][INFO][test_dist:274] - iter=27, loss=605.015, sps=3.874e+04, dt=0.00165192, dtf=0.0004257, dtb=0.001226
[2024-07-15 12:46:30.384573][INFO][test_dist:274] - iter=28, loss=603.894, sps=4.244e+04, dt=0.00150807, dtf=0.0004296, dtb=0.001078
[2024-07-15 12:46:30.387388][INFO][test_dist:274] - iter=29, loss=619.885, sps=4.03e+04, dt=0.00158808, dtf=0.0004196, dtb=0.001168
[2024-07-15 12:46:30.390148][INFO][test_dist:274] - iter=30, loss=589.771, sps=4.21e+04, dt=0.0015203, dtf=0.0004438, dtb=0.001076
[2024-07-15 12:46:30.392838][INFO][test_dist:274] - iter=31, loss=595.523, sps=4.381e+04, dt=0.001461, dtf=0.0004153, dtb=0.001046
[2024-07-15 12:46:30.395622][INFO][test_dist:274] - iter=32, loss=605.537, sps=4.104e+04, dt=0.00155956, dtf=0.0004367, dtb=0.001123
[2024-07-15 12:46:30.398489][INFO][test_dist:274] - iter=33, loss=586.025, sps=3.913e+04, dt=0.00163565, dtf=0.0003743, dtb=0.001261
[2024-07-15 12:46:30.401355][INFO][test_dist:274] - iter=34, loss=577.14, sps=3.877e+04, dt=0.0016506, dtf=0.0004581, dtb=0.001192
[2024-07-15 12:46:30.404092][INFO][test_dist:274] - iter=35, loss=568.886, sps=4.383e+04, dt=0.00146019, dtf=0.0003966, dtb=0.001064
[2024-07-15 12:46:30.406843][INFO][test_dist:274] - iter=36, loss=567.26, sps=4.228e+04, dt=0.00151377, dtf=0.0004537, dtb=0.00106
[2024-07-15 12:46:30.409591][INFO][test_dist:274] - iter=37, loss=574.633, sps=4.226e+04, dt=0.00151457, dtf=0.0003845, dtb=0.00113
[2024-07-15 12:46:30.412347][INFO][test_dist:274] - iter=38, loss=557.928, sps=4.212e+04, dt=0.00151945, dtf=0.000457, dtb=0.001062
[2024-07-15 12:46:30.415130][INFO][test_dist:274] - iter=39, loss=558.186, sps=4.135e+04, dt=0.00154767, dtf=0.0003975, dtb=0.00115
[2024-07-15 12:46:30.417880][INFO][test_dist:274] - iter=40, loss=553.377, sps=4.233e+04, dt=0.00151185, dtf=0.0004556, dtb=0.001056
[2024-07-15 12:46:30.420583][INFO][test_dist:274] - iter=41, loss=542.918, sps=4.394e+04, dt=0.00145645, dtf=0.0003928, dtb=0.001064
[2024-07-15 12:46:30.423489][INFO][test_dist:274] - iter=42, loss=547.64, sps=3.827e+04, dt=0.00167242, dtf=0.0004762, dtb=0.001196
[2024-07-15 12:46:30.426243][INFO][test_dist:274] - iter=43, loss=546.106, sps=4.22e+04, dt=0.00151648, dtf=0.0004665, dtb=0.00105
[2024-07-15 12:46:30.428998][INFO][test_dist:274] - iter=44, loss=535.946, sps=4.209e+04, dt=0.0015204, dtf=0.0004679, dtb=0.001053
[2024-07-15 12:46:30.431749][INFO][test_dist:274] - iter=45, loss=534.731, sps=4.324e+04, dt=0.00148002, dtf=0.0003821, dtb=0.001098
[2024-07-15 12:46:30.434709][INFO][test_dist:274] - iter=46, loss=520.207, sps=3.74e+04, dt=0.00171109, dtf=0.0004486, dtb=0.001263
[2024-07-15 12:46:30.437524][INFO][test_dist:274] - iter=47, loss=527.301, sps=4.191e+04, dt=0.00152713, dtf=0.0004943, dtb=0.001033
[2024-07-15 12:46:30.440277][INFO][test_dist:274] - iter=48, loss=516.108, sps=4.279e+04, dt=0.00149561, dtf=0.0004418, dtb=0.001054
[2024-07-15 12:46:30.443029][INFO][test_dist:274] - iter=49, loss=516.086, sps=4.231e+04, dt=0.00151262, dtf=0.0003899, dtb=0.001123
[2024-07-15 12:46:30.445895][INFO][test_dist:274] - iter=50, loss=508.945, sps=3.922e+04, dt=0.00163198, dtf=0.0004333, dtb=0.001199
[2024-07-15 12:46:30.448667][INFO][test_dist:274] - iter=51, loss=513.235, sps=4.163e+04, dt=0.00153721, dtf=0.0004863, dtb=0.001051
[2024-07-15 12:46:30.451581][INFO][test_dist:274] - iter=52, loss=511.549, sps=3.822e+04, dt=0.00167444, dtf=0.0004698, dtb=0.001205
[2024-07-15 12:46:30.454412][INFO][test_dist:274] - iter=53, loss=510.211, sps=4.042e+04, dt=0.00158328, dtf=0.0004193, dtb=0.001164
[2024-07-15 12:46:30.457220][INFO][test_dist:274] - iter=54, loss=492.431, sps=4.151e+04, dt=0.00154189, dtf=0.0004324, dtb=0.001109
[2024-07-15 12:46:30.460039][INFO][test_dist:274] - iter=55, loss=499.074, sps=4.121e+04, dt=0.00155284, dtf=0.0005013, dtb=0.001052
[2024-07-15 12:46:30.462836][INFO][test_dist:274] - iter=56, loss=490.718, sps=4.108e+04, dt=0.0015581, dtf=0.0004388, dtb=0.001119
[2024-07-15 12:46:30.465621][INFO][test_dist:274] - iter=57, loss=493.223, sps=4.157e+04, dt=0.00153946, dtf=0.0003867, dtb=0.001153
[2024-07-15 12:46:30.468390][INFO][test_dist:274] - iter=58, loss=486.601, sps=4.256e+04, dt=0.00150368, dtf=0.0004462, dtb=0.001057
[2024-07-15 12:46:30.471148][INFO][test_dist:274] - iter=59, loss=473.609, sps=4.326e+04, dt=0.00147947, dtf=0.0004221, dtb=0.001057
[2024-07-15 12:46:30.473847][INFO][test_dist:274] - iter=60, loss=478.577, sps=4.378e+04, dt=0.00146188, dtf=0.0004284, dtb=0.001033
[2024-07-15 12:46:30.476547][INFO][test_dist:274] - iter=61, loss=475.991, sps=4.387e+04, dt=0.00145878, dtf=0.0003923, dtb=0.001066
[2024-07-15 12:46:30.479447][INFO][test_dist:274] - iter=62, loss=471.526, sps=3.859e+04, dt=0.00165854, dtf=0.0004523, dtb=0.001206
[2024-07-15 12:46:30.482180][INFO][test_dist:274] - iter=63, loss=460.986, sps=4.277e+04, dt=0.00149626, dtf=0.0003914, dtb=0.001105
[2024-07-15 12:46:30.484857][INFO][test_dist:274] - iter=64, loss=464.409, sps=4.404e+04, dt=0.00145311, dtf=0.0004247, dtb=0.001028
[2024-07-15 12:46:30.487652][INFO][test_dist:274] - iter=65, loss=458.023, sps=4.104e+04, dt=0.00155961, dtf=0.0003934, dtb=0.001166
[2024-07-15 12:46:30.490474][INFO][test_dist:274] - iter=66, loss=456.718, sps=4.013e+04, dt=0.00159499, dtf=0.000416, dtb=0.001179
[2024-07-15 12:46:30.493187][INFO][test_dist:274] - iter=67, loss=451.993, sps=4.296e+04, dt=0.00148977, dtf=0.000395, dtb=0.001095
[2024-07-15 12:46:30.495894][INFO][test_dist:274] - iter=68, loss=454.66, sps=4.4e+04, dt=0.0014545, dtf=0.000425, dtb=0.001029
[2024-07-15 12:46:30.498664][INFO][test_dist:274] - iter=69, loss=451.727, sps=4.17e+04, dt=0.00153459, dtf=0.0003837, dtb=0.001151
[2024-07-15 12:46:30.501502][INFO][test_dist:274] - iter=70, loss=440.922, sps=4.015e+04, dt=0.00159391, dtf=0.0004272, dtb=0.001167
[2024-07-15 12:46:30.504271][INFO][test_dist:274] - iter=71, loss=442.788, sps=4.322e+04, dt=0.00148083, dtf=0.0004178, dtb=0.001063
[2024-07-15 12:46:30.507000][INFO][test_dist:274] - iter=72, loss=439.069, sps=4.307e+04, dt=0.00148594, dtf=0.0004285, dtb=0.001057
[2024-07-15 12:46:30.509755][INFO][test_dist:274] - iter=73, loss=430.236, sps=4.211e+04, dt=0.00151976, dtf=0.0003829, dtb=0.001137
[2024-07-15 12:46:30.512494][INFO][test_dist:274] - iter=74, loss=428.951, sps=4.318e+04, dt=0.00148219, dtf=0.0004357, dtb=0.001046
[2024-07-15 12:46:30.515244][INFO][test_dist:274] - iter=75, loss=430.417, sps=4.253e+04, dt=0.00150487, dtf=0.0004357, dtb=0.001069
[2024-07-15 12:46:30.517921][INFO][test_dist:274] - iter=76, loss=416.647, sps=4.401e+04, dt=0.00145412, dtf=0.0004374, dtb=0.001017
[2024-07-15 12:46:30.520610][INFO][test_dist:274] - iter=77, loss=422.518, sps=4.468e+04, dt=0.0014323, dtf=0.0003941, dtb=0.001038
[2024-07-15 12:46:30.523343][INFO][test_dist:274] - iter=78, loss=412.028, sps=4.272e+04, dt=0.00149821, dtf=0.0004521, dtb=0.001046
[2024-07-15 12:46:30.526016][INFO][test_dist:274] - iter=79, loss=406.225, sps=4.473e+04, dt=0.00143094, dtf=0.0003893, dtb=0.001042
[2024-07-15 12:46:30.528730][INFO][test_dist:274] - iter=80, loss=402.887, sps=4.382e+04, dt=0.00146036, dtf=0.0004402, dtb=0.00102
[2024-07-15 12:46:30.531536][INFO][test_dist:274] - iter=81, loss=397.311, sps=4.069e+04, dt=0.0015729, dtf=0.000404, dtb=0.001169
[2024-07-15 12:46:30.534242][INFO][test_dist:274] - iter=82, loss=411.916, sps=4.39e+04, dt=0.00145795, dtf=0.0004351, dtb=0.001023
[2024-07-15 12:46:30.537005][INFO][test_dist:274] - iter=83, loss=402.795, sps=4.366e+04, dt=0.00146599, dtf=0.0004252, dtb=0.001041
[2024-07-15 12:46:30.539805][INFO][test_dist:274] - iter=84, loss=391.05, sps=4.1e+04, dt=0.00156095, dtf=0.0004323, dtb=0.001129
[2024-07-15 12:46:30.542590][INFO][test_dist:274] - iter=85, loss=383.782, sps=4.118e+04, dt=0.00155416, dtf=0.000388, dtb=0.001166
[2024-07-15 12:46:30.545290][INFO][test_dist:274] - iter=86, loss=399.543, sps=4.396e+04, dt=0.00145595, dtf=0.0004339, dtb=0.001022
[2024-07-15 12:46:30.547992][INFO][test_dist:274] - iter=87, loss=379.003, sps=4.456e+04, dt=0.00143613, dtf=0.0004131, dtb=0.001023
[2024-07-15 12:46:30.550837][INFO][test_dist:274] - iter=88, loss=372.048, sps=3.998e+04, dt=0.00160092, dtf=0.0004375, dtb=0.001163
[2024-07-15 12:46:30.553641][INFO][test_dist:274] - iter=89, loss=376.187, sps=4.137e+04, dt=0.0015471, dtf=0.0004142, dtb=0.001133
[2024-07-15 12:46:30.556354][INFO][test_dist:274] - iter=90, loss=372.281, sps=4.408e+04, dt=0.00145186, dtf=0.0004239, dtb=0.001028
[2024-07-15 12:46:30.559101][INFO][test_dist:274] - iter=91, loss=370.701, sps=4.252e+04, dt=0.00150523, dtf=0.0004545, dtb=0.001051
[2024-07-15 12:46:30.561884][INFO][test_dist:274] - iter=92, loss=356.074, sps=4.191e+04, dt=0.00152712, dtf=0.0004291, dtb=0.001098
[2024-07-15 12:46:30.564556][INFO][test_dist:274] - iter=93, loss=360.663, sps=4.468e+04, dt=0.00143241, dtf=0.0003938, dtb=0.001039
[2024-07-15 12:46:30.567287][INFO][test_dist:274] - iter=94, loss=374.599, sps=4.296e+04, dt=0.0014897, dtf=0.0004539, dtb=0.001036
[2024-07-15 12:46:30.570088][INFO][test_dist:274] - iter=95, loss=364.476, sps=4.253e+04, dt=0.00150469, dtf=0.0004189, dtb=0.001086
[2024-07-15 12:46:30.572767][INFO][test_dist:274] - iter=96, loss=358.992, sps=4.415e+04, dt=0.00144967, dtf=0.0004295, dtb=0.00102
[2024-07-15 12:46:30.575567][INFO][test_dist:274] - iter=97, loss=354.959, sps=4.092e+04, dt=0.0015641, dtf=0.0004054, dtb=0.001159
[2024-07-15 12:46:30.578287][INFO][test_dist:274] - iter=98, loss=345.644, sps=4.374e+04, dt=0.00146311, dtf=0.0004197, dtb=0.001043
[2024-07-15 12:46:30.580996][INFO][test_dist:274] - iter=99, loss=348.209, sps=4.391e+04, dt=0.00145758, dtf=0.0004263, dtb=0.001031
                             train/dt [2024-07-15-124630]
       ┌───────────────────────────────────────────────────────────────────────┐
0.00350┤▘                                                                      │
       │                                                                       │
       │                                                                       │
0.00315┤                                                                       │
       │                                                                       │
       │                                                                       │
0.00281┤                                                                       │
       │                                                                       │
0.00247┤                                                                       │
       │                                                                       │
       │                                                                       │
0.00212┤                                                                       │
       │▗       ▖                                                              │
       │    ▝  ▖    ▖                                                          │
0.00178┤ ▗ ▝                                                                   │
       │  ▘▘ ▘▝    ▖ ▀ ▖▀ ▚    ▗     ▗  ▝   ▗       ▖                          │
       │      ▘ ▗▝▘      ▗  ▘▖▗▘   ▗   ▖ ▖ ▝▖▝▖▄▗     ▖▘ ▞  ▖    ▗ ▗▗ ▝▗ ▗   ▖ │
0.00143┤           ▝  ▝    ▝ ▝  ▗▘▀ ▘▖▘▝  ▀      ▀▝▖▝▗ ▝▖ ▝▘▝▘▄▝▖▖▗▘ ▖▖ ▞ ▖▀▗ ▚│
       └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
       1.0              25.5             50.0              74.5            99.0
train/dt                                 iter
[2024-07-15 12:46:30.627053][INFO][plot:156] - Appending plot to: /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/dt.txt
text saved in /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/dt.txt
                             train/dtf [2024-07-15-124630]
       ┌───────────────────────────────────────────────────────────────────────┐
0.00097┤▘                                                                      │
       │                                                                       │
       │                                                                       │
0.00087┤                                                                       │
       │                                                                       │
       │                                                                       │
0.00077┤                                                                       │
       │                                                                       │
0.00067┤                                                                       │
       │                                                                       │
       │   ▗                                                                   │
0.00057┤    ▗                                                                  │
       │ ▝ ▖                                                                   │
       │▝ ▘  ▖      ▘                                                          │
0.00047┤      ▚ ▖    ▗               ▗ ▖ ▘  ▚  ▘                               │
       │        ▗▗ ▘ ▘ ▘▝ ▘  ▖▗▝ ▘▝ ▘ ▘ ▗ ▖    ▗ ▖  ▘       ▗▖▖▝ ▖▗  ▖▗ ▝  ▘   │
       │          ▖   ▗ ▘▝▝▝▘▗             ▝ ▝▘  ▝▝  ▝ ▖▘▝▝▘     ▗ ▀  ▖▗▘▝ ▝▝▖▀│
0.00037┤       ▘   ▝           ▖▝ ▘▝ ▘ ▗  ▝     ▝  ▘▝ ▘▝ ▘  ▘ ▝ ▘   ▝     ▘    │
       └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
       1.0              25.5             50.0              74.5            99.0
train/dtf                                iter
[2024-07-15 12:46:30.638406][INFO][plot:156] - Appending plot to: /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/dtf.txt
text saved in /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/dtf.txt
                             train/dtb [2024-07-15-124630]
       ┌───────────────────────────────────────────────────────────────────────┐
0.00253┤▘                                                                      │
       │                                                                       │
       │                                                                       │
0.00228┤                                                                       │
       │                                                                       │
       │                                                                       │
0.00203┤                                                                       │
       │                                                                       │
0.00177┤                                                                       │
       │                                                                       │
       │                                                                       │
0.00152┤       ▖▖                                                              │
       │▝                                                                      │
       │    ▝       ▖                                                          │
0.00127┤             ▄  ▄      ▖        ▗                                      │
       │ ▗▖▞ ▖▝   ▘▘   ▖  ▀ ▖  ▗     ▗     ▗▗       ▖  ▖         ▗             │
       │      ▘ ▝▝ ▗     ▝    ▗   ▘▝   ▗  ▗  ▝▖▗▝   ▗ ▘▗ ▀  ▘      ▝▝ ▝▝ ▗ ▗ ▘ │
0.00102┤              ▝    ▝ ▀  ▝▘▝ ▘▘▘▘ ▖▘ ▘  ▘ ▀▗▘ ▗  ▖ ▝▘▝▘▄▝▘▖▗▘ ▖▖ ▞ ▘▖▗ ▚│
       └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
       1.0              25.5             50.0              74.5            99.0
train/dtb                                iter
[2024-07-15 12:46:30.648950][INFO][plot:156] - Appending plot to: /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/dtb.txt
text saved in /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/dtb.txt
                            train/loss [2024-07-15-124630]
      ┌────────────────────────────────────────────────────────────────────────┐
1977.4┤▘                                                                       │
      │                                                                        │
      │                                                                        │
1705.4┤                                                                        │
      │                                                                        │
      │▝                                                                       │
1433.5┤                                                                        │
      │                                                                        │
      │                                                                        │
1161.5┤ ▗                                                                      │
      │                                                                        │
 889.5┤  ▖                                                                     │
      │   ▘                                                                    │
      │   ▝▝▘▄▗▖                                                               │
 617.6┤         ▀▘▀▗▖▚▗▖▄▖▄▗ ▗                                                 │
      │                     ▘▘▝▘▀▝▀▗▖▄▗▖▄▗▖▄▖▖                                 │
      │                                      ▝▝▘▀▝▘▀▖▚▗▖▄▗▖▄▗▄▗                │
 345.6┤                                                        ▘▀▝▘▀▝▀▝▘▀▗▖▚▗▖▄│
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
      1.0              25.5              50.0             74.5             99.0
train/loss                               iter
[2024-07-15 12:46:30.702717][INFO][plot:156] - Appending plot to: /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/loss.txt
text saved in /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/loss.txt
                           train/iter [2024-07-15-124630]
    ┌──────────────────────────────────────────────────────────────────────────┐
99.0┤                                                                      ▗▗▖▀│
    │                                                                   ▄▝▘▘   │
    │                                                              ▗▖▞▝▘       │
82.7┤                                                          ▄▗▘▀            │
    │                                                      ▖▄▝▘                │
    │                                                 ▗▗▖▀▝                    │
66.3┤                                              ▄▝▘▘                        │
    │                                         ▗▖▞▝▘                            │
50.0┤                                     ▄▗▘▀                                 │
    │                                 ▖▄▝▘                                     │
    │                            ▗▗▖▀▝                                         │
33.7┤                         ▄▝▘▘                                             │
    │                    ▗▖▞▝▘                                                 │
    │                ▄▗▘▀                                                      │
17.3┤            ▖▄▝▘                                                          │
    │       ▗▗▖▀▝                                                              │
    │    ▄▝▘▘                                                                  │
 1.0┤▖▞▝▘                                                                      │
    └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
    1.0              25.5               50.0              74.5             99.0
train/iter                              iter
[2024-07-15 12:46:30.714042][INFO][plot:156] - Appending plot to: /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/iter.txt
text saved in /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/iter.txt
                             train/sps [2024-07-15-124630]
       ┌───────────────────────────────────────────────────────────────────────┐
44725.9┤                     ▗  ▗    ▖            ▗▖ ▗  ▖     ▞ ▘▖▗▖ ▖▘ ▖ ▘ ▗ ▄│
       │           ▝  ▗    ▗ ▖   ▖▄ ▖ ▖▞  ▄      ▞  ▗  ▝  ▝▘▞▖ ▗        ▗  ▚   │
       │        ▗        ▗    ▗    ▝     ▘  ▘ ▘▄▝     ▖  ▘       ▗ ▗▗  ▝ ▝   ▖ │
40319.7┤      ▖  ▝▖         ▘  ▖           ▗ ▝         ▘ ▝            ▝        │
       │           ▖   ▘▖ ▞    ▝     ▗      ▗       ▘                          │
       │ ▗▘▘ ▘▝      ▄  ▝               ▝                                      │
35913.4┤                                                                       │
       │   ▝   ▖    ▘                                                          │
31507.2┤▗   ▝   ▖                                                              │
       │                                                                       │
       │                                                                       │
27100.9┤                                                                       │
       │                                                                       │
       │                                                                       │
22694.7┤                                                                       │
       │                                                                       │
       │                                                                       │
18288.4┤▖                                                                      │
       └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
       1.0              25.5             50.0              74.5            99.0
train/sps                                iter
[2024-07-15 12:46:30.724919][INFO][plot:156] - Appending plot to: /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/sps.txt
text saved in /home/foremans/tmp/foremans/2024-07-15-124441/test-dist-plots/train/sps.txt

  _     ._   __/__   _ _  _  _ _/_   Recorded: 12:46:28  Samples:  2182
 /_//_/// /_\ / //_// / //_'/ //     Duration: 2.688     CPU time: 2.356
/   _/                      v4.6.2

Program: /home/foremans/tmp/foremans/2024-07-15-124441/venvs/2024-04-29/src/ezpz/src/ezpz/test_dist.py

2.688 <module>  ezpz/test_dist.py:1
└─ 2.687 main  ezpz/test_dist.py:217
   ├─ 2.104 build_model_and_optimizer  ezpz/test_dist.py:171
   │  └─ 2.089 Adam.__init__  torch/optim/adam.py:15
   │        [142 frames hidden]  torch, transformers, jax, huggingface...
   ├─ 0.199 _backward_step  ezpz/test_dist.py:236
   │  ├─ 0.104 Tensor.backward  torch/_tensor.py:466
   │  │     [4 frames hidden]  torch, <built-in>
   │  └─ 0.094 wrapper  torch/optim/optimizer.py:374
   │        [6 frames hidden]  torch, <built-in>
   ├─ 0.145 tplot_dict  ezpz/plot.py:136
   │  ├─ 0.085 show  plotext/_core.py:292
   │  │     [5 frames hidden]  plotext
   │  └─ 0.031 <module>  plotext/__init__.py:1
   ├─ 0.136 _forward_step  ezpz/test_dist.py:231
   │  ├─ 0.084 DistributedDataParallel._wrapped_call_impl  torch/nn/modules/module.py:1528
   │  │     [6 frames hidden]  torch
   │  │        0.071 Network._call_impl  torch/nn/modules/module.py:1534
   │  │        └─ 0.071 Network.forward  ezpz/test_dist.py:164
   │  │           └─ 0.071 Sequential._wrapped_call_impl  torch/nn/modules/module.py:1528
   │  │                 [7 frames hidden]  torch, <built-in>
   │  └─ 0.052 calc_loss  ezpz/test_dist.py:168
   └─ 0.100 Logger.info  logging/__init__.py:1479
         [6 frames hidden]  logging, rich
            0.100 RichHandler.emit  rich/logging.py:126
            └─ 0.098 Console.print  ezpz/log/console.py:79
               └─ 0.098 Console.print  rich/console.py:1624
                     [4 frames hidden]  rich


[2024-07-15 12:46:30.921061][INFO][profile:115] - Saving pyinstrument profile output to: /home/foremans/tmp/foremans/2024-07-15-124441/ezpz_pyinstrument_profiles
[2024-07-15 12:46:30.921555][INFO][profile:123] - PyInstrument profile saved (as html) to:  /home/foremans/tmp/foremans/2024-07-15-124441/ezpz_pyinstrument_profiles/pyinstrument-profile-2024-07-15-124630.html
[2024-07-15 12:46:30.922034][INFO][profile:131] - PyInstrument profile saved (as text) to:  /home/foremans/tmp/foremans/2024-07-15-124441/ezpz_pyinstrument_profiles/pyinstrument-profile-2024-07-15-124630.txt
[2024-07-15 12:46:31.432464][INFO][profile:143] - Finished with pyinstrument profiler. Took: 2.68764s
[2024-07-15 12:46:31.433122][INFO][test_dist:318] - [0] runtime=6.820802s
Application 5ccc89be resources: utime=21s stime=21s maxrss=1383056KB inblock=8080 oublock=3456 minflt=659443 majflt=896 nvcsw=192077 nivcsw=672014
```

[^1]: `deepspeed`, `DDP` only support `pytorch`

[^2]: If necessary, otherwise activate if already exists

[^3]: Note that the virtual environment will be created at
    `./venvs/${CONDA_NAME}`, where `${CONDA_NAME}` will match the prefix
    of the active `conda` environment
