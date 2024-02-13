---
# sidebar: false
title: "Megatron DeepSpeed on `xpu`"
callout-appearance: simple
title-block-banner: false
citeproc: true
citation:
   author: Sam Foreman
   type: webpage
   title: "Megatron DeepSpeed on `xpu`"
   url: https://samforeman.me/qmd/aurora-gpt/megatron-ds-intel.html
# https://www.anl.gov/sites/www/files/2021-09/CPA_RESIZE_Climate%20Resilience%20Images_01_1920x1080.jpg
editor:
   render-on-save: true
execute:
   freeze: auto
twitter-card:
    image: "./assets/thumbnail.png"
    creator: "@saforem2"
    site: "@saforem2"
open-graph:
    image: "./assets/thumbnail.png"
format:
  # html:
  html:
    callout-appearance: simple
    code-link: true
    code-line-numbers: false
    toc: true
    self-contained: false
    toc-location: right
    # grid:
    #   body-width: 1150px
    fig-responsive: true
    anchor-sections: true
    highlight-style: atom-one
    code-overflow: scroll
    # code-fold: false
    code-copy: true
    code-summary: " "
    code-tools:
      source: repo
      toggle: true
      caption: none
    html-math-method: katex
    css:
      - /css/default.css
      - /css/callouts.css
      - /css/profile.css
    theme:
      dark:
        - /css/syntax-dark.scss
        - /css/dark.scss 
        - /css/common.scss
      light:
        - /css/syntax-light.scss
        - /css/light.scss
        - /css/common.scss
    # smooth-scroll: true
    citations-hover: true
    footnotes-hover: true
    header-includes: |
      <link href="https://pvinis.github.io/iosevka-webfont/3.4.1/iosevka.css" rel="stylesheet" />
  gfm:
    author: Sam Foreman
    output-file: "megatron-ds-intel.md"
---

## Install / Setup

- Setup script / history:

    <details closed><summary>Interactive Session</summary>

    ```bash
    $ export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
    $ export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
    $ export http_proxy=http://proxy.alcf.anl.gov:3128
    $ export https_proxy=http://proxy.alcf.anl.gov:3128

    $ export DRM_LIB="$(pwd)/usr/include/libdrm"
    # $ export PATH="${HOME}/miniconda3/bin:$PATH"

    $ conda create --name anl_release_q4 python=3.9 --y
    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    $ eval "$(${HOME} shell.zsh hook)"

    $ export DRM_LIB="$(pwd)/usr/include/libdrm"
    $ conda config --add channels conda-forge && conda install -c conda-forge mpi4py -y --no-deps && conda install -c conda-forge libssh -y && conda uninstall mpi -y && python3 -m pip install -r requirements.txt && python3 -m pip install *.whl

    $ module unload oneapi/eng-compiler/2022.12.30.003
    $ module unload intel_compute_runtime/release/agama-devel-551
    $ module use -a /soft/modulefiles
    $ module load oneapi/release/2023.12.15.001
    $ module use /home/ftartagl/graphics-compute-runtime/modulefiles
    $ module load graphics-compute-runtime/agama-ci-devel-736.9
    # one-liner for modules:
    # module unload oneapi/eng-compiler/2022.12.30.003 && module unload intel_compute_runtime/release/agama-devel-551&& module use -a /soft/modulefiles && module load oneapi/release/2023.12.15.001 && module use /home/ftartagl/graphics-compute-runtime/modulefiles && module load graphics-compute-runtime/agama-ci-devel-736.9

    $ cd torch-ccl
    $ ls
    $ COMPUTE_BACKEND=dpcpp python3 setup.py develop |& tee build.log
    $ cd ../

    $ cd intel-extension-for-deepspeed
    $ python3 setup.py develop |& tee build.log
    $ cd ../

    $ cd DeepSpeed
    $ ls
    $ python3 -m pip install -r requirements/requirements.txt
    $ python3 setup.py develop |& tee build.log
    ```

    </details>

## Running

### Using `launch`


- Setup:

    <details closed><summary>Setup</summary>

    ```bash
    $ qsub -q EarlyAppAccess -A Aurora_Deployment -l walltime=08:00:00 -l select=2 -I
    qsub: waiting for job 604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov to start
    qsub: job 604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov ready

    $ module unload oneapi/eng-compiler/2022.12.30.003 && module unload intel_compute_runtime/release/agama-devel-551&& module use -a /soft/modulefiles && module load oneapi/release/2023.12.15.001 && module use /home/ftartagl/graphics-compute-runtime/modulefiles && module load graphics-compute-runtime/agama-ci-devel-736.9
     UMD: agama-ci-devel-736.9 successfully loaded:
     UMD: graphics-compute-runtime/agama-ci-devel-736.9

    $ eval "$(/home/foremans/miniconda3/bin/conda shell.zsh hook)"

    $ conda activate anl_release_q4

    # [BUG] for some reason, need to run twice ¯\_(ツ)_/¯
    $ git clone https://github.com/saforem2/ezpz
    $ python3 -m pip install -e "ezpz[dev]"
    $ source ./ezpz/src/ezpz/bin/savejobenv && source ./ezpz/src/ezpz/bin/savejobenv
    ```

    <details closed><summary>Output</summary>

    ```bash
    ┌───────────────────────────────────────────────────────────────────
    │ Writing PBS vars to /home/foremans/.pbsenv
    │ HOSTFILE: /var/spool/pbs/aux/604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    │ NHOSTS: 2
    │ NGPU_PER_HOST: 12 GPUs per host
    │ NGPUS: 24 GPUs total
    └───────────────────────────────────────────────────────────────────
    ┌───────────────────────────────────────────────────────────────────
    │ [DIST INFO]:
    │   • Writing Job info to /home/foremans/.pbsenv
    │     • HOSTFILE: /var/spool/pbs/aux/604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    │     • NHOSTS: 2
    │     • NGPU_PER_HOST: 12
    │     • NGPUS = (NHOSTS * NGPU_PER_HOST) = 24
    │ [Hosts]:
    │       • x4502c0s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov, x4502c0s2b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov
    │ [Launch]:
    │     • Use: 'launch' (=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov)
    │       to launch job
    └───────────────────────────────────────────────────────────────────
    ┌────────────────────────────────────────────────────────────────────────────────
    │ YOU ARE HERE: /home/foremans
    │ Run 'source ./bin/getjobenv' in a NEW SHELL to automatically set env vars
    └────────────────────────────────────────────────────────────────────────────────
    ```

    </details>


    <details closed><summary>[WIP] Building out `python` API</summary>

    ```bash
    $ python3 -m ezpz.savejobenv
    /home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 'If you dont plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?'
      warn(
    [2024-01-23 10:02:37][INFO][jobs:185] - Saving job env to /home/foremans/PBS-jobs/604319/jobenv.sh
    [2024-01-23 10:02:37][INFO][jobs:193] - Saving job env to dot-env (.env) file in /home/foremans
    [2024-01-23 10:02:37][INFO][jobs:211] - Saving job env to /home/foremans/PBS-jobs/604319/jobenv.json
    [2024-01-23 10:02:37][INFO][jobs:225] - Saving job env to /home/foremans/PBS-jobs/604319/jobenv.yaml
    [2024-01-23 10:02:37][INFO][jobs:253] - Writing PBS env vars to /home/foremans/PBS-jobs/604319 / jobenv{.sh, .yaml, .json}
    [2024-01-23 10:02:37][INFO][jobs:258] - jobenv={
        "BACKEND": "gloo",
        "DEVICE": "xpu",
        "DEVICE_ID": "xpu:0",
        "DISTRIBUTED_BACKEND": "gloo",
        "FRAMEWORK": "pytorch",
        "GPUS_PER_NODE": 12,
        "HOSTFILE": "/var/spool/pbs/aux/604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        "HOSTNAME": "x4502c0s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov",
        "HOSTS": "[x4502c0s0b0n0, x4502c0s2b0n0]",
        "LAUNCH_CMD": "mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        "LOCAL_RANK": 0,
        "MACHINE": "Aurora",
        "NGPUS": 24,
        "NGPU_PER_HOST": "12",
        "NHOSTS": "2",
        "NODE_ID": 0,
        "NUM_NODES": 2,
        "PBS_ACCOUNT": "Aurora_Deployment",
        "PBS_ENVIRONMENT": "PBS_INTERACTIVE",
        "PBS_HOOK_RESOURCES": "eJyVUV1vwjAM/EOblHZbgUV54yfs3TKp22bkozgJqP9+LjAJ9jYpD7HvfL5L0Pt0AbQ21VjATmSPMKDzlck0Gq9opBGLOxOspZVriit2Qe5BShoTL2bvsmVaMeTlDpZlpj/AoXIEXjWM0rYyk6x90H1tPza73a7rNq1SejoECKknM3gsepptABdwJGNTmGshKJQLtKp9V03TfMnlremgUUO3lelo55pNq6MDppwqWzJYOTHqKKK2CJbOxKsncTN7FEIWI4VYz5y+hQIzu8SuLMLltK48oD0Oznt5gkz+ppKjvfk8Vex1SQX9YyilL1IVF8io7adScvSpUiV4Dnjv/TPmberZQiaWYDDKdyYY8tXrtTOlQE+NM3rXgwSivORCIZs75eV3+Ady/coN",
        "PBS_JOBCOOKIE": "6B8C4F9D774B0AA5174EAAFB6E2CC14F",
        "PBS_JOBDIR": "/home/foremans",
        "PBS_JOBID": "604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        "PBS_JOBNAME": "STDIN",
        "PBS_MOMPORT": "15003",
        "PBS_NODEFILE": "/var/spool/pbs/aux/604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        "PBS_NODENUM": "0",
        "PBS_O_HOME": "/home/foremans",
        "PBS_O_HOST": "aurora-uan-0010.hostmgmt1000.cm.aurora.alcf.anl.gov",
        "PBS_O_LANG": "en_US.UTF-8",
        "PBS_O_LOGNAME": "foremans",
        "PBS_O_MAIL": "/var/spool/mail/foremans",
        "PBS_O_PATH": "/home/foremans/.nvm/versions/node/v21.5.0/bin:/home/foremans/homebrew/bin:/home/foremans/homebrew/sbin:/opt/cray/pals/1.3.2/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/aurora/23.073.0/support/libraries/intel-compute-samples/2021.27.01:/opt/aurora/23.073.0/support/libraries/khronos/clinfo/default/bin:/opt/aurora/23.073.0/support/tools/gpu_validation:/opt/aurora/23.073.0/intel-gpu-umd/agama-devel-551/compiler/bin:/opt/aurora/23.073.0/intel-gpu-umd/agama-devel-551/driver/bin:/opt/aurora/23.073.0/CNDA/mpich/51.2/mpich-ofi-all-icc-default-pmix-gpu-drop51/bin:/opt/aurora/23.073.0/support/tools/mpi_wrapper_utils:/opt/aurora/23.073.0/oneapi/debugger/2023.0.0/gdb/intel64/bin:/opt/aurora/23.073.0/CNDA/oneapi/compiler/trunk-20230201/dpcpp-ct/bin:/opt/aurora/23.073.0/oneapi/advisor/2023.0.0/bin64:/opt/aurora/23.073.0/CNDA/oneapi/vtune/2023.0.0_624810_nda/bin64:/opt/aurora/23.073.0/CNDA/oneapi/compiler/trunk-20230201/compiler/linux/bin/intel64:/opt/aurora/23.073.0/CNDA/oneapi/compiler/trunk-20230201/compiler/linux/bin:/opt/aurora/23.073.0/oneapi/inspector/2023.0.0/bin64:/opt/cray/pe/gcc/11.2.0/snos/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/home/foremans/.local/bin:/home/foremans/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/home/foremans/.local/share/kitty-ssh-kitten/kitty/bin:/home/foremans/.cargo/bin:/home/foremans/.fzf/bin:/home/foremans/.luarocks/bin:/home/foremans/.luarocks/bin",
        "PBS_O_QUEUE": "EarlyAppAccess",
        "PBS_O_SHELL": "/bin/zsh",
        "PBS_O_SYSTEM": "Linux",
        "PBS_O_TZ": "America/Chicago",
        "PBS_O_WORKDIR": "/home/foremans",
        "PBS_QUEUE": "LustreApps",
        "PBS_TASKNUM": "1",
        "RANK": 0,
        "SCHEDULER": "PBS",
        "WORLD_SIZE_IN_USE": 1,
        "WORLD_SIZE_TOTAL": 24,
        "jobfile_json": "/home/foremans/PBS-jobs/604319/jobenv.json",
        "jobfile_sh": "/home/foremans/PBS-jobs/604319/jobenv.sh",
        "jobfile_yaml": "/home/foremans/PBS-jobs/604319/jobenv.yaml"
    }
    ```
    </details>

    ```bash
    $ source "$(tail -1 ~/PBS-jobs.log)/jobenv.sh"
    $ which launch
    launch: aliased to mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/604319.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    ```


- **Take 1**: Crash with

    ```bash
    RuntimeError: oneCCL: global.cpp:150 getenv_local_coord: EXCEPTION: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    ```

    <details closed><summary>Run:</summary>

    ```bash
    $ launch python3 pretrain_llama.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --num-layers 32 \
        --hidden-size 4096 \
        --ffn-hidden-size 5504 \
        --num-attention-heads 32 \
        --micro-batch-size 1 \
        --global-batch-size 24 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --train-iters 250000 \
        --save /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1 \
        --load /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1 \
        --data-path \
        --data-impl mmap \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ./tmp/tokenizer.model \
        --split 949,50,1 \
        --distributed-backend ccl \
        --lr 3e-4 \
        --lr-decay-style cosine \
        --min-lr 3e-5 \
        --weight-decay 0.1 \
        --clip-grad 1 \
        --lr-warmup-iters 2000 \
        --optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --bf16 \
        --no-query-key-layer-scaling \
        --attention-dropout 0 \
        --hidden-dropout 0 \
        --use-rotary-position-embeddings \
        --untie-embeddings-and-output-weightss \
        --swiglus \
        --normalization rmsnorms \
        --disable-bias-linears \
        --num-key-value-heads 4s \
        --tensorboard-dir /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/outputs/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_ \
        hs4096_gb24_mb1/tensorboard \
        --log-timers-to-tensorboard \
        --tensorboard-log-interval 1 \
        --data-path /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/BookCorpusDataset_text_documents \
        --vocab-file /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-vocab.json \
        --merge-file /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-merges.txt \
        --zero-stage=3 \
        --deepspeed_config=/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/deepspeed.json \
        --deepspeed

    Connected to tcp://x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov:7919
    Found executable /home/foremans/miniconda3/envs/anl_release_q4/bin/python3
    Launching application 2e559157-da5e-4185-9902-dc8d932e8bb3
    /home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 'If you dont plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?'
      warn(
    [2024-01-23 00:02:13,326] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
    [2024-01-23 00:02:19,177] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
    [2024-01-23 00:02:19,177] [INFO] [comm.py:637:init_distributed] cdb=None
    [2024-01-23 00:02:19,177] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=3, local_rank=3, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=9, local_rank=9, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=11, local_rank=11, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=1, local_rank=1, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=2, local_rank=2, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=4, local_rank=4, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=5, local_rank=5, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=6, local_rank=6, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=7, local_rank=7, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=13, local_rank=1, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=8, local_rank=8, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=10, local_rank=10, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=0, local_rank=0, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,889] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend ccl
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=15, local_rank=3, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=17, local_rank=5, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=19, local_rank=7, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=21, local_rank=9, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=12, local_rank=0, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=14, local_rank=2, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=16, local_rank=4, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=18, local_rank=6, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=20, local_rank=8, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=22, local_rank=10, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:19,888] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=23, local_rank=11, world_size=24, master_addr=10.115.53.137, master_port=29500
    [2024-01-23 00:02:20][INFO][dist:257] - DistInfo={
        "DEVICE": "xpu",
        "DEVICE_ID": "xpu:0",
        "DISTRIBUTED_BACKEND": "gloo",
        "GPUS_PER_NODE": 12,
        "HOSTFILE": "/var/spool/pbs/aux/604213.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        "HOSTNAME": "x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov",
        "HOSTS": "['x4502c1s0b0n0', 'x4502c1s3b0n0']",
        "LOCAL_RANK": 0,
        "MACHINE": "Aurora",
        "NGPUS": 24,
        "NODE_ID": 0,
        "NUM_NODES": 2,
        "RANK": 0,
        "SCHEDULER": "PBS",
        "WORLD_SIZE_IN_USE": 24,
        "WORLD_SIZE_TOTAL": 24
    }
    [2024-01-23 00:02:20,987] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
    --------------------------------------------------
    DeepSpeed C++/CUDA extension op report
    --------------------------------------------------
    NOTE: Ops not installed will be just-in-time (JIT) compiled at
          runtime if needed. Op compatibility means that your system
          meet the required dependencies to JIT install the op.
    --------------------------------------------------
    JIT compiled ops requires ninja
    ninja .................. [OKAY]
    --------------------------------------------------
    op name ................ installed .. compatible
    --------------------------------------------------
    [2024-01-23 00:02:20][INFO][spawn:38] - icx -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/foremans/miniconda3/envs/anl_release_q4/include -fPIC -O2 -isystem /home/foremans/miniconda3/envs/anl_release_q4/include -fPIC -c /tmp/tmph01efr3s/test.c -o /tmp/tmph01efr3s/test.o
    WARNING: TensorBoard writing requested but is not available (are you using PyTorch 1.1.0 or later?), no TensorBoard logs will be written.
    2024:01:23-00:02:21:(122507) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(122510) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122515) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122507) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122508) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122509) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122511) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122512) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122513) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122514) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122516) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122517) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(122518) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141071) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141072) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141073) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141075) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141076) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141078) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141079) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141081) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141074) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141077) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141080) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:02:21:(141071) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141075) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141078) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141081) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141072) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141073) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141074) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141076) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141077) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141079) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    2024:01:23-00:02:21:(141080) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
    to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    >fused kernel is only supported in cuda, skip loading fused kernel
    Traceback (most recent call last):
      File "/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/pretrain_llama.py", line 583, in <module>
        model = main()
      File "/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/pretrain_llama.py", line 561, in main
        model = pretrain(
      File "/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/megatron/training.py", line 136, in pretrain
        torch.distributed.all_reduce(start_time_tensor,
      File "/home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
        return func(*args, **kwargs)
      File "/home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 2050, in all_reduce
        work = group.allreduce([tensor], opts)

    RuntimeError: oneCCL: global.cpp:150 getenv_local_coord: EXCEPTION: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly

    </details>
    ```


- **Take 2**: Trying with `CCL_ZE_IPC_EXCHANGE=sockets` (still no luck)

    <details closed><summary>Run:</summary>

    ```bash
    $ CCL_ZE_IPC_EXCHANGE=sockets !!
    [...]
    2024:01:23-00:03:41:(123335) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
    2024:01:23-00:03:41:(123330) |CCL_WARN| sockets exchange mode is set. It may cause potential problem of 'Too many open file descriptors'
    2024:01:23-00:03:41:(123337) |CCL_WARN| sockets exchange mode is set. It may cause potential problem of 'Too many open file descriptors'
    2024:01:23-00:03:41:(123327) |CCL_WARN| sockets exchange mode is set. It may cause potential problem of 'Too many open file descriptors'
    [...]
    RuntimeError: oneCCL: global.cpp:150 getenv_local_coord: EXCEPTION: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
    ```

    </details>


## Using `deepspeed`

- Setup:

    <details closed><summary>Setup:</summary>

    ```bash
    $ cat $PBS_NODEFILE > hostfile ; sed -e 's/$/ slots=12/' -i hostfile
    $ echo "PATH=${PATH}" >> .deepspeed_env ; echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env ; echo "http_proxy=${http_proxy}" >> .deepspeed_env ; echo "https_proxy=${https_proxy}" >> .deepspeed_env
    ```

- Run:

    - Command:

        ```bash
          $ RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost deepspeed --hostfile hostfile pretrain_llama.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --num-layers 32 --hidden-size 4096 --ffn-hidden-size 5504 --num-attention-heads 32 --micro-batch-size 1 --global-batch-size 24 --seq-length 2048 --max-position-embeddings 2048 --train-iters 250000 --save /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1 --load /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1 --data-path --data-impl mmap --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model ./tmp/tokenizer.model --split 949,50,1 --distributed-backend ccl --lr 3e-4 --lr-decay-style cosine --min-lr 3e-5 --weight-decay 0.1 --clip-grad 1 --lr-warmup-iters 2000 --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --log-interval 1 --save-interval 10000 --eval-interval 1000 --eval-iters 10 --bf16 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --num-key-value-heads 4 --tensorboard-dir /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/outputs/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1/tensorboard --log-timers-to-tensorboard --tensorboard-log-interval 1 --data-path /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/BookCorpusDataset_text_document --vocab-file /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-vocab.json --merge-file /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-merges.txt --zero-stage=3 --deepspeed_config=/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/deepspeed.json --deepspeed
        ```


    - Output:

        <details closed><summary>Output</summary>


        ```bash
        $ RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost deepspeed --hostfile hostfile pretrain_llama.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --num-layers 32 --hidden-size 4096 --ffn-hidden-size 5504 --num-attention-heads 32 --micro-batch-size 1 --global-batch-size 24 --seq-length 2048 --max-position-embeddings 2048 --train-iters 250000 --save /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1 --load /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1 --data-path --data-impl mmap --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model ./tmp/tokenizer.model --split 949,50,1 --distributed-backend ccl --lr 3e-4 --lr-decay-style cosine --min-lr 3e-5 --weight-decay 0.1 --clip-grad 1 --lr-warmup-iters 2000 --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --log-interval 1 --save-interval 10000 --eval-interval 1000 --eval-iters 10 --bf16 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --num-key-value-heads 4 --tensorboard-dir /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/outputs/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1/tensorboard --log-timers-to-tensorboard --tensorboard-log-interval 1 --data-path /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/BookCorpusDataset_text_document --vocab-file /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-vocab.json --merge-file /lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-merges.txt --zero-stage=3 --deepspeed_config=/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/deepspeed.json --deepspeed 

        home/foremans/miniconda3/envs/anl_release_q4/bin/deepspeed:4: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
          __import__('pkg_resources').require('deepspeed==0.12.3+6ea44d02')
        /home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 'If you dont plan on using image functionality from `torchvision.io`, you can igno re this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?'
          warn(
        My guessed rank = 0
        [2024-01-23 00:09:56,016] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
        [2024-01-23 00:10:00,790] [INFO] [runner.py:463:main] Using IP address of 10.115.53.137 for node x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov
        [2024-01-23 00:10:00,812] [INFO] [runner.py:559:main] deepspeed_env file = .deepspeed_env
        [2024-01-23 00:10:00,813] [INFO] [runner.py:559:main] deepspeed_env file = .deepspeed_env
        [2024-01-23 00:10:00,813] [INFO] [multinode_runner.py:72:get_cmd] Running on the following workers: x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov,x4502c1s3b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov
        [2024-01-23 00:10:00,813] [INFO] [runner.py:570:main] cmd = pdsh -S -f 1024 -w x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov,x4502c1s3b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov export PYTHONSTARTUP=/etc/pythonstart; export PYTHONPATH=/l
        us/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed:/soft/compilers/oneapi/2023.12.15.001/oneapi/advisor/2024.0/pythonapi; export PATH=/home/ftartagl/graphics-compute-runtime/agama-ci-devel-736.9/u
        sr/bin:/soft/tools/gpu_validation:/soft/libraries/khronos/clinfo/master-13ae34-2020.12.14/bin:/soft/libraries/intel-compute-samples/2021.27.01:/soft/libraries/intel-gpu-umd/stable_736_25_20231031/compiler/bin:/soft/libraries/intel-gpu-umd
        /stable_736_25_20231031/driver/bin:/soft/restricted/CNDA/updates/mpich/52.2/mpich-ofi-all-icc-default-pmix-gpu-drop52/bin:/soft/tools/mpi_wrapper_utils:/soft/compilers/oneapi/2023.12.15.001/oneapi/dpcpp-ct/2024.0/bin:/soft/compilers/oneap
        i/2023.12.15.001/oneapi/advisor/2024.0/bin64:/soft/compilers/oneapi/2023.12.15.001/oneapi/vtune/2024.0/bin64:/soft/compilers/oneapi/2023.12.15.001/oneapi/inspector/2024.0/bin64:/soft/compilers/oneapi/2023.12.15.001/oneapi/debugger/2024.0/
        opt/debugger/bin:/soft/compilers/oneapi/2023.12.15.001/oneapi/mkl/2024.0/bin:/soft/compilers/oneapi/2023.12.15.001/oneapi/compiler/2024.0/bin:/home/foremans/miniconda3/envs/anl_release_q4/bin:/home/foremans/miniconda3/condabin:/home/forem
        ans/.nvm/versions/node/v21.5.0/bin:/home/foremans/homebrew/bin:/home/foremans/homebrew/sbin:/opt/cray/pals/1.3.2/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/cray/pe/gcc/11.2.0/snos/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/b
        in:/home/foremans/.local/bin:/home/foremans/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/home/foremans/.local/share/kitty-ssh-kitten/kitty/bin:/home/foremans/.cargo/bin:/home/foremans
        /.fzf/bin:/home/foremans/.luarocks/bin; export LD_LIBRARY_PATH=/home/ftartagl/graphics-compute-runtime/agama-ci-devel-736.9/usr/lib64/dri:/home/ftartagl/graphics-compute-runtime/agama-ci-devel-736.9/usr/lib64/mfx:/home/ftartagl/graphics-c
        ompute-runtime/agama-ci-devel-736.9/usr/lib64/intel-opencl:/home/ftartagl/graphics-compute-runtime/agama-ci-devel-736.9/usr/lib64:/soft/libraries/khronos/loader/master-2022.05.18/lib64:/soft/libraries/intel-gpu-umd/stable_736_25_20231031/
        compiler/lib64:/soft/libraries/intel-gpu-umd/stable_736_25_20231031/driver/lib64/intel-opencl:/soft/libraries/intel-gpu-umd/stable_736_25_20231031/driver/lib64:/soft/restricted/CNDA/updates/mpich/52.2/mpich-ofi-all-icc-default-pmix-gpu-dr
        op52/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/ipp/2021.10/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/ippcp/2021.9/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/dpl/2022.3/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/d
        ebugger/2024.0/opt/debugger/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/ccl/2021.11/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/dal/2024.0/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/dnnl/2024.0/lib:/soft/compilers/oneapi/2
        023.12.15.001/oneapi/tbb/2021.11/lib/intel64/gcc4.8:/soft/compilers/oneapi/2023.12.15.001/oneapi/mkl/2024.0/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/compiler/2024.0/opt/compiler/lib:/soft/compilers/oneapi/2023.12.15.001/oneapi/com
        piler/2024.0/lib:/opt/cray/libfabric/1.15.2.0/lib64:/opt/cray/pe/gcc/11.2.0/snos/lib64; export http_proxy=http://proxy-01.pub.alcf.anl.gov:3128; export https_proxy=http://proxy.alcf.anl.gov:3128;  cd /lus/gecko/projects/Aurora_deployment/
        foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed; /home/foremans/miniconda3/envs/anl_release_q4/bin/python3 -u -m deepspeed.launcher.launch --world_info=eyJ4NDUwMmMxczBiMG4wLmhvc3RtZ210MjUwMi5jbS5hdXJvcmEuYWxjZi5hbmwuZ292IjogWzAsI
        DEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxMV0sICJ4NDUwMmMxczNiMG4wLmhvc3RtZ210MjUwMi5jbS5hdXJvcmEuYWxjZi5hbmwuZ292IjogWzAsIDEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxMV19 --node_rank=%n --master_addr=10.115.53.137 --master_port=29500 pre
        train_llama.py --tensor-model-parallel-size '1' --pipeline-model-parallel-size '1' --num-layers '32' --hidden-size '4096' --ffn-hidden-size '5504' --num-attention-heads '32' --micro-batch-size '1' --global-batch-size '24' --seq-length '20
        48' --max-position-embeddings '2048' --train-iters '250000' --save '/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1'
        --load '/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/checkpoints/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1' --data-path --data-impl 'mmap' --tokenizer-type 'GPTSentence
        PieceTokenizer' --tokenizer-model './tmp/tokenizer.model' --split '949,50,1' --distributed-backend 'ccl' --lr '3e-4' --lr-decay-style 'cosine' --min-lr '3e-5' --weight-decay '0.1' --clip-grad '1' --lr-warmup-iters '2000' --optimizer 'adam
        ' --adam-beta1 '0.9' --adam-beta2 '0.95' --log-interval '1' --save-interval '10000' --eval-interval '1000' --eval-iters '10' --bf16 --no-query-key-layer-scaling --attention-dropout '0' --hidden-dropout '0' --use-rotary-position-embeddings
         --untie-embeddings-and-output-weights --swiglu --normalization 'rmsnorm' --disable-bias-linear --num-key-value-heads '4' --tensorboard-dir '/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/ou
        tputs/LLAMA_7B_LLAMA_7B_z3_seqlen_mp1_pp1_sp24_nl32_hs4096_gb24_mb1/tensorboard' --log-timers-to-tensorboard --tensorboard-log-interval '1' --data-path '/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-
        DeepSpeed/dataset/BookCorpusDataset_text_document' --vocab-file '/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-vocab.json' --merge-file '/lus/gecko/projects/Aurora_deployment/f
        oremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/dataset/gpt2-merges.txt' --zero-stage=3 --deepspeed_config=/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/deepspeed.json --deepspeed
        x4502c1s3b0n0: Warning: Permanently added 'x4502c1s3b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov,10.115.53.138' (ECDSA) to the list of known hosts.

        x4502c1s0b0n0: /home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: ''If you don't plan on using image functionality from `torchvision.io
        `, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
        x4502c1s0b0n0:   warn(
        x4502c1s3b0n0: /home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: ''If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
        x4502c1s3b0n0:   warn(
        x4502c1s0b0n0: [2024-01-23 06:10:07,853] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
        x4502c1s0b0n0: [2024-01-23 06:10:08,419] [INFO] [launch.py:145:main] WORLD INFO DICT: {'x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'x4502c1s3b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
        x4502c1s0b0n0: [2024-01-23 06:10:08,419] [INFO] [launch.py:151:main] nnodes=2, num_local_procs=12, node_rank=0
        x4502c1s0b0n0: [2024-01-23 06:10:08,419] [INFO] [launch.py:162:main] global_rank_mapping=defaultdict(<class 'list'>, {'x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'x4502c1s3b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]})
        x4502c1s0b0n0: [2024-01-23 06:10:08,419] [INFO] [launch.py:163:main] dist_world_size=24
        x4502c1s0b0n0: [2024-01-23 06:10:08,419] [INFO] [launch.py:165:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
        x4502c1s3b0n0: [2024-01-23 06:10:08,885] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
        x4502c1s3b0n0: [2024-01-23 06:10:09,452] [INFO] [launch.py:145:main] WORLD INFO DICT: {'x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'x4502c1s3b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
        x4502c1s3b0n0: [2024-01-23 06:10:09,452] [INFO] [launch.py:151:main] nnodes=2, num_local_procs=12, node_rank=1
        x4502c1s3b0n0: [2024-01-23 06:10:09,452] [INFO] [launch.py:162:main] global_rank_mapping=defaultdict(<class 'list'>, {'x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'x4502c1s3b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]})
        x4502c1s3b0n0: [2024-01-23 06:10:09,452] [INFO] [launch.py:163:main] dist_world_size=24
        x4502c1s3b0n0: [2024-01-23 06:10:09,452] [INFO] [launch.py:165:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
        x4502c1s0b0n0: My guessed rank = 4
        x4502c1s0b0n0: My guessed rank = 9
        x4502c1s0b0n0: My guessed rank = 8
        x4502c1s0b0n0: My guessed rank = 0
        x4502c1s0b0n0: My guessed rank = 6
        x4502c1s0b0n0: My guessed rank = 7
        x4502c1s0b0n0: My guessed rank = 11
        x4502c1s0b0n0: My guessed rank = 5
        x4502c1s0b0n0: My guessed rank = 10
        x4502c1s0b0n0: My guessed rank = 3
        x4502c1s0b0n0: My guessed rank = 2
        x4502c1s0b0n0: My guessed rank = 1
        x4502c1s3b0n0: My guessed rank = 21
        x4502c1s3b0n0: My guessed rank = 18
        x4502c1s3b0n0: My guessed rank = 22
        x4502c1s3b0n0: My guessed rank = 20
        x4502c1s3b0n0: My guessed rank = 14
        x4502c1s3b0n0: My guessed rank = 12
        x4502c1s3b0n0: My guessed rank = 23
        x4502c1s3b0n0: My guessed rank = 16
        x4502c1s3b0n0: My guessed rank = 17
        x4502c1s3b0n0: My guessed rank = 19
        x4502c1s3b0n0: My guessed rank = 15
        x4502c1s3b0n0: My guessed rank = 13
        x4502c1s0b0n0: [2024-01-23 06:10:14,751] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
        x4502c1s0b0n0: [2024-01-23 06:10:19,225] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
        x4502c1s0b0n0: [2024-01-23 06:10:19,225] [INFO] [comm.py:637:init_distributed] cdb=None
        x4502c1s0b0n0: [2024-01-23 06:10:20,891] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend ccl
        x4502c1s0b0n0: [2024-01-23 06:10:21][INFO][dist:257] - DistInfo={
        x4502c1s0b0n0:     "DEVICE": "xpu",
        x4502c1s0b0n0:     "DEVICE_ID": "xpu:0",
        x4502c1s0b0n0:     "DISTRIBUTED_BACKEND": "gloo",
        x4502c1s0b0n0:     "GPUS_PER_NODE": 12,
        x4502c1s0b0n0:     "HOSTFILE": "/var/spool/pbs/aux/604213.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        x4502c1s0b0n0:     "HOSTNAME": "x4502c1s0b0n0.hostmgmt2502.cm.aurora.alcf.anl.gov",
        x4502c1s0b0n0:     "HOSTS": "['x4502c1s0b0n0', 'x4502c1s3b0n0']",
        x4502c1s0b0n0:     "LOCAL_RANK": 0,
        x4502c1s0b0n0:     "MACHINE": "Aurora",
        x4502c1s0b0n0:     "NGPUS": 24,
        x4502c1s0b0n0:     "NODE_ID": 0,
        x4502c1s0b0n0:     "NUM_NODES": 2,
        x4502c1s0b0n0:     "RANK": 0,
        x4502c1s0b0n0:     "SCHEDULER": "PBS",
        x4502c1s0b0n0:     "WORLD_SIZE_IN_USE": 1,
        x4502c1s0b0n0:     "WORLD_SIZE_TOTAL": 24
        x4502c1s0b0n0: }
        x4502c1s0b0n0: [2024-01-23 06:10:21,533] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to xpu (auto detect)
        x4502c1s0b0n0: --------------------------------------------------
        x4502c1s0b0n0: DeepSpeed C++/CUDA extension op report
        x4502c1s0b0n0: --------------------------------------------------
        x4502c1s0b0n0: NOTE: Ops not installed will be just-in-time (JIT) compiled at
        x4502c1s0b0n0:       runtime if needed. Op compatibility means that your system
        x4502c1s0b0n0:       meet the required dependencies to JIT install the op.
        x4502c1s0b0n0: --------------------------------------------------
        x4502c1s0b0n0: JIT compiled ops requires ninja
        x4502c1s0b0n0: ninja .................. [OKAY]
        x4502c1s0b0n0: --------------------------------------------------
        x4502c1s0b0n0: op name ................ installed .. compatible
        x4502c1s0b0n0: --------------------------------------------------
        x4502c1s0b0n0: [2024-01-23 06:10:21][INFO][spawn:38] - gcc -pthread -B /home/foremans/miniconda3/envs/anl_release_q4/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/foremans/miniconda3/envs/anl_release_q4/include -fPIC -O2 -isystem /home/foremans/miniconda3/envs/anl_release_q4/include -fPIC -c /tmp/tmptqyph55g/test.c -o /tmp/tmptqyph55g/test.o
        x4502c1s3b0n0: [2024-01-23 06:10:21,671] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
        x4502c1s3b0n0: [2024-01-23 06:10:21,672] [INFO] [comm.py:637:init_distributed] cdb=None

        [...]
        x4502c1s0b0n0: >fused kernel is only supported in cuda, skip loading fused kernel
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153241) |CCL_WARN| did not find MPI-launcher specific variables, switch to ATL/OFI, to force enable ATL/MPI set CCL_ATL_TRANSPORT=mpi
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153241) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153241) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
        x4502c1s0b0n0: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153242) |CCL_WARN| did not find MPI-launcher specific variables, switch to ATL/OFI, to force enable ATL/MPI set CCL_ATL_TRANSPORT=mpi
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153242) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153242) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
        x4502c1s0b0n0: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
        x4502c1s3b0n0:  > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153237) |CCL_WARN| did not find MPI-launcher specific variables, switch to ATL/OFI, to force enable ATL/MPI set CCL_ATL_TRANSPORT=mpi
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153237) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
        x4502c1s0b0n0: 2024:01:23-06:12:16:(153237) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
        x4502c1s0b0n0: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
        x4502c1s0b0n0:  > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
        x4502c1s0b0n0:  > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
        x4502c1s3b0n0: 2024:01:23-06:12:16:(129554) |CCL_WARN| could not get local_idx/count from environment variables, trying to get them from ATL
        x4502c1s3b0n0: 2024:01:23-06:12:16:(129554) |CCL_ERROR| global.cpp:150 getenv_local_coord: condition global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets failed
        x4502c1s3b0n0: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
        x4502c1s3b0n0: RuntimeError: oneCCL: global.cpp:150 getenv_local_coord: EXCEPTION: to get local_idx/count from ATL, set CCL_ZE_IPC_EXCHANGE=sockets explicitly
        x4502c1s3b0n0: Traceback (most recent call last):
        x4502c1s3b0n0:   File "/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/pretrain_llama.py", line 583, in <module>
        x4502c1s3b0n0:     model = main()
        x4502c1s3b0n0:   File "/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/pretrain_llama.py", line 561, in main
        x4502c1s3b0n0:     model = pretrain(
        x4502c1s3b0n0:   File "/lus/gecko/projects/Aurora_deployment/foremans/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed/megatron/training.py", line 136, in pretrain
        x4502c1s3b0n0:     torch.distributed.all_reduce(start_time_tensor,
        x4502c1s3b0n0:   File "/home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
        x4502c1s3b0n0:     return func(*args, **kwargs)
        x4502c1s3b0n0:   File "/home/foremans/miniconda3/envs/anl_release_q4/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 2050, in all_reduce
        x4502c1s3b0n0:     work = group.allreduce([tensor], opts)
        ```

        </details>
