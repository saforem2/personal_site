# 🍋 <code>ezpz</code> @ ALCF
Sam Foreman
2024-08-23

- [🐣 Getting Started](#hatching_chick-getting-started)
- [🐚 Shell Utilities](#shell-shell-utilities)
  - [Setup Environment](#setup-environment)
  - [Setup Environment](#setup-environment-1)
- [🐍 Python Library](#snake-python-library)

> *Work smarter, not harder*.

## 🐣 Getting Started

There are two main, distinct components of `ezpz`:

1.  🐍 [**Python Library**](#python-libraryy)

2.  🐚 [**Shell Utilities**](#shell-utilities):

    1.  [Setup Environment](#setup-environment):
        1.  [Setup Python](#setup-python):
            1.  [Setup Conda](#setup-conda)
            2.  [Setup Virtual Environment](#setup-virtual-environment)
        2.  [Setup Job](#setup-job):

## 🐚 Shell Utilities

we provide a variety of helper functions designed to make your life
easier when working with job schedulers (e.g. `PBS Pro` @ ALCF or
`slurm` elsewhere).

**All** of these functions are:

- located in [`utils.sh`](../../src/ezpz/bin/utils.sh)
- prefixed with `ezpz_*` (e.g. `ezpz_setup_python`)[^1]

To use these, we can `source` the file directly via:

``` bash
export PBS_O_WORKDIR=$(pwd) # if on ALCF
source /dev/stdin <<< $(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
```

### Setup Environment

We would like to write our application in such a way that it is able to
take full advantage of the resources allocated by the job scheduler.

That is to say, we want to have a single script with the ability to
dynamically `launch` python applications across any number of
accelerators on any of the systems under consideration.

In order to do this, there is some basic setup and information gathering
that needs to occur.

In particular, we need mechanisms for:

1.  Setting up a python environment
2.  Determining what system / machine we’re on
    - \+ what job scheduler we’re using (e.g. `PBS Pro` @ ALCF or
      `slurm` elsewhere)
3.  Determining how many nodes have been allocated in the current job
    (`NHOSTS`)
    - \+ Determining how many accelerators exist on each of these nodes
      (`NGPU_PER_HOST`)

This allows us to calculate the total number of accelerators (GPUs) as:

$$\mathrm{NGPUS} = \mathrm{NHOSTS} \times \mathrm{NGPU\_PER\_HOST}$$

With this we have everything we need to build the appropriate
{`mpi`{`run`, `exec`}, `slurm`} command for launching our python
application across them.

Now, there are a few functions in particular worth elaborating on.

<div id="tbl-shell-fns">

|   | Function |
|:---|:---|
| [Setup Environment](#setup-environment) | `ezpz_setup_env` |
| [Setup Job](#setup-job) | `ezpz_setup_job` |
| [Setup Python](#setup-python) | `ezpz_setup_python` |
| [Setup Conda](#setup-conda) | `ezpz_setup_conda` |
| [Setup Virtual Environment](#setup-virtual-environment) | `ezpz_setup_venv_from_conda` |

Table 1: Shell Functions

</div>

> [!WARNING]
>
> ### Where am I?
>
> *Some* of the `ezpz_*` functions (e.g. `ezpz_setup_python`), will try
> to create / look for certain directories.
>
> In an effort to be explicit, these directories will be defined
> **relative to** a `WORKING_DIR` (e.g. `"${WORKING_DIR}/venvs/"`)
>
> This `WORKING_DIR` will be assigned to the first non-zero match found
> below:
>
> 1.  `PBS_O_WORKDIR`: If found in environment, paths will be relative
>     to this
> 2.  `SLURM_SUBMIT_DIR`: Next in line. If not @ ALCF, maybe using
>     `slurm`…
> 3.  `$(pwd)`: Otherwise, no worries. Use your *actual* working
>     directory.

#### Setup Python

``` bash
ezpz_setup_python
```

This will:

1.  Automatically load and activate `conda` using the `ezpz_setup_conda`
    function.

    How this is done, in practice, varies from machine to machine:

    - **ALCF**[^2]: Automatically load the most recent `conda` module
      and activate the base environment.

    - **Frontier**: Load the appropriate AMD modules (e.g. `rocm`,
      `RCCL`, etc.), and activate base `conda`

    - **Perlmutter**: Load the appropriate `pytorch` module and activate
      environment

    - **Unknown**: In this case, we will look for a `conda`, `mamba`, or
      `micromamba` executable, and if found, use that to activate the
      base environment.

2.  Build (or activate, if found) a virtual environment on top of (the
    active) base `conda` environment.

    By default, it will try looking in:

    - `$PBS_O_WORKDIR`, otherwise
    - ‘\${SLURM_SUBMIT_DIR}’, otherwise
    - `$(pwd)`

    for a nested folder named `"venvs/${CONDA_NAME}"`.

    If this doesn’t exist, it will attempt to create a new virtual
    environment at this location using:

    ``` bash
    python3 -m venv venvs/${CONDA_NAME} --system-site-packages
    ```

    (where we’ve pulled in the `--system-site-packages` from conda).

#### Setup Job

Now that we are in a suitable python environment, we need to construct
the command that we will use to run python on each of our acceleratorss.

To do this, we need a few things:

1.  What machine we’re on (and what scheduler is it using i.e. {PBS,
    SLURM})
2.  How many nodes are available in our active job
3.  How many GPUs are on each of those nodes
4.  What type of GPUs are they

With this information, we can then use `mpi{exec,run}` or `srun` to
launch python across all of our accelerators.

Again, how this is done will vary from machine to machine.

To identify where we are, we look at our `$(hostname)` and see if:

- `x*`: ALCF

  - Aurora: `x4*` (or `aurora*`)
  - Sunspot: `x1*` (or `uan*`)
  - Polaris / Sirius: `x3*`
    - to determine, we look at `"${PBS_O_HOST}"`
  - `sophia*`: Sophia

  Once we have this, we can:

  1.  Get `PBS_NODEFILE` from `$(hostname)`:

      - `ezpz_qsme_running`: For each (running) job owned by `${USER}`,
        print out both the jobid as well as a list of hosts the job is
        running on, e.g.:

        ``` bash
        <jobid0> host00 host01 host02 host03 ...
        <jobid1> host10 host11 host12 host13 ...
        ...
        ```

      - `ezpz_get_pbs_nodefile_from_hostname`: Look for `$(hostname)` in
        the output from the above command to determine our
        `${PBS_JOBID}`.

        Once we’ve identified our `${PBS_JOBID}` we then know the
        location of our `${PBS_NODEFILE}` since they are named according
        to:

        ``` bash
        jobid=$(ezpz_qsme_running | grep "$(hostname)" | awk '{print $1}')
        prefix=/var/spool/pbs/aux
        match=$(/bin/ls "${prefix}" | grep "${jobid}")
        hostfile="${prefix}/${match}""
        ```

  2.  Identify number of available accelerators:

### Setup Environment

------------------------------------------------------------------------

## 🐍 Python Library

To install[^3]:

``` bash
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
```

- 📂 [`bin/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/):
  - [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):
    Common utilities for EZPZ
- 📂
  [`conf/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/):
  - ⚙️
    [`conf/config.yaml`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/config.yaml):
    Default `TrainConfig` object
  - ⚙️
    [`conf/ds_config.json`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/ds_config.json):
    DeepSpeed configuration
- 📂 [`log/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/log/):
  Logging configuration.
- 🐍
  [`__about__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__about__.py):
  Version information
- 🐍
  [`__init__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__init__.py):
  Main module
- 🐍
  [`__main__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__main__.py):
  Entry point
- 🐍
  [`configs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/configs.py):
  Configuration module
- 🐍[`cria.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/create.py):
  Baby Llama
- 🐍[**`dist.py`**](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py):
  Distributed training module
- 🐍[*`history.py`*](https://github.com/saforem2/ezpz/blob/main/src/ezpz/history.py):
  History module
- 🐍[`jobs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/jobs.py):
  Jobs module
- 🐍[`model.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/model.py):
  Model module
- 🐍[`plot.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/plot.py):
  Plot modul
- 🐍[`profile.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/profile.py):
  Profile module
- 🐍[`runtime.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/runtime.py):
  Runtime module
- 🐍[`test.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test.py):
  Test module
- 🐍[**`test_dist.py`**](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py):
  Distributed training test module
- 🐍[`train.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/train.py):
  train module
- 🐍[`trainer.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/trainer.py):
  trainer module
- 🐍[`utils.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/utils.py):
  utility module

``` bash
📂 /ezpz/src/ezpz/
┣━━ 📂 bin/
┃   ┣━━ 📄 affinity.sh
┃   ┣━━ 📄 getjobenv
┃   ┣━━ 📄 savejobenv
┃   ┣━━ 📄 saveslurmenv
┃   ┣━━ 📄 setup.sh
┃   ┣━━ 📄 train.sh
┃   ┗━━ 📄 utils.sh
┣━━ 📂 conf/
┃   ┣━━ 📂 hydra/
┃   ┃   ┗━━ 📂 job_logging/
┃   ┃       ┣━━ ⚙️ colorlog1.yaml
┃   ┃       ┣━━ ⚙️ custom.yaml
┃   ┃       ┗━━ ⚙️ enrich.yaml
┃   ┣━━ 📂 logdir/
┃   ┃   ┗━━ ⚙️ default.yaml
┃   ┣━━ ⚙️ config.yaml
┃   ┣━━ 📄 ds_config.json
┃   ┗━━ ⚙️ ds_config.yaml
┣━━ 📂 log/
┃   ┣━━ 📂 conf/
┃   ┃   ┗━━ 📂 hydra/
┃   ┃       ┗━━ 📂 job_logging/
┃   ┃           ┗━━ ⚙️ enrich.yaml
┃   ┣━━ 🐍 __init__.py
┃   ┣━━ 🐍 __main__.py
┃   ┣━━ 🐍 config.py
┃   ┣━━ 🐍 console.py
┃   ┣━━ 🐍 handler.py
┃   ┣━━ 🐍 style.py
┃   ┣━━ 🐍 test.py
┃   ┗━━ 🐍 test_log.py
┣━━ 🐍 __about__.py
┣━━ 🐍 __init__.py
┣━━ 🐍 __main__.py
┣━━ 🐍 configs.py
┣━━ 🐍 cria.py
┣━━ 🐍 dist.py
┣━━ 🐍 history.py
┣━━ 🐍 jobs.py
┣━━ 🐍 loadjobenv.py
┣━━ 🐍 model.py
┣━━ 🐍 plot.py
┣━━ 🐍 profile.py
┣━━ 🐍 runtime.py
┣━━ 🐍 savejobenv.py
┣━━ 🐍 test.py
┣━━ 🐍 test_dist.py
┣━━ 🐍 train.py
┣━━ 🐍 trainer.py
┗━━ 🐍 utils.py
```

[^1]: Plus this is useful for tab-completions in your shell, e.g.:

    ``` bash
    $ ezpz_<TAB>
    ezpz_check_and_kill_if_running
    ezpz_get_dist_launch_cmd
    ezpz_get_job_env
    --More--
    ```

[^2]: Any of {Aurora, Polaris, Sophia, Sunspot, Sirius}

[^3]: Note the `--require-virtualenv` isn’t *strictly* required, but I
    highly recommend to always try and work within a virtual
    environment, when possible.
