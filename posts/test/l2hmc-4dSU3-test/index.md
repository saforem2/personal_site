---
title: '`l2hmc-qcd`'
jupyter: python3
---


This notebook contains a minimal working example for the 4D $SU(3)$ model

Uses `torch.complex128` by default


## Setup


```python
import lovely_tensors as lt
lt.monkey_patch()
lt.set_config(color=False)
```


```python
%load_ext autoreload
%autoreload 2
# automatically detect and reload local changes to modules
%matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
```


```python
import os
import ezpz
import ambivalent
from pathlib import Path
from typing import Optional

import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import l2hmc.group.su3.pytorch.group as g
#from l2hmc.utils.rich import get_console
from l2hmc.common import grab_tensor, print_dict
from l2hmc.configs import dict_to_list_of_overrides, get_experiment
from l2hmc.experiment.pytorch.experiment import Experiment, evaluate  # noqa  # noqa
#from l2hmc.utils.plot_helpers import set_plot_style

logger = ezpz.get_logger('[l2hmc][4dSU3]')
os.environ['COLORTERM'] = 'truecolor'
#os.environ['MASTER_PORT'] = '5433'
# os.environ['MPLBACKEND'] = 'module://matplotlib-backend-kitty'
# plt.switch_backend('module://matplotlib-backend-kitty')
#console = get_console()

import matplotlib.pyplot as plt
plt.style.use(ambivalent.STYLES['ambivalent'])

rank = ezpz.setup_torch()
torch.set_default_dtype(torch.float64)
#from l2hmc.utils.dist import setup_torch
#_ = setup_torch(port=9421, precision='float64', backend='DDP', seed=4351)

# set_plot_style()

from l2hmc.utils.plot_helpers import (  # noqa
    set_plot_style,
    plot_scalar,
    plot_chains,
    plot_leapfrogs
)

def savefig(fig: plt.Figure, fname: str, outdir: os.PathLike):
    pngfile = Path(outdir).joinpath(f"pngs/{fname}.png")
    svgfile = Path(outdir).joinpath(f"svgs/{fname}.svg")
    pngfile.parent.mkdir(exist_ok=True, parents=True)
    svgfile.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(svgfile, transparent=True, bbox_inches='tight')
    fig.savefig(pngfile, transparent=True, bbox_inches='tight', dpi=300)

def plot_metrics(metrics: dict, title: Optional[str] = None, **kwargs):
    outdir = Path(f"./plots-4dSU3/{title}")
    outdir.mkdir(exist_ok=True, parents=True)
    for key, val in metrics.items():
        fig, ax = plot_metric(val, name=key, **kwargs)
        if title is not None:
            ax.set_title(title)
        logger.info(f"Saving {key} to {outdir}")
        savefig(fig, f"{key}", outdir=outdir)
        plt.show()

def plot_metric(
        metric: torch.Tensor,
        name: Optional[str] = None,
        **kwargs,
):
    assert len(metric) > 0
    if isinstance(metric[0], (int, float, bool, np.floating)):
        y = np.stack(metric)
        return plot_scalar(y, ylabel=name, **kwargs)
    element_shape = metric[0].shape
    if isinstance(metric[0], np.ndarray):
        y = np.array(metric)
    else:
        try:
            y = grab_tensor(torch.tensor(metric))
        except Exception:
            y = grab_tensor(torch.stack(metric))
    if len(element_shape) == 2:
        return plot_leapfrogs(y, ylabel=name)
    if len(element_shape) == 1:
        # y = grab_tensor(torch.stack(metric))
        return plot_chains(y, ylabel=name, **kwargs)
    if len(element_shape) == 0:
        # y = grab_tensor(torch.stack(metric))
        return plot_scalar(y, ylabel=name, **kwargs)
    raise ValueError
```

    [2025-04-04 08:48:53,301] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to mps (auto detect)


    [Sams-MacBook-Pro-2.local:53580] shmem: mmap: an error occurred while determining whether or not /var/folders/53/5t2nv83136j76rld14vgfh2h0000gq/T//ompi.Sams-MacBook-Pro-2.503/jf.0/475529216/sm_segment.Sams-MacBook-Pro-2.503.1c580000.0 could be created.
    W0404 08:48:54.282000 53580 torch/distributed/elastic/multiprocessing/redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Using device: cpu
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:01</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">dist</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">557</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Using <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">get_torch_device_type</span><span style="color: #ffffff; text-decoration-color: #ffffff">()</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">'mps'</span> with <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">backend</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">'gloo'</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:01</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">dist</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">873</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Using <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">device</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">'mps'</span> with <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">backend</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">'ddp'</span> + <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">'gloo'</span> for distributed training.
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:01</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">dist</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">923</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">'Sams-MacBook-Pro-2.local'</span><span style="color: #ffffff; text-decoration-color: #ffffff">][</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>/<span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> 
</pre>



## Load config + build Experiment


```python
import json
#set_plot_style()

from l2hmc.configs import CONF_DIR

su3conf = Path(f"{CONF_DIR}/su3-min.yaml")
with su3conf.open('r') as stream:
    conf = dict(yaml.safe_load(stream))
logger.info(json.dumps(conf, indent=4))
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">4208194900</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">9</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"annealing_schedule"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"beta_final"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"beta_init"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0</span>
    <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"backend"</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"DDP"</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"conv"</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"none"</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"dynamics"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"eps"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"eps_fixed"</span>: false,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"group"</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"SU3"</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"latvolume"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">[</span>
            <span style="color: #ff00ff; text-decoration-color: #ff00ff">4</span>,
            <span style="color: #ff00ff; text-decoration-color: #ff00ff">4</span>,
            <span style="color: #ff00ff; text-decoration-color: #ff00ff">4</span>,
            <span style="color: #ff00ff; text-decoration-color: #ff00ff">4</span>
        <span style="color: #ffffff; text-decoration-color: #ffffff">]</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"nchains"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"nleapfrog"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"use_separate_networks"</span>: false,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"use_split_xnets"</span>: false,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"verbose"</span>: true
    <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"framework"</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"pytorch"</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"init_aim"</span>: false,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"init_wandb"</span>: false,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"learning_rate"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"lr_init"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.001</span>
    <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"loss"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"charge_weight"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"plaq_weight"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"rmse_weight"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"use_mixed_loss"</span>: false
    <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"net_weights"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"v"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
            <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"q"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0</span>,
            <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"s"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0</span>,
            <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"t"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0</span>
        <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"x"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
            <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"q"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0</span>,
            <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"s"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0</span>,
            <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"t"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0</span>
        <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>
    <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"network"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"activation_fn"</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"tanh"</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"dropout_prob"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"units"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">[</span>
            <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>
        <span style="color: #ffffff; text-decoration-color: #ffffff">]</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"use_batch_norm"</span>: false
    <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"restore"</span>: false,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"save"</span>: false,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"steps"</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">{</span>
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"log"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"nepoch"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">10</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"nera"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"print"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>,
        <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"test"</span>: <span style="color: #ff00ff; text-decoration-color: #ff00ff">50</span>
    <span style="color: #ffffff; text-decoration-color: #ffffff">}</span>,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"use_tb"</span>: false,
    <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">"use_wandb"</span>: false
<span style="color: #ffffff; text-decoration-color: #ffffff">}</span>
</pre>




```python
overrides = dict_to_list_of_overrides(conf)
ptExpSU3 = get_experiment(overrides=[*overrides], build_networks=True)
state = ptExpSU3.trainer.dynamics.random_state(6.0)
logger.info(f"checkSU(state.x): {g.checkSU(state.x)}")
assert isinstance(state.x, torch.Tensor)
assert isinstance(state.beta, torch.Tensor)
assert isinstance(ptExpSU3, Experiment)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">utils</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">dist</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">229</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.utils.dist</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Caught MASTER_PORT:<span style="color: #ff00ff; text-decoration-color: #ff00ff">1234</span> from environment!
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">utils</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">dist</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">229</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.utils.dist</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Caught MASTER_PORT:<span style="color: #ff00ff; text-decoration-color: #ff00ff">1234</span> from environment!
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #ffff00; text-decoration-color: #ffff00">W</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">trainer</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">467</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.trainers.pytorch.trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Using torch.float32 on cpu!
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #ffff00; text-decoration-color: #ffff00">W</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">trainer</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">467</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.trainers.pytorch.trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Using `torch.optim.Adam` optimizer
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">trainer</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">305</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.trainers.pytorch.trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>num_params in model: <span style="color: #ff00ff; text-decoration-color: #ff00ff">200710</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #ffff00; text-decoration-color: #ffff00">W</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">trainer</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">271</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.trainers.pytorch.trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>logging with freq <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span> for wandb.watch
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">982607162</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">4</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">checkSU</span><span style="color: #ffffff; text-decoration-color: #ffffff">(</span>state.x<span style="color: #ffffff; text-decoration-color: #ffffff">)</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">(</span>tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">1.420e-07</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.194e-07</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.695e-07</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.290e-07</span>, tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">1.341e-06</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.532e-05</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">5.465e-06</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.482e-06</span><span style="color: #ffffff; text-decoration-color: #ffffff">)</span>
</pre>




```python
state.x.real.plt()
```




    
![svg](output_7_0.svg)
    




```python
state.x.imag.plt()
```




    
![svg](output_8_0.svg)
    



## HMC


```python
xhmc, history_hmc = evaluate(
    nsteps=50,
    exp=ptExpSU3,
    beta=state.beta,
    x=state.x,
    eps=0.1,
    nleapfrog=1,
    job_type='hmc',
    nlog=1,
    nprint=25,
    grab=True
)
xhmc = ptExpSU3.trainer.dynamics.unflatten(xhmc)
logger.info(f"checkSU(x_hmc): {g.checkSU(xhmc)}")
plot_metrics(history_hmc.history, title='HMC', marker='.')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">117</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Running <span style="color: #ff00ff; text-decoration-color: #ff00ff">50</span> steps of hmc at <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">beta</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0000</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:02</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:03</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:03</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">2</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:03</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:04</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">4</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:04</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">5</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:04</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">6</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:04</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">7</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:05</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">8</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:05</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">9</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:05</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">10</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:06</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">11</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:06</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">12</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:06</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">13</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:06</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">14</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:07</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">15</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:07</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">16</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:07</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">17</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:08</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">18</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:08</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">19</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:08</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">20</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:08</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">21</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:09</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">22</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:09</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">23</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:09</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">24</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:09</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">25</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:10</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">l2hmc</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">common</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">97</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>energy: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">2</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-2483.55359092</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2557.30777147</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2569.68391661</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2415.52649303</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2542.98323085</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2475.07511443</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2411.8236972</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2323.06436417</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2470.44317764</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2447.26703789</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2427.40503379</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2453.62779918</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2400.80537591</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2324.41454227</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2490.75272387</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2591.1159088</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2397.93020104</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2402.67231173</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2371.30690534</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2338.84347499</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2519.45503186</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2281.22974801</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2329.74864504</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2491.90123284</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2440.27715867</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2598.22569114</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2584.9812363</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2378.1701018</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2316.85535187</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2393.31637321</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2484.60172331</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2377.06200585</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2496.29136542</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2534.28031452</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2577.53192</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2436.63139345</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2399.57716185</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2347.47361136</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2441.95535438</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2566.58408094</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2460.29163585</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2593.38250749</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2441.04460458</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2431.28985458</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2401.89810369</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2637.68941046</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2420.85507377</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2475.42062047</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2303.10952922</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2628.80351088</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2239.88008518</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2508.72357389</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2366.20981104</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2503.88156318</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2363.30325862</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2379.93114517</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2591.33883052</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2389.58739512</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2460.94758518</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2368.76791198</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2549.8729475</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2339.2945588</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2541.8599455</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2634.88680707</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-2484.53700102</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2558.13796442</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2570.44783922</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2416.78334252</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2544.93850321</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2476.62418987</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2412.02033035</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2324.07202827</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2471.6157443</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2449.93060316</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2430.90050805</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2454.22800804</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2402.28338057</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2325.30228278</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2491.10609742</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2592.01868192</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2398.29504781</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2406.03559186</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2373.35397906</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2339.8206204</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2520.49282984</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2282.09319473</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2331.43112897</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2493.19697579</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2441.21516352</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2599.51284394</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2586.90173407</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2379.163794</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2317.61961495</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2393.25685798</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2485.4061845</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2378.34426804</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2498.0874452</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2534.44460039</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2579.43566993</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2437.97747324</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2401.28995612</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2348.30897589</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2443.38738316</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2568.82647561</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2459.94920269</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2594.97170165</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2442.77039709</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2431.70446786</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2403.64518194</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2638.77404065</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2422.42198488</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2476.04203931</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2303.80389373</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2631.56715884</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2240.97156957</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2510.53060623</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2367.52135632</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2505.486682</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2365.00583508</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2380.94794184</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2591.42943766</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2392.46872054</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2461.55798761</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2368.94580808</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2549.7306536</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2340.77982804</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2542.13206284</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2637.55489518</span><span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
logprob: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">2</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-2483.55359092</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2557.30777147</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2569.68391661</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2415.52649303</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2542.98323085</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2475.07511443</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2411.8236972</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2323.06436417</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2470.44317764</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2447.26703789</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2427.40503379</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2453.62779918</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2400.80537591</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2324.41454227</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2490.75272387</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2591.1159088</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2397.93020104</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2402.67231173</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2371.30690534</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2338.84347499</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2519.45503186</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2281.22974801</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2329.74864504</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2491.90123284</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2440.27715867</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2598.22569114</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2584.9812363</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2378.1701018</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2316.85535187</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2393.31637321</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2484.60172331</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2377.06200585</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2496.29136542</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2534.28031452</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2577.53192</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2436.63139345</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2399.57716185</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2347.47361136</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2441.95535438</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2566.58408094</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2460.29163585</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2593.38250749</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2441.04460458</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2431.28985458</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2401.89810369</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2637.68941046</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2420.85507377</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2475.42062047</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2303.10952922</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2628.80351088</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2239.88008518</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2508.72357389</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2366.20981104</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2503.88156318</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2363.30325862</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2379.93114517</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2591.33883052</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2389.58739512</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2460.94758518</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2368.76791198</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2549.8729475</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2339.2945588</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2541.8599455</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2634.88680707</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-2484.53700102</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2558.13796442</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2570.44783922</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2416.78334252</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2544.93850321</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2476.62418987</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2412.02033035</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2324.07202827</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2471.6157443</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2449.93060316</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2430.90050805</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2454.22800804</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2402.28338057</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2325.30228278</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2491.10609742</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2592.01868192</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2398.29504781</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2406.03559186</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2373.35397906</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2339.8206204</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2520.49282984</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2282.09319473</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2331.43112897</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2493.19697579</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2441.21516352</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2599.51284394</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2586.90173407</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2379.163794</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2317.61961495</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2393.25685798</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2485.4061845</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2378.34426804</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2498.0874452</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2534.44460039</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2579.43566993</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2437.97747324</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2401.28995612</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2348.30897589</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2443.38738316</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2568.82647561</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2459.94920269</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2594.97170165</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2442.77039709</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2431.70446786</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2403.64518194</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2638.77404065</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2422.42198488</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2476.04203931</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2303.80389373</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2631.56715884</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2240.97156957</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2510.53060623</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2367.52135632</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2505.486682</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2365.00583508</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2380.94794184</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2591.42943766</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2392.46872054</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2461.55798761</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2368.94580808</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2549.7306536</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2340.77982804</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2542.13206284</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2637.55489518</span><span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
logdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">2</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
acc: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.94222118</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.71004058</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.8673663</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.        <span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
sumlogdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
acc_mask: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float32 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
plaqs: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26565118</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26675697</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26671521</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2595438</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26608945</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26141018</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26772605</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26227424</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26278102</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27397995</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26129312</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.271993</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26490271</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26200964</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26904515</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.28261108</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27097516</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26464937</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25703543</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26735446</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26577915</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25128268</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25473539</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25975797</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2738854</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27907766</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2785959</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25881592</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2631273</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2724468</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27273498</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2628463</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.275129</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27743835</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27716043</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26745182</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26300919</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26667102</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25733196</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26980392</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26217471</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27609216</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2630092</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26988129</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25770703</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2762922</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26734789</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26800669</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25643129</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27200483</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.25529696</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27413101</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26149775</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27018596</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26550158</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27180247</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2718259</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26682019</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.26721861</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2603834</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27303692</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27150009</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.2745546</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.27679144</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
sinQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.84026401e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.92662057e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.35862543e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.63013362e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.83385464e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.32393272e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.77247034e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.73649495e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.30134375e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.96021325e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.12179935e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.39696482e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.60010134e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.45515011e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.18920775e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.43361429e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.18954696e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.11907862e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.59103483e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.64856149e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.80948591e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.01897732e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.51695166e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.51794426e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.78045430e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.76078969e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.35992328e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.49423279e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.20790558e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.03165028e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.83053863e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.87412864e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.24124601e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.15024839e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.23475950e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.87708322e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.80834946e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.00766291e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.16497801e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.28580249e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.43794092e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.94030700e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.28866862e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.91934899e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.22297582e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.70361817e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.09961028e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.32440148e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.89991219e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.86293442e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.77291515e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.43798040e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.28663530e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.72099072e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.47855068e-05</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.39361577e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.13853715e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.03473121e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.13515281e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.54638187e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.94816779e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.76686950e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.72354211e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.38130555e-03</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
intQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.07062066</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02810988</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00781837</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03837431</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0559369</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.10685801</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.09881204</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.06910665</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.09193818</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.07237075</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.03095758</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02038207</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03793613</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.03582125</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01735084</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.10845829</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.10489729</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00892789</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.09616485</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.09700418</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0409911</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.05863789</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0095084</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02214719</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.06974803</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.02569036</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.09279287</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.02180123</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0613944</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.07341304</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.04129827</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05652451</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.12024184</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03137266</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0617862</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.08574812</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01431063</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.04388256</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.07535832</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01876018</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05016042</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05749007</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.018802</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05718428</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.07620453</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.05403672</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.05981434</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01932335</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02772019</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01439027</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.08422828</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06475125</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06254308</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.06888044</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00109114</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.12246496</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.10415304</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.15096987</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03115241</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0517426</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.08678526</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01133206</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05432741</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.04933409</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
dQint: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00096735</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00835489</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02496554</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00922351</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03155457</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01917131</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00343928</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01049283</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01229644</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.011411</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01283495</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01449024</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03320719</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03010577</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00253129</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01562862</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00347696</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01170708</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.04563128</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02836631</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01601253</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02631382</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00205165</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01545652</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01814839</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00662477</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01396899</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02540488</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0054182</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01205654</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00785633</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01136065</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0472901</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02498471</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02936974</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01632454</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01191557</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03719811</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02561164</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0100296</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01311732</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03833206</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00960854</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01056563</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01297445</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01020872</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0216924</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01885664</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01043572</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03707911</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02485306</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00116944</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0253031</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01734161</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02022261</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01325454</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03177384</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00011639</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01843065</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05295013</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.         <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03532572</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00079687</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.04828707</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
dQsin: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">6.63014135e-05</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.72635206e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.71111138e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.32169139e-04</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.16271600e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.31398098e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.35724396e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.19166931e-04</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.42784648e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.82097814e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.79693581e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.93145051e-04</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.27598507e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.06341675e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.73491855e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.07116905e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.38306908e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.02390350e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.12751896e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.94419657e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.09748174e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.80352047e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.40617542e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.05937345e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.24387077e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.54054909e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.57419267e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.74122269e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.71357349e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.26342113e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.38464670e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.78646985e-04</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.24121261e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.71242494e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01297034e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.11886628e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.16679983e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.54951839e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.75539435e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.87417744e-04</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.99046732e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.62723822e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.58558696e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.24156733e-04</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.89254502e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.99694834e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.48677394e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.29241406e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.15253211e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.54136222e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.70340161e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.01522086e-05</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.73424684e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.18857503e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.38603587e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.08452147e-04</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.17774438e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.97749893e-06</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.26321668e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.62914499e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.42118665e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.46169136e-05</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.30954393e-03</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
loss: <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.008913131413735311</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:10</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">26</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:10</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">27</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:10</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">28</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:10</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">29</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:11</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">30</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:11</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">31</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:11</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">32</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:12</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">33</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:12</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">34</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:12</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">35</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:13</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">36</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:13</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">37</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:13</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">38</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:13</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">39</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:14</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">40</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:14</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">41</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:14</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">42</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:14</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">43</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:15</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">44</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:15</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">45</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:15</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">46</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:16</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">47</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:16</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">48</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:16</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">49</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:17</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3726184181</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">14</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">checkSU</span><span style="color: #ffffff; text-decoration-color: #ffffff">(</span>x_hmc<span style="color: #ffffff; text-decoration-color: #ffffff">)</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">(</span>tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3.116e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.699e-16</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">3.474e-16</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.353e-17</span>, tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">8.387e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.252e-15</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.014e-15</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">8.265e-17</span><span style="color: #ffffff; text-decoration-color: #ffffff">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:17</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving energy to plots-4dSU3/HMC
</pre>




    
![svg](output_10_54.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:17</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logprob to plots-4dSU3/HMC
</pre>




    
![svg](output_10_56.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:17</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logdet to plots-4dSU3/HMC
</pre>




    
![svg](output_10_58.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:17</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc to plots-4dSU3/HMC
</pre>




    
![svg](output_10_60.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:17</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sumlogdet to plots-4dSU3/HMC
</pre>




    
![svg](output_10_62.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:18</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc_mask to plots-4dSU3/HMC
</pre>




    
![svg](output_10_64.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:18</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving plaqs to plots-4dSU3/HMC
</pre>




    
![svg](output_10_66.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:18</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sinQ to plots-4dSU3/HMC
</pre>




    
![svg](output_10_68.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:18</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving intQ to plots-4dSU3/HMC
</pre>




    
![svg](output_10_70.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:19</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQint to plots-4dSU3/HMC
</pre>




    
![svg](output_10_72.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:19</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQsin to plots-4dSU3/HMC
</pre>




    
![svg](output_10_74.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:20</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving loss to plots-4dSU3/HMC
</pre>




    
![svg](output_10_76.svg)
    


## Evaluation


```python
# ptExpSU3.trainer.dynamics.init_weights(
#     method='uniform',
#     min=-1e-16,
#     max=1e-16,
#     bias=True,
#     # xeps=0.001,
#     # veps=0.001,
# )
xeval, history_eval = evaluate(
    nsteps=50,
    exp=ptExpSU3,
    beta=6.0,
    x=state.x,
    job_type='eval',
    nlog=1,
    nprint=25,
    grab=True,
)
xeval = ptExpSU3.trainer.dynamics.unflatten(xeval)
logger.info(f"checkSU(x_eval): {g.checkSU(xeval)}")
plot_metrics(history_eval.history, title='Evaluate', marker='.')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:20</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">117</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Running <span style="color: #ff00ff; text-decoration-color: #ff00ff">50</span> steps of eval at <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">beta</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0000</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:20</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:21</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:22</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">2</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:23</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:23</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">4</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:24</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">5</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:25</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">6</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:25</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">7</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:26</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">8</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:27</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">9</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:27</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">10</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:28</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">11</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:29</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">12</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:29</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">13</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:30</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">14</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:31</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">15</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:31</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">16</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:32</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">17</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:33</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">18</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:33</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">19</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:34</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">20</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:35</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">21</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:35</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">22</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:36</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">23</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:36</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">24</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:37</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">25</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:38</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">l2hmc</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">common</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">97</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>energy: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">79.49445451</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">101.00635741</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-81.21837479</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-73.66244007</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-169.06303339</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-55.22017346</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-54.40422111</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">45.08682706</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.0424534</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-196.15911517</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">64.46543869</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">62.38160316</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.75490979</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-27.42819387</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-32.77947734</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-166.35563407</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-68.30758438</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">82.13738035</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">131.8534792</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-56.76658301</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-34.70132335</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-88.74457873</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.92343277</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-113.13175195</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">97.94059788</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-70.27511073</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-162.04345347</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-29.46810254</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">45.21666582</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">22.55426611</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">46.31853553</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.02670745</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">118.32143284</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-83.98844302</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-134.36114719</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-22.8834119</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-21.33646438</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.90423539</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-29.75755195</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-16.19885253</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.97128486</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-132.65659118</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-157.69997778</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-66.38769841</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">99.32692455</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.6700919</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">25.79328966</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-35.62142589</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">31.2777833</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-195.99551538</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-46.44948644</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">28.61400272</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.62992431</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-93.85934351</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-56.15298022</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-11.81614989</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-23.08367223</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-39.29136652</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-82.24095638</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">116.96387105</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">84.8692009</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-94.58639345</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-84.30394415</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-20.51809224</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">91.95566967</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">118.08675151</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-69.50420909</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-62.27864257</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-153.22651314</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-43.12490855</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-35.46865351</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">63.44021472</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">24.16166028</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-182.51005735</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">80.45369508</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">70.38491412</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.5779436</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-12.81597318</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-20.68883427</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-150.53056022</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-54.00945893</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">94.86626898</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">151.08381448</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-39.45249272</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-22.5376592</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-70.31040378</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">17.70389924</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-94.66955269</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">113.95914228</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-53.53007514</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-140.45837998</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.79108902</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">59.57656227</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">35.67194325</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">59.79837987</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.87607308</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">130.74717561</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-70.4677965</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-121.89883133</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-12.44909127</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-11.30370403</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.917969</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-19.04374239</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.49309313</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.01122023</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-115.72941638</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-139.64402429</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-48.80192053</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">114.29589582</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">12.0744224</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">33.91041706</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-26.86619037</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">49.43477179</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-180.28086479</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-25.40862303</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">50.63547116</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">22.16672682</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-77.27899944</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-40.90793677</span>
     <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.10448221</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-11.7406818</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-23.26840969</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-56.92989015</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">138.30345889</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">101.53522486</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-83.74898081</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-67.44565206</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-12.91308687</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">105.75229222</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">123.10808711</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-58.47229693</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-49.43633366</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-142.37965219</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-33.34569369</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.5150778</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">71.06588978</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">31.79674521</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-175.47076962</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">86.39643687</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">75.49630161</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.03357858</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.58752397</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.82114528</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-141.81008932</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-42.19426893</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">111.66499537</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">156.79759188</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-26.1645673</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.90574841</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-61.83452026</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">20.62152606</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-86.58360015</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">115.26311275</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-53.28908049</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-130.87142574</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.49479302</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">70.42222501</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">46.63136228</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">68.70633121</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">13.71016972</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">143.66599228</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-57.34345373</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-112.2107068</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.04144527</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.74823514</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">11.77495457</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.7284109</span>     <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.78422774</span>
     <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.96318489</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-108.50608538</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-122.95385295</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-39.32770019</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">118.76604315</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">21.31800245</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">42.39750191</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.91141125</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">52.25064361</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-175.35059023</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.98501419</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">55.2067126</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">28.96931808</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-68.08136449</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-33.92169419</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">11.91971141</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.56148961</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-17.18922131</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-50.71622825</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">147.45653353</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">108.57465937</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-73.89519618</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-62.31091917</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.73214169</span><span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
logprob: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">79.49445451</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">101.00635741</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-81.21837479</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-73.66244007</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-169.06303339</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-55.22017346</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-54.40422111</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">45.08682706</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.0424534</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-196.15911517</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">64.46543869</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">62.38160316</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.75490979</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-27.42819387</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-32.77947734</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-166.35563407</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-68.30758438</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">82.13738035</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">131.8534792</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-56.76658301</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-34.70132335</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-88.74457873</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.92343277</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-113.13175195</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">97.94059788</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-70.27511073</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-162.04345347</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-29.46810254</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">45.21666582</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">22.55426611</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">46.31853553</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.02670745</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">118.32143284</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-83.98844302</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-134.36114719</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-22.8834119</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-21.33646438</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.90423539</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-29.75755195</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-16.19885253</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.97128486</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-132.65659118</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-157.69997778</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-66.38769841</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">99.32692455</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.6700919</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">25.79328966</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-35.62142589</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">31.2777833</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-195.99551538</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-46.44948644</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">28.61400272</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.62992431</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-93.85934351</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-56.15298022</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-11.81614989</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-23.08367223</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-39.29136652</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-82.24095638</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">116.96387105</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">84.8692009</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-94.58639345</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-84.30394415</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-20.51809224</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">91.29595024</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">116.49953859</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-72.24341965</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-63.70052653</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-153.65981084</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-45.98264236</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-35.36508712</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">62.3859076</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">23.28243206</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-184.64794498</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">78.24857106</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">68.2908182</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.85468504</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-15.88357289</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-22.67040041</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-152.93776349</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-54.72929995</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">94.36460934</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">149.52636071</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-40.16546168</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-23.29567101</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-73.35831635</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">15.69231983</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-97.16918329</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">111.39750611</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-55.70298398</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-143.61880175</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-12.97944437</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">58.44695673</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">35.32383654</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">57.81913871</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.99340481</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">128.23688194</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-71.54147042</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-123.0942146</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-12.99854931</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-12.55677512</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.05304138</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-20.99351814</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.29089274</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.71141351</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-117.74246928</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-142.95627828</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-51.61632777</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">111.95726676</span>
     <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.60754109</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">30.76470827</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-27.26574257</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">46.57562713</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-181.70035372</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-28.7031146</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">47.72863527</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">20.17425906</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-80.54205919</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-43.52826353</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.48276199</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-13.28159265</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-25.76493914</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-59.97673391</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">137.53587945</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">100.78676379</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-85.04736257</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-70.23559051</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-15.31297143</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">105.75203175</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">123.40522114</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-58.25890016</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-49.50808365</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-142.45946801</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-33.27800549</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.37622895</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">71.20654289</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">31.77662919</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-175.34431256</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">86.18756536</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">75.71390689</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.88412376</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.58056854</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-18.81291117</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-141.85559323</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-42.19841505</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">111.59509963</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">156.45913414</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-26.08469958</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.00213508</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-61.99589193</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">20.95772851</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-86.38855421</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">115.55226353</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-53.74782825</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-131.08130084</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.46744538</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">70.43817327</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">46.70525794</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">68.68403472</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">13.71047619</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">143.72282529</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-57.2905513</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-112.41585618</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.16129362</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.95551375</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">11.39432335</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.53312617</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.07875374</span>
     <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.82525745</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-108.37231032</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-122.99000229</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-39.32378265</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">118.67669215</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">21.00851467</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">42.19164381</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.88915346</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">51.90250698</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-175.93494925</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.96852561</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">55.11921125</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">29.16037182</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-68.14889816</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-33.8873187</span>
    <span style="color: #ff00ff; text-decoration-color: #ff00ff">11.53811671</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.20720299</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-17.39987266</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-50.86279135</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">147.88630561</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">108.37842981</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-73.9582958</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-62.37739983</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.86172473</span><span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
logdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.59719432e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.58721292e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.73921056e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.42188396e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.33297706e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.85773382e+00</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.03566383e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.05430713e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.79228223e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.13788764e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.20512402e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.09409592e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.27674144e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.06759971e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.98156614e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.40720327e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.19841025e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.01659647e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.55745378e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.12968957e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.58011813e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.04791258e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01157941e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.49963059e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.56163617e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.17290884e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.16042177e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.18835535e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.12960554e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.48106704e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.97924116e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.88266827e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.51029367e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.07367392e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.19538328e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.49458040e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.25307109e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.64927619e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.94977575e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.79779961e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.70019329e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01305290e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.31225399e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.81440724e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.33862905e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.46688130e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.14570880e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.99552197e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.85914466e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.41948893e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.29449157e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.90683590e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.99246776e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.26305975e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.62032676e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.58724420e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.54091085e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.49652945e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.04684376e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.67579441e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.48461067e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.29838177e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.78993845e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.39988456e+00</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.60464470e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.97134032e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.13396777e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.17499880e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.98158169e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.76882015e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.38848849e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.40653107e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01160176e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.26457064e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.08871510e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.17605279e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.49454815e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.95543163e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.23411258e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.55039008e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.14611943e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.98957348e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.38457747e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.98677207e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.63866647e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.61371668e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.36202458e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.95045941e-01</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.89150784e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.58747757e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.09875092e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.73476438e-02</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.59482568e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.38956628e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.22964880e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.06469343e-04</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.68330099e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.29024315e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.05149385e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.19848349e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.07278611e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.80631226e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.95284735e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.05474009e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.37927437e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.33775065e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.61493367e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.91753950e-03</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.93509987e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.09487783e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.05858101e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.22577897e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.48136631e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.84359020e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.64885779e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.75013458e-02</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.91053739e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.75336795e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.43754885e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.81594694e-01</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.54286621e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.10651350e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.46563109e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.29772077e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.96229559e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.30996179e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.64806593e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.29583041e-01</span><span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
sldf: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.        <span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.65971943</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.58721292</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.73921056</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.42188396</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.43329771</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.85773382</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.10356638</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.05430713</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.87922822</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.13788764</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.20512402</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.09409592</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.27674144</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.06759971</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.98156614</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.40720327</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.71984102</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.50165965</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.55745378</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.71296896</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.75801181</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.04791258</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01157941</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.49963059</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.56163617</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.17290884</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.16042177</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.18835535</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.12960554</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.3481067</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.97924116</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.88266827</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.51029367</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.07367392</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.19538328</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.54945804</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.25307109</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.86492762</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.94977575</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.79779961</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.70019329</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.0130529</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.31225399</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.81440724</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.33862905</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.4668813</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.1457088</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.3995522</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.85914466</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.41948893</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.29449157</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.9068359</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.99246776</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.26305975</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.62032676</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.5872442</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.54091085</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.49652945</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.04684376</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.76757944</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.74846107</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.29838177</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.78993845</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.39988456</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.        <span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
sldb: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.        <span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.          <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.        <span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.65945897</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.88434695</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.95260734</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.35013397</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.35348189</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.92542202</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.03528247</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.19496024</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.85911221</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.2643447</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.99625251</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.31170119</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.12728663</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.07455514</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.98980025</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.36169937</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.71569491</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.43176391</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.21899603</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.79283668</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.66162515</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.88654091</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.34778187</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.69467654</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.85078696</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.71416108</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.95054668</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.21570299</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.1455538</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.42200237</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.95694468</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.88297474</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.56712668</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.12657635</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.99023389</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.42960969</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.04579248</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.48429639</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.14506048</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.0923256</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.56226585</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.14682796</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.27610465</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.81832478</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.24927806</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.15739352</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.93985069</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.42180999</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.51100803</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.83512991</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.31098014</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.81933455</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.1835215</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.19552607</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.65470225</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.2056495</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.89519747</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.2858781</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.90028065</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.19735152</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.55223151</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.23528215</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.72345779</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.27030152</span><span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
sld: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00000000e+00</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.59719432e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.58721292e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.73921056e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.42188396e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.33297706e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.85773382e+00</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.03566383e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.05430713e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.79228223e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.13788764e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.20512402e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.09409592e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.27674144e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.06759971e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.98156614e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.40720327e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.19841025e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.01659647e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.55745378e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.12968957e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.58011813e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.04791258e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01157941e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.49963059e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.56163617e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.17290884e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.16042177e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.18835535e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.12960554e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.48106704e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.97924116e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.88266827e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.51029367e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.07367392e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.19538328e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.49458040e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.25307109e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.64927619e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.94977575e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.79779961e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.70019329e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01305290e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.31225399e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.81440724e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.33862905e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.46688130e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.14570880e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.99552197e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.85914466e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.41948893e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.29449157e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.90683590e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.99246776e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.26305975e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.62032676e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.58724420e+00</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.54091085e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.49652945e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.04684376e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.67579441e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.48461067e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.29838177e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.78993845e+00</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.39988456e+00</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
 <span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.60464470e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.97134032e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.13396777e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.17499880e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.98158169e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.76882015e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.38848849e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.40653107e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.01160176e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.26457064e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.08871510e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.17605279e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.49454815e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.95543163e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.23411258e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.55039008e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.14611943e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.98957348e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.38457747e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.98677207e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.63866647e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.61371668e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.36202458e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.95045941e-01</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.89150784e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.58747757e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.09875092e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.73476438e-02</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.59482568e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.38956628e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.22964880e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.06469343e-04</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.68330099e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.29024315e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.05149385e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.19848349e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.07278611e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.80631226e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.95284735e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.05474009e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.37927437e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.33775065e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.61493367e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.91753950e-03</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.93509987e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.09487783e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.05858101e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.22577897e-02</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.48136631e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.84359020e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.64885779e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.75013458e-02</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.91053739e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.75336795e-02</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.43754885e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.81594694e-01</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.54286621e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.10651350e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.46563109e-01</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.29772077e-01</span>
   <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.96229559e-01</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.30996179e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.64806593e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.29583041e-01</span><span style="color: #ffffff; text-decoration-color: #ffffff">]]</span>
xeps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
veps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
acc: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3.94892690e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.87196222e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.06862877e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.23516395e-11</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.79394824e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.95554467e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.25549469e-16</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.53264334e-12</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.92468461e-11</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.12527759e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.68298755e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.62126519e-06</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.00209262e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.18889267e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.59799883e-07</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.28964132e-11</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.58069981e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.60944102e-13</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.06015473e-11</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.73181039e-14</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.90214919e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.41653407e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.47165115e-08</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.42983506e-12</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.24569510e-08</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.64190131e-08</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.57526275e-14</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.90985768e-13</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.11285477e-11</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.24606725e-11</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.93547300e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.62794835e-10</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.29641336e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.54245309e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.94632880e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.72140301e-09</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.04052991e-08</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.62540076e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.64680995e-09</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.74451030e-10</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.70215264e-11</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.84099730e-11</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.42653791e-16</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.76315595e-12</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.94914160e-09</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.04567920e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.55588394e-08</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.03705891e-13</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.10356092e-09</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.94002276e-09</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.94057777e-17</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.08272079e-12</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.06802940e-13</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.82488691e-12</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.13868126e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.20063647e-11</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">8.57955187e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.10917391e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.35850106e-14</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.72012305e-14</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.16696678e-11</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.09984382e-09</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.00208368e-10</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.90028105e-09</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
sumlogdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
beta: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0</span>
acc_mask: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float32 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
plaqs: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00497154</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00414064</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00427913</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00226867</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01663784</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00616541</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00105746</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00156758</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0091615</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0086752</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0109217</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00100653</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0041472</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00032765</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00272777</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00766932</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00808775</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00060328</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00766759</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00063916</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00489646</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00311886</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.005582</span>    <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00282993</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0061585</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00284356</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01144724</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00465963</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00927236</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00389873</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00213857</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0054465</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00887319</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00673512</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00611957</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00114471</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00163998</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00281674</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0005934</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00579626</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00218655</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00311794</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00972355</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00286911</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0058137</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00290567</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00325403</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00201399</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00972395</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00693588</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00703437</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00266406</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00090393</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0023359</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0056368</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00101901</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00247059</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0108304</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00577694</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00818003</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00150447</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00627092</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0004898</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00180721</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
sinQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.16478333e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.49243489e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.94720443e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.30174004e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.00487302e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.68941722e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.94387800e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.39006967e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.58893271e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.91929263e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.24948727e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.42618623e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-9.83745088e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.92372817e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.47574640e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.64737678e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.40989902e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.20764285e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.56257960e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.09584116e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.47145653e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.43198810e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.15502967e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.15040679e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.03899639e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.05862114e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.12735582e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.10788349e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.58630773e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.16495029e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.19444686e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-3.41869322e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.93326148e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-8.72929362e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.48626261e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">7.93156876e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.77004745e-05</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.46817158e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.51133880e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.30116936e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.61433857e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.96051370e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-7.24651099e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.23028105e-02</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.29978415e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.29134322e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">3.63601498e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.31530121e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.59927039e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.22738322e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.68606192e-04</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.74072773e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.11515872e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.42404059e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">1.09845817e-02</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.22761819e-03</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">6.23431714e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.46737359e-04</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.66783640e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-2.93462384e-03</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-6.85219562e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">4.11077535e-03</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-5.59287835e-03</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">5.73603354e-03</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
intQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.06076523</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03636525</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.1159517</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.10653422</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.14661349</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03923927</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01159032</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.07864247</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.03777318</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.02800296</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.06200108</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.04998892</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01435309</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.28067676</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.07989251</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06780639</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.09352203</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01761981</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.02279843</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0159886</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05064942</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03548332</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06062292</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03137497</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.08811047</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.15445548</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0456289</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.08911555</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.12527638</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00899482</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.10496878</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.04987959</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.08656777</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01273626</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.02168494</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.11572357</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00069596</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.03601124</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03664106</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03357464</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.08191461</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00286044</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01057284</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.17950109</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.04814468</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06261177</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05305037</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.09214183</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01400558</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.07626883</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.01413221</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0691684</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.01627045</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03536736</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.1602678</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03250151</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.09096025</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00359996</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0389244</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0428169</span>
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.09997525</span>  <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05997724</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0816015</span>   <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.08369017</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
dQint: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
dQsin: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">])</span> torch.float64 
<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.
 <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>. <span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span>.<span style="color: #ffffff; text-decoration-color: #ffffff">]</span>
loss: <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-4.866806313547992e-10</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:38</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">26</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:38</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">27</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:39</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">28</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">29</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">30</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:41</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">31</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:42</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">32</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:43</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">33</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:43</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">34</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:44</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">35</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:45</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">36</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:45</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">37</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:46</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">38</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:46</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">39</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:47</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">40</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:48</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">41</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:48</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">42</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:49</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">43</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:49</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">44</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:51</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">45</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:51</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">46</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:52</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">47</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:53</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">48</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:53</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">pytorch</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">experiment</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">121</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc.experiment.pytorch.experiment</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">49</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:54</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">2686438015</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">20</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">checkSU</span><span style="color: #ffffff; text-decoration-color: #ffffff">(</span>x_eval<span style="color: #ffffff; text-decoration-color: #ffffff">)</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">(</span>tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">1.747e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">2.043e-16</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.895e-16</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.744e-18</span>, tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">6.246e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">9.007e-16</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">7.611e-16</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">5.544e-17</span><span style="color: #ffffff; text-decoration-color: #ffffff">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:54</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving energy to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_54.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:54</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logprob to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_56.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:54</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logdet to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_58.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:55</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldf to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_60.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:55</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldb to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_62.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:55</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sld to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_64.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:55</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving xeps to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_66.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving veps to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_68.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_70.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sumlogdet to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_72.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving beta to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_74.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc_mask to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_76.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving plaqs to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_78.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sinQ to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_80.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving intQ to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_82.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:58</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQint to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_84.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:58</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQsin to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_86.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:49:58</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving loss to plots-4dSU3/Evaluate
</pre>




    
![svg](output_12_88.svg)
    


## Training


```python
history = {}
x = state.x
for step in range(50):
    x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
    if (step > 0 and step % 10 == 0):
        logger.info(f'TRAIN STEP: {step}')
        _metrics = {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        print_dict(_metrics, grab=True)
    if (step > 0 and step % 1 == 0):
        for key, val in metrics.items():
            try:
                history[key].append(val)
            except KeyError:
                history[key] = [val]

x = ptExpSU3.trainer.dynamics.unflatten(x)
logger.info(f"checkSU(x_train): {g.checkSU(x)}")
plot_metrics(history, title='train', marker='.')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:50:21</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3918287316</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">6</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>TRAIN STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">10</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:50:21</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">l2hmc</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">common</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">97</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>energy: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-667.6420884467811</span>
logprob: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-668.5198611560013</span>
logdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.8777727092201583</span>
sldf: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.8634789723300694</span>
sldb: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.8491852354399807</span>
sld: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.8777727092201583</span>
xeps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06786588053223853</span>
veps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.05219242197608772</span>
acc: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.6294382796251724</span>
sumlogdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03409656770521637</span>
beta: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0</span>
acc_mask: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float32 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.625</span>
loss: <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.009005818050184255</span>
plaqs: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0722341978542969</span>
sinQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0008759514672325299</span>
intQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.012780351284146906</span>
dQint: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.013722721066206539</span>
dQsin: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0009405404738191228</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:50:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3918287316</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">6</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>TRAIN STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">20</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:50:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">l2hmc</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">common</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">97</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>energy: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1245.3145585854759</span>
logprob: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1246.2921552569476</span>
logdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9775966714718195</span>
sldf: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9807152309193539</span>
sldb: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.983833790366888</span>
sld: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9775966714718195</span>
xeps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.07490975518633654</span>
veps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.044950955056694636</span>
acc: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.19925189314758668</span>
sumlogdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0013929522365201154</span>
beta: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0</span>
acc_mask: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float32 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.21875</span>
loss: <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0034123919862975</span>
plaqs: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.13631806191104623</span>
sinQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0008186531342882409</span>
intQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.011944354256437575</span>
dQint: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.003533569362731753</span>
dQsin: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.00024218702593035995</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:00</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3918287316</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">6</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>TRAIN STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">30</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:00</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">l2hmc</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">common</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">97</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>energy: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1355.8775734331887</span>
logprob: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1356.8464972370004</span>
logdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9689238038120794</span>
sldf: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9743043931045948</span>
sldb: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.9796849823971104</span>
sld: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9689238038120794</span>
xeps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.07814788465244042</span>
veps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.04110563094621567</span>
acc: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.06602050553167274</span>
sumlogdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0033283920941650676</span>
beta: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0</span>
acc_mask: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float32 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0625</span>
loss: <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0012154087028711587</span>
plaqs: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.14833780959388745</span>
sinQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0006137451668855338</span>
intQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.008954695693959323</span>
dQint: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.003168028703219909</span>
dQsin: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0002171332641116432</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:19</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3918287316</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">6</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>TRAIN STEP: <span style="color: #ff00ff; text-decoration-color: #ff00ff">40</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:19</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">l2hmc</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">common</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">97</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>energy: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1396.38155455171</span>
logprob: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1397.3717471161244</span>
logdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9901925644143645</span>
sldf: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9948478030098865</span>
sldb: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.9995030416054086</span>
sld: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9901925644143645</span>
xeps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.07903654312294543</span>
veps: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.03969318599197484</span>
acc: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.04748749449638852</span>
sumlogdet: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.006477426076826311</span>
beta: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0</span>
acc_mask: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float32 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.046875</span>
loss: <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff; font-style: italic">None</span> 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0009028692931150591</span>
plaqs: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.15431528443824147</span>
sinQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.00039612917663782</span>
intQ: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.005779623895518039</span>
dQint: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.001203357307478027</span>
dQsin: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ffffff; text-decoration-color: #ffffff">([])</span> torch.float64 
<span style="color: #ff00ff; text-decoration-color: #ff00ff">8.24768095692234e-05</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:36</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3918287316</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">17</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">checkSU</span><span style="color: #ffffff; text-decoration-color: #ffffff">(</span>x_train<span style="color: #ffffff; text-decoration-color: #ffffff">)</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">(</span>tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">1.950e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.010</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.001</span>, tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">7.454e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.051</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.001</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.006</span><span style="color: #ffffff; text-decoration-color: #ffffff">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:36</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving energy to plots-4dSU3/train
</pre>




    
![svg](output_14_10.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:36</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logprob to plots-4dSU3/train
</pre>




    
![svg](output_14_12.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:36</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logdet to plots-4dSU3/train
</pre>




    
![svg](output_14_14.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:36</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldf to plots-4dSU3/train
</pre>




    
![svg](output_14_16.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:37</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldb to plots-4dSU3/train
</pre>




    
![svg](output_14_18.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:37</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sld to plots-4dSU3/train
</pre>




    
![svg](output_14_20.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:37</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving xeps to plots-4dSU3/train
</pre>




    
![svg](output_14_22.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:37</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving veps to plots-4dSU3/train
</pre>




    
![svg](output_14_24.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:38</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc to plots-4dSU3/train
</pre>




    
![svg](output_14_26.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:38</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sumlogdet to plots-4dSU3/train
</pre>




    
![svg](output_14_28.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:38</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving beta to plots-4dSU3/train
</pre>




    
![svg](output_14_30.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:38</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc_mask to plots-4dSU3/train
</pre>




    
![svg](output_14_32.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:39</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving loss to plots-4dSU3/train
</pre>




    
![svg](output_14_34.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:39</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving plaqs to plots-4dSU3/train
</pre>




    
![svg](output_14_36.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:39</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sinQ to plots-4dSU3/train
</pre>




    
![svg](output_14_38.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving intQ to plots-4dSU3/train
</pre>




    
![svg](output_14_40.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQint to plots-4dSU3/train
</pre>




    
![svg](output_14_42.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:51:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">[l2hmc][4dSU3]</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQsin to plots-4dSU3/train
</pre>




    
![svg](output_14_44.svg)
    



```python
#history = {}

import ezpz

logger = ezpz.get_logger('l2hmc-4dSU3-trainer')
history = ezpz.History()

x = state.x
for step in range(10):
    x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
    _metrics = {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    if step % 2 == 0:
        logger.info(history.update(_metrics, summarize=True, precision=4))

x = ptExpSU3.trainer.dynamics.unflatten(x)
logger.info(f"checkSU(x_train): {g.checkSU(x)}")
plot_metrics(history.history, title='train', marker='.')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:13</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">17990128</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">13</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">energy</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-37.7267</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logprob</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-38.7513</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0246</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0217</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.0189</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sld</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0246</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">xeps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0792</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">veps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0393</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sumlogdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0086</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">beta</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc_mask</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0190</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">plaqs</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0010</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sinQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0002</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">intQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0030</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQint</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0359</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQsin</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0025</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:17</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">17990128</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">13</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">energy</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-370.7112</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logprob</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-371.7393</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0281</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0193</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.0105</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sld</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0281</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">xeps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0793</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">veps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0392</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sumlogdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0264</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">beta</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc_mask</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0190</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">plaqs</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0362</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sinQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0008</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">intQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0111</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQint</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0286</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQsin</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0020</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:21</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">17990128</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">13</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">energy</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-605.4306</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logprob</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-606.4347</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0042</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0006</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.9971</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sld</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0042</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">xeps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0796</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">veps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0392</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9553</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sumlogdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0109</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">beta</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc_mask</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9531</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0184</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">plaqs</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0658</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sinQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0011</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">intQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0159</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQint</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0308</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQsin</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0021</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:25</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">17990128</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">13</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">energy</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-809.9318</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logprob</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-810.9380</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0062</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0107</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.0151</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sld</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0062</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">xeps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0801</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">veps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0392</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.8991</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sumlogdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0060</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">beta</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc_mask</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.9062</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0174</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">plaqs</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0880</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sinQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0008</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">intQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0110</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQint</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0240</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQsin</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0016</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:29</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">17990128</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">13</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">energy</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-988.5552</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logprob</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-989.5746</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">logdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0194</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0276</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sldb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-1.0358</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sld</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1.0194</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">xeps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0806</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">veps</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0393</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.6809</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sumlogdet</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0199</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">beta</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6.0000</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">acc_mask</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.6719</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">-0.0133</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">plaqs</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.1064</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">sinQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0011</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">intQ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0157</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQint</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0189</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dQsin</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.0013</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:31</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">17990128</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">16</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">checkSU</span><span style="color: #ffffff; text-decoration-color: #ffffff">(</span>x_train<span style="color: #ffffff; text-decoration-color: #ffffff">)</span>: <span style="color: #ffffff; text-decoration-color: #ffffff">(</span>tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">3.052e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.011</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.005</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.005</span>, tensor<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">64</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> f64 x∈<span style="color: #ffffff; text-decoration-color: #ffffff">[</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">8.817e-16</span>, <span style="color: #ff00ff; text-decoration-color: #ff00ff">0.053</span><span style="color: #ffffff; text-decoration-color: #ffffff">]</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">μ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.019</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">σ</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.020</span><span style="color: #ffffff; text-decoration-color: #ffffff">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:31</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving energy to plots-4dSU3/train
</pre>




    
![svg](output_15_7.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:31</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logprob to plots-4dSU3/train
</pre>




    
![svg](output_15_9.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:31</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logdet to plots-4dSU3/train
</pre>




    
![svg](output_15_11.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:32</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldf to plots-4dSU3/train
</pre>




    
![svg](output_15_13.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:32</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldb to plots-4dSU3/train
</pre>




    
![svg](output_15_15.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:32</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sld to plots-4dSU3/train
</pre>




    
![svg](output_15_17.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:32</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving xeps to plots-4dSU3/train
</pre>




    
![svg](output_15_19.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:33</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving veps to plots-4dSU3/train
</pre>




    
![svg](output_15_21.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:33</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc to plots-4dSU3/train
</pre>




    
![svg](output_15_23.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:33</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sumlogdet to plots-4dSU3/train
</pre>




    
![svg](output_15_25.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:33</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving beta to plots-4dSU3/train
</pre>




    
![svg](output_15_27.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:34</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc_mask to plots-4dSU3/train
</pre>




    
![svg](output_15_29.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:34</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving loss to plots-4dSU3/train
</pre>




    
![svg](output_15_31.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:34</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving plaqs to plots-4dSU3/train
</pre>




    
![svg](output_15_33.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:34</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sinQ to plots-4dSU3/train
</pre>




    
![svg](output_15_35.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:35</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving intQ to plots-4dSU3/train
</pre>




    
![svg](output_15_37.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:35</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQint to plots-4dSU3/train
</pre>




    
![svg](output_15_39.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:35</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_53580</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">297162003</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">59</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">l2hmc-4dSU3-trainer</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQsin to plots-4dSU3/train
</pre>




    
![svg](output_15_41.svg)
    



```python
dataset = history.finalize()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:55</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving energy plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_1.svg)
    



    
![svg](output_16_2.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logprob plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_4.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving logdet plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_6.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldf plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_8.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sldb plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_10.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sld plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_12.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving xeps plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_14.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving veps plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_16.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_18.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sumlogdet plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_20.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:56</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving beta plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_22.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving acc_mask plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_24.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving loss plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_26.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving plaqs plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_28.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving sinQ plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_30.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving intQ plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_32.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQint plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_34.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">704</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dQsin plot to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">mplot</span>
</pre>




    
![svg](output_16_36.svg)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">history</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">602</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving tplots to <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/</span><span style="color: #800080; text-decoration-color: #800080">tplot</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                    energy [2025-04-04-085257]              
      ┌────────────────────────────────────────────────────┐
 -37.7┤▚▖                                                  │
      │ ▝▀▄                                                │
-196.2┤    ▀▚▄                                             │
      │       ▀▄▖                                          │
      │         ▝▚▄                                        │
-354.7┤            ▀▚▄                                     │
      │               ▀▀▄▄                                 │
-513.1┤                   ▀▚▄▖                             │
      │                      ▝▀▚▄▖                         │
-671.6┤                          ▝▀▄▄                      │
      │                              ▀▚▄▖                  │
      │                                 ▝▀▄▄               │
-830.1┤                                     ▀▀▄▄▖          │
      │                                         ▝▀▀▄▄▖     │
-988.6┤                                              ▝▀▀▄▄▄│
      └┬────────────┬────────────┬───────────┬─────────────┘
       1            2            3           4              
energy                         iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/energy.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                    logprob [2025-04-04-085257]             
      ┌────────────────────────────────────────────────────┐
 -38.8┤▚▖                                                  │
      │ ▝▀▄                                                │
-197.2┤    ▀▚▄                                             │
      │       ▀▄▖                                          │
      │         ▝▚▄                                        │
-355.7┤            ▀▚▄                                     │
      │               ▀▀▄▄                                 │
-514.2┤                   ▀▚▄▖                             │
      │                      ▝▀▚▄▖                         │
-672.6┤                          ▝▀▄▄                      │
      │                              ▀▚▄▖                  │
      │                                 ▝▀▄▄               │
-831.1┤                                     ▀▀▄▄▖          │
      │                                         ▝▀▀▄▄▖     │
-989.6┤                                              ▝▀▀▄▄▄│
      └┬────────────┬────────────┬───────────┬─────────────┘
       1            2            3           4              
logprob                        iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/logprob.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                    logdet [2025-04-04-085257]              
      ┌────────────────────────────────────────────────────┐
1.0281┤          ▄▄▄▌                                      │
      │   ▗▄▄▞▀▀▀   ▝▖                                     │
1.0241┤▀▀▀▘          ▝▖                                    │
      │               ▝▖                                   │
      │                ▝▖                                  │
1.0202┤                 ▚                                 ▗│
      │                  ▚                               ▄▘│
1.0162┤                   ▚                            ▗▞  │
      │                    ▚                          ▞▘   │
1.0122┤                     ▚                       ▄▀     │
      │                     ▝▖                    ▗▞       │
      │                      ▝▖                  ▄▘        │
1.0082┤                       ▝▖               ▗▀          │
      │                        ▝▖            ▗▞▘           │
1.0042┤                         ▝▄▄▄▄▄▄▞▀▀▀▀▀▘             │
      └┬────────────┬────────────┬───────────┬─────────────┘
       1            2            3           4              
logdet                         iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/logdet.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     sldf [2025-04-04-085257]               
      ┌────────────────────────────────────────────────────┐
1.0276┤                                                   ▞│
      │                                                 ▗▞ │
1.0231┤                                                ▄▘  │
      │▚▄▄▄▖                                         ▗▞    │
      │    ▝▀▀▀▀▄▄▄▄▖                               ▄▘     │
1.0186┤             ▝▖                             ▞       │
      │              ▝▄                          ▗▀        │
1.0141┤                ▚                        ▞▘         │
      │                 ▀▖                    ▗▀           │
1.0096┤                  ▝▄                 ▗▞▘            │
      │                    ▚              ▗▞▘              │
      │                     ▚▖          ▄▀▘                │
1.0051┤                      ▝▖       ▄▀                   │
      │                       ▝▚   ▗▞▀                     │
1.0006┤                         ▚▄▞▘                       │
      └┬────────────┬────────────┬───────────┬─────────────┘
       1            2            3           4              
sldf                           iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/sldf.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     sldb [2025-04-04-085257]               
       ┌───────────────────────────────────────────────────┐
-0.9971┤                        ▗▞▖                        │
       │                      ▄▀▘ ▝▚▖                      │
-1.0036┤                   ▄▞▀      ▝▚▖                    │
       │                ▗▄▀           ▝▚▖                  │
       │              ▄▞▘               ▝▚▖                │
-1.0100┤          ▗▄▞▀                    ▝▚▖              │
       │      ▗▄▞▀▘                         ▝▚▖            │
-1.0165┤  ▗▄▞▀▘                               ▝▄           │
       │▀▀▘                                     ▚▖         │
-1.0229┤                                         ▝▄        │
       │                                           ▚▖      │
       │                                            ▝▚     │
-1.0294┤                                              ▀▖   │
       │                                               ▝▚  │
-1.0358┤                                                 ▀▄│
       └┬────────────┬───────────┬────────────┬────────────┘
        1            2           3            4             
sldb                           iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/sldb.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                      sld [2025-04-04-085257]               
      ┌────────────────────────────────────────────────────┐
1.0281┤          ▄▄▄▌                                      │
      │   ▗▄▄▞▀▀▀   ▝▖                                     │
1.0241┤▀▀▀▘          ▝▖                                    │
      │               ▝▖                                   │
      │                ▝▖                                  │
1.0202┤                 ▚                                 ▗│
      │                  ▚                               ▄▘│
1.0162┤                   ▚                            ▗▞  │
      │                    ▚                          ▞▘   │
1.0122┤                     ▚                       ▄▀     │
      │                     ▝▖                    ▗▞       │
      │                      ▝▖                  ▄▘        │
1.0082┤                       ▝▖               ▗▀          │
      │                        ▝▖            ▗▞▘           │
1.0042┤                         ▝▄▄▄▄▄▄▞▀▀▀▀▀▘             │
      └┬────────────┬────────────┬───────────┬─────────────┘
       1            2            3           4              
sld                            iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/sld.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     xeps [2025-04-04-085257]               
       ┌───────────────────────────────────────────────────┐
0.08062┤                                                 ▗▞│
       │                                               ▗▞▘ │
0.08037┤                                             ▄▀▘   │
       │                                           ▄▀      │
       │                                        ▗▞▀        │
0.08013┤                                      ▄▞▘          │
       │                                   ▗▄▀             │
0.07989┤                                ▗▄▀▘               │
       │                              ▄▞▘                  │
0.07965┤                           ▄▞▀                     │
       │                       ▗▄▞▀                        │
       │                   ▄▄▞▀▘                           │
0.07941┤               ▄▄▀▀                                │
       │        ▗▄▄▄▞▀▀                                    │
0.07917┤▄▄▄▄▞▀▀▀▘                                          │
       └┬────────────┬───────────┬────────────┬────────────┘
        1            2           3            4             
xeps                           iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/xeps.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                      veps [2025-04-04-085257]              
        ┌──────────────────────────────────────────────────┐
0.039262┤▚                                                 │
        │ ▀▖                                               │
0.039252┤  ▝▚                                             ▗│
        │    ▀▖                                          ▗▘│
        │     ▝▚▖                                       ▗▘ │
0.039242┤       ▝▄                                     ▗▘  │
        │         ▚▖                                  ▗▘   │
0.039232┤          ▝▄                                ▗▘    │
        │            ▀▖                             ▄▘     │
0.039221┤             ▝▚▖                          ▞       │
        │               ▝▚▖                       ▞        │
        │                 ▝▚▖                    ▞         │
0.039211┤                   ▝▚▖                 ▞          │
        │                     ▝▚▖              ▞           │
0.039201┤                       ▝▚▄▄▄▄▄▄▄▄▄▄▄▄▀            │
        └┬───────────┬────────────┬───────────┬────────────┘
         1           2            3           4             
veps                            iter                        
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/veps.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     acc [2025-04-04-085257]                
     ┌─────────────────────────────────────────────────────┐
1.000┤▀▀▀▀▀▀▀▀▀▀▀▀▀▚▄▄▖                                    │
     │                ▝▀▀▀▄▄▄▖                             │
0.947┤                       ▝▀▀▀▄▄▖                       │
     │                             ▝▀▀▄▄▖                  │
     │                                  ▝▀▀▄▄▄             │
0.894┤                                        ▚            │
     │                                         ▚▖          │
0.840┤                                          ▝▖         │
     │                                           ▝▚        │
0.787┤                                             ▚▖      │
     │                                              ▝▖     │
     │                                               ▝▄    │
0.734┤                                                 ▚   │
     │                                                  ▀▖ │
0.681┤                                                   ▝▄│
     └┬────────────┬────────────┬────────────┬─────────────┘
      1            2            3            4              
acc                           iter                          
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/acc.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                   sumlogdet [2025-04-04-085257]            
       ┌───────────────────────────────────────────────────┐
 0.0264┤           ▗▞▄                                     │
       │         ▗▞▘  ▀▚▖                                  │
 0.0187┤       ▄▀▘      ▝▀▄▖                               │
       │     ▄▀            ▝▚▄                             │
       │  ▗▞▀                 ▀▄▖                          │
 0.0110┤▄▞▘                     ▝▀▄                        │
       │                           ▀▚▖                     │
 0.0033┤                             ▝▀▄                   │
       │                                ▀▚▖                │
-0.0045┤                                  ▝▀▄              │
       │                                     ▀▚▖           │
       │                                       ▝▀▄▖        │
-0.0122┤                                          ▝▀▄      │
       │                                             ▀▚▄   │
-0.0199┤                                                ▀▚▄│
       └┬────────────┬───────────┬────────────┬────────────┘
        1            2           3            4             
sumlogdet                      iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/sumlogdet.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                  beta [2025-04-04-085257]                  
 ┌─────────────────────────────────────────────────────────┐
9┤                                                         │
 │                                                         │
8┤                                                         │
 │                                                         │
 │                                                         │
7┤                                                         │
 │                                                         │
6┤▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
 │                                                         │
5┤                                                         │
 │                                                         │
 │                                                         │
4┤                                                         │
 │                                                         │
3┤                                                         │
 └┬─────────────┬─────────────┬─────────────┬──────────────┘
  1             2             3             4               
beta                        iter                            
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/beta.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                  acc_mask [2025-04-04-085257]              
     ┌─────────────────────────────────────────────────────┐
1.000┤▀▀▀▀▀▀▀▀▀▀▀▀▀▚▄▄▖                                    │
     │                ▝▀▀▀▄▄▄▖                             │
0.945┤                       ▝▀▀▀▄▄▄                       │
     │                              ▀▀▀▚▄▄▖                │
     │                                    ▝▀▀▀▖            │
0.891┤                                        ▝▖           │
     │                                         ▝▄          │
0.836┤                                           ▚         │
     │                                            ▚▖       │
0.781┤                                             ▝▖      │
     │                                              ▝▄     │
     │                                                ▚    │
0.727┤                                                 ▚▖  │
     │                                                  ▝▖ │
0.672┤                                                   ▝▄│
     └┬────────────┬────────────┬────────────┬─────────────┘
      1            2            3            4              
acc_mask                      iter                          
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/acc_mask.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     loss [2025-04-04-085257]               
       ┌───────────────────────────────────────────────────┐
-0.0133┤                                                  ▞│
       │                                                 ▞ │
-0.0143┤                                                ▞  │
       │                                              ▗▀   │
       │                                             ▗▘    │
-0.0152┤                                            ▄▘     │
       │                                           ▞       │
-0.0162┤                                          ▞        │
       │                                        ▗▀         │
-0.0171┤                                       ▗▘          │
       │                                      ▄▘           │
       │                                 ▄▄▞▀▀             │
-0.0181┤                            ▄▄▞▀▀                  │
       │                     ▗▄▄▄▞▀▀                       │
-0.0190┤▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▀▀▀▀▘                             │
       └┬────────────┬───────────┬────────────┬────────────┘
        1            2           3            4             
loss                           iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/loss.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                    plaqs [2025-04-04-085257]               
     ┌─────────────────────────────────────────────────────┐
0.106┤                                                  ▄▄▞│
     │                                             ▄▄▞▀▀   │
0.089┤                                       ▗▄▄▞▀▀        │
     │                                   ▗▄▞▀▘             │
     │                               ▄▄▀▀▘                 │
0.071┤                          ▗▄▄▀▀                      │
     │                       ▗▄▀▘                          │
0.054┤                    ▄▞▀▘                             │
     │                ▗▄▞▀                                 │
0.036┤             ▄▄▀▘                                    │
     │          ▗▄▀                                        │
     │        ▄▞▘                                          │
0.019┤     ▗▞▀                                             │
     │   ▄▀▘                                               │
0.001┤▄▞▀                                                  │
     └┬────────────┬────────────┬────────────┬─────────────┘
      1            2            3            4              
plaqs                         iter                          
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/plaqs.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                      sinQ [2025-04-04-085257]              
        ┌──────────────────────────────────────────────────┐
 0.00109┤                       ▗▄▚▄                     ▄▞│
        │                    ▄▞▀▘   ▀▚▄               ▄▞▀  │
 0.00088┤                ▗▄▀▀          ▀▚▄         ▄▞▀     │
        │            ▗▄▞▀▘                ▀▚▄   ▄▞▀        │
        │           ▗▘                       ▀▀▀           │
 0.00066┤          ▗▘                                      │
        │         ▗▘                                       │
 0.00045┤        ▄▘                                        │
        │       ▞                                          │
 0.00023┤      ▞                                           │
        │     ▞                                            │
        │   ▗▀                                             │
 0.00001┤  ▗▘                                              │
        │ ▗▘                                               │
-0.00020┤▄▘                                                │
        └┬───────────┬────────────┬───────────┬────────────┘
         1           2            3           4             
sinQ                            iter                        
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/sinQ.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     intQ [2025-04-04-085257]               
       ┌───────────────────────────────────────────────────┐
 0.0159┤                        ▄▞▄▖                     ▄▞│
       │                    ▄▄▀▀   ▝▀▄▖               ▄▞▀  │
 0.0128┤                ▗▄▞▀          ▝▀▄▖         ▄▞▀     │
       │            ▗▄▞▀▘                ▝▀▄▖   ▄▞▀        │
       │           ▗▘                       ▝▀▀▀           │
 0.0096┤          ▗▘                                       │
       │         ▗▘                                        │
 0.0065┤        ▄▘                                         │
       │       ▞                                           │
 0.0033┤      ▞                                            │
       │     ▞                                             │
       │   ▗▀                                              │
 0.0002┤  ▗▘                                               │
       │ ▗▘                                                │
-0.0030┤▄▘                                                 │
       └┬────────────┬───────────┬────────────┬────────────┘
        1            2           3            4             
intQ                           iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/intQ.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     dQint [2025-04-04-085257]              
      ┌────────────────────────────────────────────────────┐
0.0359┤▚▖                                                  │
      │ ▝▚▖                                                │
0.0330┤   ▝▚▖                                              │
      │     ▝▀▄                                            │
      │        ▀▄                ▖                         │
0.0302┤          ▀▄     ▗▄▄▄▄▀▀▀▀▝▚▖                       │
      │            ▀▀▀▀▀▘          ▝▚▄                     │
0.0274┤                               ▀▄                   │
      │                                 ▀▄▖                │
0.0246┤                                   ▝▚▖              │
      │                                     ▝▀▄            │
      │                                        ▀▚▄         │
0.0217┤                                           ▀▚▄      │
      │                                              ▀▚▄   │
0.0189┤                                                 ▀▚▄│
      └┬────────────┬────────────┬───────────┬─────────────┘
       1            2            3           4              
dQint                          iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/dQint.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     dQsin [2025-04-04-085257]              
       ┌───────────────────────────────────────────────────┐
0.00246┤▚▖                                                 │
       │ ▝▚▖                                               │
0.00226┤   ▝▚▖                                             │
       │     ▝▚▖                                           │
       │       ▝▚▖               ▗                         │
0.00207┤         ▝▚▖     ▄▄▄▄▞▀▀▀▘▀▄                       │
       │           ▝▀▀▀▀▀           ▀▄▖                    │
0.00188┤                              ▝▚▖                  │
       │                                ▝▚▄                │
0.00168┤                                   ▀▄              │
       │                                     ▀▚▖           │
       │                                       ▝▀▄▖        │
0.00149┤                                          ▝▀▄      │
       │                                             ▀▚▄   │
0.00130┤                                                ▀▚▄│
       └┬────────────┬───────────┬────────────┬────────────┘
        1            2           3            4             
dQsin                          iter                         
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">text saved in</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/plots/tplot/dQsin.txt</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-04-04 </span><span style="color: #808080; text-decoration-color: #808080">08:52:57</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">utils</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">192</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span>Saving dataset to: <span style="color: #008000; text-decoration-color: #008000">/Users/samforeman/projects/saforem2/personal_site_CLEAN/posts/ai-for-physics/l2hmc-qcd/4dSU3/outputs/History-2025-04-04-085255/2025-04-04-085255/History-2025-04-04-085255/</span><span style="color: #800080; text-decoration-color: #800080">dataset_dataset.h5</span>
</pre>




```python
 def main():
    from l2hmc.experiment.pytorch.experiment import train_step
    from l2hmc.configs import CONF_DIR
    su3conf = Path(CONF_DIR).joinpath('su3-min-cpu.yaml')
    assert su3conf.is_file()
    # su3conf = Path('su3-min-cpu.yaml')
    with su3conf.open('r') as stream:
        conf = dict(yaml.safe_load(stream))

    log.info(conf)
    overrides = dict_to_list_of_overrides(conf)
    ptExpSU3 = get_experiment(overrides=[*overrides], build_networks=True)
    state = ptExpSU3.trainer.dynamics.random_state(6.0)
    assert isinstance(state.x, torch.Tensor)
    assert isinstance(state.beta, torch.Tensor)
    assert isinstance(ptExpSU3, Experiment)
    xhmc, history_hmc = evaluate(
        nsteps=100,
        exp=ptExpSU3,
        beta=state.beta,
        x=state.x,
        eps=0.1,
        nleapfrog=1,
        job_type='hmc',
        nlog=1,
        nprint=25,
        grab=True
    )
    xhmc = ptExpSU3.trainer.dynamics.unflatten(xhmc)
    log.info(f"checkSU(x_hmc): {g.checkSU(xhmc)}")
    plot_metrics(history_hmc.history, title='HMC', marker='.')
    # ptExpSU3.trainer.dynamics.init_weights(
    #     method='uniform',
    #     min=-1e-16,
    #     max=1e-16,
    #     bias=True,
    #     # xeps=0.001,
    #     # veps=0.001,
    # )
    xeval, history_eval = evaluate(
        nsteps=10,
        exp=ptExpSU3,
        beta=6.0,
        x=state.x,
        job_type='eval',
        nlog=1,
        nprint=25,
        grab=True,
    )
    xeval = ptExpSU3.trainer.dynamics.unflatten(xeval)
    log.info(f"checkSU(x_eval): {g.checkSU(xeval)}")
    plot_metrics(history_eval.history, title='Evaluate', marker='.')

    history = {}
    x = state.x
    for step in range(20):
        log.info(f'TRAIN STEP: {step}')
        x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
        if (step > 0 and step % 2 == 0):
            print_dict(metrics, grab=True)
        if (step > 0 and step % 1 == 0):
            for key, val in metrics.items():
                try:
                    history[key].append(val)
                except KeyError:
                    history[key] = [val]

    x = ptExpSU3.trainer.dynamics.unflatten(x)
    log.info(f"checkSU(x_train): {g.checkSU(x)}")
    plot_metrics(history, title='train', marker='.')
    #
    # for step in range(20):
    #     log.info(f"train step: {step}")
    #     x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
    #     if step % 5 == 0:
    #         print_dict(metrics, grab=True)

    return x, history
```


      Cell In[1], line 12
        _metrics = {k: v.mean() if isinstance(v, torch.Tensor), else v for k, v in metrics.items()}
                       ^
    SyntaxError: expected 'else' after 'if' expression




```python

```


```python

```
