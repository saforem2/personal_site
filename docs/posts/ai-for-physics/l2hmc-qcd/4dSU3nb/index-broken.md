# 🕸️ l2hmc-qcd Example: 4D SU(3)
Sam Foreman
2023-12-06

<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://iosevka-webfonts.github.io/iosevka/iosevka.css" rel="stylesheet">

- [`l2hmc-qcd`](#l2hmc-qcd)
- [Setup](#setup)
- [Load config + build Experiment](#load-config--build-experiment)
- [HMC](#hmc)
- [Evaluation](#evaluation)
- [Training](#training)

## `l2hmc-qcd`

This notebook contains a minimal working example for the 4D $SU(3)$
model

Uses `torch.complex128` by default

## Setup

``` python
import lovely_tensors as lt
lt.monkey_patch()
lt.set_config(color=False)
```

``` python
%load_ext autoreload
%autoreload 2
# automatically detect and reload local changes to modules
%matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
```

``` python
import os
from pathlib import Path
from typing import Optional

import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import l2hmc.group.su3.pytorch.group as g
from l2hmc.utils.rich import get_console
from l2hmc.common import grab_tensor, print_dict
from l2hmc.configs import dict_to_list_of_overrides, get_experiment
from l2hmc.experiment.pytorch.experiment import Experiment, evaluate  # noqa  # noqa
from l2hmc.utils.dist import setup_torch
from l2hmc.utils.plot_helpers import set_plot_style

os.environ['COLORTERM'] = 'truecolor;'
os.environ['MASTER_PORT'] = '5433'
# os.environ['MPLBACKEND'] = 'module://matplotlib-backend-kitty'
# plt.switch_backend('module://matplotlib-backend-kitty')
console = get_console()

_ = setup_torch(precision='float64', backend='DDP', seed=4351)

set_plot_style()

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
        console.log(f"Saving {key} to {outdir}")
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
    if len(element_shape) == 2:
        y = grab_tensor(torch.stack(metric))
        return plot_leapfrogs(y, ylabel=name)
    if len(element_shape) == 1:
        y = grab_tensor(torch.stack(metric))
        return plot_chains(y, ylabel=name, **kwargs)
    if len(element_shape) == 0:
        y = grab_tensor(torch.stack(metric))
        return plot_scalar(y, ylabel=name, **kwargs)
    raise ValueError
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
> <pre><code>[07/10/23 07:55:42][INFO][dist.py:226] - Caught MASTER_PORT:5433 from environment!
> [07/10/23 07:55:42][WARNING][dist.py:332] - Setting default dtype: float64
> [07/10/23 07:55:42][INFO][dist.py:338] - Global Rank: 0 / 0
> </code></pre>
>
> </div>

## Load config + build Experiment

``` python
from rich import print
set_plot_style()

from l2hmc.configs import CONF_DIR
su3conf = Path(f"{CONF_DIR}/su3-min.yaml")
with su3conf.open('r') as stream:
    conf = dict(yaml.safe_load(stream))
console.log(conf)
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:55:42] </span><span style="font-weight: bold">{</span>                                                                                                       
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'annealing_schedule'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'beta_final'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">6.0</span>, <span style="color: #008000; text-decoration-color: #008000">'beta_init'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">6.0</span><span style="font-weight: bold">}</span>,                                        
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'backend'</span>: <span style="color: #008000; text-decoration-color: #008000">'DDP'</span>,                                                                                   
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'conv'</span>: <span style="color: #008000; text-decoration-color: #008000">'none'</span>,                                                                                     
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'dynamics'</span>: <span style="font-weight: bold">{</span>                                                                                       
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'eps'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">0.001</span>,                                                                                   
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'eps_fixed'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                             
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'group'</span>: <span style="color: #008000; text-decoration-color: #008000">'SU3'</span>,                                                                                 
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'latvolume'</span>: <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">4</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">4</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">8</span><span style="font-weight: bold">]</span>,                                                                      
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'nchains'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">4</span>,                                                                                   
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'nleapfrog'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">4</span>,                                                                                 
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'use_separate_networks'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                 
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'use_split_xnets'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                       
> <span style="color: #696969; text-decoration-color: #696969">           </span>        <span style="color: #008000; text-decoration-color: #008000">'verbose'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>                                                                                 
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="font-weight: bold">}</span>,                                                                                                  
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'framework'</span>: <span style="color: #008000; text-decoration-color: #008000">'pytorch'</span>,                                                                             
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'init_aim'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                                  
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'init_wandb'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                                
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'learning_rate'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'clip_norm'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1.0</span>, <span style="color: #008000; text-decoration-color: #008000">'lr_init'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1e-05</span><span style="font-weight: bold">}</span>,                                              
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'loss'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'charge_weight'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">0.0</span>, <span style="color: #008000; text-decoration-color: #008000">'plaq_weight'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">0.0</span>, <span style="color: #008000; text-decoration-color: #008000">'rmse_weight'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1.0</span>, <span style="color: #008000; text-decoration-color: #008000">'use_mixed_loss'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">}</span>,    
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'net_weights'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'v'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'q'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1.0</span>, <span style="color: #008000; text-decoration-color: #008000">'s'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1.0</span>, <span style="color: #008000; text-decoration-color: #008000">'t'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1.0</span><span style="font-weight: bold">}</span>, <span style="color: #008000; text-decoration-color: #008000">'x'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'q'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">0.0</span>, <span style="color: #008000; text-decoration-color: #008000">'s'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">0.0</span>, <span style="color: #008000; text-decoration-color: #008000">'t'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">0.0</span><span style="font-weight: bold">}}</span>,          
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'network'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'activation_fn'</span>: <span style="color: #008000; text-decoration-color: #008000">'tanh'</span>, <span style="color: #008000; text-decoration-color: #008000">'dropout_prob'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">0.0</span>, <span style="color: #008000; text-decoration-color: #008000">'units'</span>: <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">1</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'use_batch_norm'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">}</span>,   
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'restore'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                                   
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'save'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                                      
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'steps'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'log'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1</span>, <span style="color: #008000; text-decoration-color: #008000">'nepoch'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">10</span>, <span style="color: #008000; text-decoration-color: #008000">'nera'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1</span>, <span style="color: #008000; text-decoration-color: #008000">'print'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">1</span>, <span style="color: #008000; text-decoration-color: #008000">'test'</span>: <span style="color: #2094f3; text-decoration-color: #2094f3">50</span><span style="font-weight: bold">}</span>,                               
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'use_tb'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,                                                                                    
> <span style="color: #696969; text-decoration-color: #696969">           </span>    <span style="color: #008000; text-decoration-color: #008000">'use_wandb'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>                                                                                  
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="font-weight: bold">}</span>                                                                                                       
> </pre>
>
> </div>

``` python
overrides = dict_to_list_of_overrides(conf)
ptExpSU3 = get_experiment(overrides=[*overrides], build_networks=True)
state = ptExpSU3.trainer.dynamics.random_state(6.0)
console.log(f"checkSU(state.x): {g.checkSU(state.x)}")
assert isinstance(state.x, torch.Tensor)
assert isinstance(state.beta, torch.Tensor)
assert isinstance(ptExpSU3, Experiment)
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
>     [38;2;105;105;105m[07/10/23 07:55:42][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mdist.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m226[0m[2;38;2;144;144;144m][0m - Caught MASTER_PORT:[38;2;32;148;243m5433[0m from environment!
>     [38;2;105;105;105m[07/10/23 07:55:42][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mdist.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m226[0m[2;38;2;144;144;144m][0m - Caught MASTER_PORT:[38;2;32;148;243m5433[0m from environment!
>     [38;2;105;105;105m[07/10/23 07:55:42][0m[33m[WARNING][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mtrainer.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m436[0m[2;38;2;144;144;144m][0m - Using `torch.optim.Adam` optimizer
>     [38;2;105;105;105m[07/10/23 07:55:42][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mtrainer.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m284[0m[2;38;2;144;144;144m][0m - num_params in model: [38;2;32;148;243m401420[0m
>     [38;2;105;105;105m[07/10/23 07:55:42][0m[33m[WARNING][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mtrainer.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m250[0m[2;38;2;144;144;144m][0m - logging with freq [38;2;32;148;243m1[0m for wandb.watch
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">checkSU</span><span style="font-weight: bold">(</span>state.x<span style="font-weight: bold">)</span>: <span style="font-weight: bold">(</span>tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">2.174e-07</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.981e-07</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">3</span><span style="color: #2094f3; text-decoration-color: #2094f3">.044e-07</span> <span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">7</span><span style="color: #2094f3; text-decoration-color: #2094f3">.432e-08</span> grad SqrtBackward0    
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">2.174e-07</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.981e-07</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.107e-07</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">2.914e-07</span><span style="font-weight: bold">]</span>, tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4.942e-06</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.187e-05</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">8</span><span style="color: #2094f3; text-decoration-color: #2094f3">.282e-06</span>        
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">2</span><span style="color: #2094f3; text-decoration-color: #2094f3">.848e-06</span> grad SqrtBackward0 <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4.942e-06</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.187e-05</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">8.531e-06</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">7.784e-06</span><span style="font-weight: bold">])</span>                            
> </pre>
>
> </div>

``` python
state.x.real.plt()
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
> ![svg](output_8_0.svg)
>
> </div>

``` python
state.x.imag.plt()
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
> ![svg](output_9_0.svg)
>
> </div>

## HMC

``` python
xhmc, history_hmc = evaluate(
    nsteps=50,
    exp=ptExpSU3,
    beta=state.beta,
    x=state.x,
    eps=0.1,
    nleapfrog=1,
    job_type='hmc',
    nlog=2,
    nprint=5,
    grab=True
)
xhmc = ptExpSU3.trainer.dynamics.unflatten(xhmc)
console.log(f"checkSU(x_hmc): {g.checkSU(xhmc)}")
plot_metrics(history_hmc, title='HMC', marker='.')
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m115[0m[2;38;2;144;144;144m][0m - Running [38;2;32;148;243m50[0m steps of hmc at [38;2;125;134;151mbeta[0m=[38;2;32;148;243m6[0m[38;2;32;148;243m.0000[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m0[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m1[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m2[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m3[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m4[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m5[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-1455.17436446[0m [38;2;32;148;243m-1652.1514185[0m  [38;2;32;148;243m-1517.90987068[0m [38;2;32;148;243m-1455.94178323[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1459.25165577[0m [38;2;32;148;243m-1657.51842424[0m [38;2;32;148;243m-1524.14091372[0m [38;2;32;148;243m-1460.98822723[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-1455.17436446[0m [38;2;32;148;243m-1652.1514185[0m  [38;2;32;148;243m-1517.90987068[0m [38;2;32;148;243m-1455.94178323[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1459.25165577[0m [38;2;32;148;243m-1657.51842424[0m [38;2;32;148;243m-1524.14091372[0m [38;2;32;148;243m-1460.98822723[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.07198033[0m [38;2;32;148;243m0.07955383[0m [38;2;32;148;243m0.07632937[0m [38;2;32;148;243m0.079714[0m  [1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-4.59449208e-03[0m  [38;2;32;148;243m6.97601890e-05[0m [38;2;32;148;243m-1.36834282e-03[0m  [38;2;32;148;243m2.93498261e-03[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.13406958[0m  [38;2;32;148;243m0.00203564[0m [38;2;32;148;243m-0.03992893[0m  [38;2;32;148;243m0.08564426[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.01024449[0m [38;2;32;148;243m0.02805506[0m [38;2;32;148;243m0.01725271[0m [38;2;32;148;243m0.01645253[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00035107[0m [38;2;32;148;243m0.00096143[0m [38;2;32;148;243m0.00059124[0m [38;2;32;148;243m0.00056382[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.008833834588589714[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m6[0m
>     [38;2;105;105;105m[07/10/23 07:56:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m7[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m8[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m9[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m10[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-2774.80624156[0m [38;2;32;148;243m-2851.59293222[0m [38;2;32;148;243m-2661.49055236[0m [38;2;32;148;243m-2425.15656787[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2779.73017905[0m [38;2;32;148;243m-2856.7767309[0m  [38;2;32;148;243m-2666.12510868[0m [38;2;32;148;243m-2429.23105695[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-2774.80624156[0m [38;2;32;148;243m-2851.59293222[0m [38;2;32;148;243m-2661.49055236[0m [38;2;32;148;243m-2425.15656787[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2779.73017905[0m [38;2;32;148;243m-2856.7767309[0m  [38;2;32;148;243m-2666.12510868[0m [38;2;32;148;243m-2429.23105695[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.13573134[0m [38;2;32;148;243m0.14720937[0m [38;2;32;148;243m0.14114639[0m [38;2;32;148;243m0.14401027[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00396796[0m [38;2;32;148;243m-0.00146382[0m  [38;2;32;148;243m0.0040015[0m   [38;2;32;148;243m0.00239218[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.11578697[0m [38;2;32;148;243m-0.04271513[0m  [38;2;32;148;243m0.11676574[0m  [38;2;32;148;243m0.06980506[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.01591121[0m [38;2;32;148;243m0.07023641[0m [38;2;32;148;243m0.03053575[0m [38;2;32;148;243m0.01694004[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00054527[0m [38;2;32;148;243m0.00240696[0m [38;2;32;148;243m0.00104644[0m [38;2;32;148;243m0.00058053[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.008887014270663943[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m11[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m12[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m13[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m14[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m15[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-3589.39696557[0m [38;2;32;148;243m-3672.75566478[0m [38;2;32;148;243m-3699.9655895[0m  [38;2;32;148;243m-3758.19489614[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3591.62366753[0m [38;2;32;148;243m-3676.34262087[0m [38;2;32;148;243m-3704.28601825[0m [38;2;32;148;243m-3763.08583919[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-3589.39696557[0m [38;2;32;148;243m-3672.75566478[0m [38;2;32;148;243m-3699.9655895[0m  [38;2;32;148;243m-3758.19489614[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3591.62366753[0m [38;2;32;148;243m-3676.34262087[0m [38;2;32;148;243m-3704.28601825[0m [38;2;32;148;243m-3763.08583919[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.19442852[0m [38;2;32;148;243m0.20120752[0m [38;2;32;148;243m0.19236084[0m [38;2;32;148;243m0.19703231[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00550476[0m [38;2;32;148;243m-0.00195778[0m  [38;2;32;148;243m0.00094388[0m  [38;2;32;148;243m0.00348485[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.16063176[0m [38;2;32;148;243m-0.05712905[0m  [38;2;32;148;243m0.027543[0m    [38;2;32;148;243m0.10168963[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00838519[0m [38;2;32;148;243m0.00298843[0m [38;2;32;148;243m0.01475424[0m [38;2;32;148;243m0.05713639[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00028736[0m [38;2;32;148;243m0.00010241[0m [38;2;32;148;243m0.00050562[0m [38;2;32;148;243m0.00195803[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.00887361562615175[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m16[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m17[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m18[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m19[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m20[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-4151.54650574[0m [38;2;32;148;243m-4494.28752887[0m [38;2;32;148;243m-4160.10384616[0m [38;2;32;148;243m-4408.71292573[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-4156.03583628[0m [38;2;32;148;243m-4496.77600268[0m [38;2;32;148;243m-4162.17053388[0m [38;2;32;148;243m-4412.58171459[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-4151.54650574[0m [38;2;32;148;243m-4494.28752887[0m [38;2;32;148;243m-4160.10384616[0m [38;2;32;148;243m-4408.71292573[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-4156.03583628[0m [38;2;32;148;243m-4496.77600268[0m [38;2;32;148;243m-4162.17053388[0m [38;2;32;148;243m-4412.58171459[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.22897834[0m [38;2;32;148;243m0.23997054[0m [38;2;32;148;243m0.22982328[0m [38;2;32;148;243m0.23599929[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00693299[0m [38;2;32;148;243m-0.00347184[0m [38;2;32;148;243m-0.00120909[0m  [38;2;32;148;243m0.00308169[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.20230825[0m [38;2;32;148;243m-0.10131016[0m [38;2;32;148;243m-0.03528176[0m  [38;2;32;148;243m0.08992527[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.04257003[0m [38;2;32;148;243m0.05104837[0m [38;2;32;148;243m0.00619903[0m [38;2;32;148;243m0.00752605[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00145885[0m [38;2;32;148;243m0.0017494[0m  [38;2;32;148;243m0.00021244[0m [38;2;32;148;243m0.00025791[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.008967695306967319[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m21[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m22[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m23[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m24[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m25[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-4814.63799281[0m [38;2;32;148;243m-4919.94688869[0m [38;2;32;148;243m-4788.79616489[0m [38;2;32;148;243m-5002.9215054[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-4817.09579284[0m [38;2;32;148;243m-4921.24269566[0m [38;2;32;148;243m-4790.50072208[0m [38;2;32;148;243m-5005.14858853[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-4814.63799281[0m [38;2;32;148;243m-4919.94688869[0m [38;2;32;148;243m-4788.79616489[0m [38;2;32;148;243m-5002.9215054[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-4817.09579284[0m [38;2;32;148;243m-4921.24269566[0m [38;2;32;148;243m-4790.50072208[0m [38;2;32;148;243m-5005.14858853[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.26288089[0m [38;2;32;148;243m0.27682883[0m [38;2;32;148;243m0.26268439[0m [38;2;32;148;243m0.26544038[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00063547[0m [38;2;32;148;243m-0.00425824[0m [38;2;32;148;243m-0.00254524[0m  [38;2;32;148;243m0.00551469[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01854337[0m [38;2;32;148;243m-0.12425771[0m [38;2;32;148;243m-0.07427149[0m  [38;2;32;148;243m0.16092137[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00165815[0m [38;2;32;148;243m0.03643215[0m [38;2;32;148;243m0.00549913[0m [38;2;32;148;243m0.03666496[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m5.68238145e-05[0m [38;2;32;148;243m1.24851003e-03[0m [38;2;32;148;243m1.88452222e-04[0m [38;2;32;148;243m1.25648827e-03[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.008991637304362567[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m26[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m27[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m28[0m
>     [38;2;105;105;105m[07/10/23 07:56:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m29[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m30[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-5291.93327564[0m [38;2;32;148;243m-5277.74230381[0m [38;2;32;148;243m-5250.76164114[0m [38;2;32;148;243m-5396.46879447[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-5292.63668436[0m [38;2;32;148;243m-5278.24552901[0m [38;2;32;148;243m-5252.56634155[0m [38;2;32;148;243m-5396.60232376[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-5291.93327564[0m [38;2;32;148;243m-5277.74230381[0m [38;2;32;148;243m-5250.76164114[0m [38;2;32;148;243m-5396.46879447[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-5292.63668436[0m [38;2;32;148;243m-5278.24552901[0m [38;2;32;148;243m-5252.56634155[0m [38;2;32;148;243m-5396.60232376[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.2875013[0m  [38;2;32;148;243m0.29490873[0m [38;2;32;148;243m0.28911575[0m [38;2;32;148;243m0.29292899[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00214532[0m [38;2;32;148;243m-0.00195831[0m  [38;2;32;148;243m0.00125583[0m  [38;2;32;148;243m0.00433508[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.06260166[0m [38;2;32;148;243m-0.05714455[0m  [38;2;32;148;243m0.03664583[0m  [38;2;32;148;243m0.1264999[0m [1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00996321[0m [38;2;32;148;243m0.01477696[0m [38;2;32;148;243m0.00678131[0m [38;2;32;148;243m0.0104012[0m [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00034143[0m [38;2;32;148;243m0.0005064[0m  [38;2;32;148;243m0.00023239[0m [38;2;32;148;243m0.00035644[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.009008451409977573[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m31[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m32[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m33[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m34[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m35[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-5677.87109891[0m [38;2;32;148;243m-5799.43835247[0m [38;2;32;148;243m-5783.77900075[0m [38;2;32;148;243m-5966.2188758[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-5679.3788349[0m  [38;2;32;148;243m-5801.05951826[0m [38;2;32;148;243m-5785.90693956[0m [38;2;32;148;243m-5967.61796828[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-5677.87109891[0m [38;2;32;148;243m-5799.43835247[0m [38;2;32;148;243m-5783.77900075[0m [38;2;32;148;243m-5966.2188758[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-5679.3788349[0m  [38;2;32;148;243m-5801.05951826[0m [38;2;32;148;243m-5785.90693956[0m [38;2;32;148;243m-5967.61796828[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.30658624[0m [38;2;32;148;243m0.31260177[0m [38;2;32;148;243m0.31303925[0m [38;2;32;148;243m0.31611992[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00207013[0m  [38;2;32;148;243m0.00102852[0m  [38;2;32;148;243m0.00099099[0m  [38;2;32;148;243m0.00199912[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.06040742[0m  [38;2;32;148;243m0.03001266[0m  [38;2;32;148;243m0.02891754[0m  [38;2;32;148;243m0.05833535[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.02325924[0m [38;2;32;148;243m0.04047365[0m [38;2;32;148;243m0.0205901[0m  [38;2;32;148;243m0.02235072[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00079708[0m [38;2;32;148;243m0.00138701[0m [38;2;32;148;243m0.00070561[0m [38;2;32;148;243m0.00076595[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.008902220811433029[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m36[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m37[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m38[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m39[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m40[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-6009.28484422[0m [38;2;32;148;243m-6075.42435918[0m [38;2;32;148;243m-6195.95087466[0m [38;2;32;148;243m-6048.18856638[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6009.38784106[0m [38;2;32;148;243m-6076.5575995[0m  [38;2;32;148;243m-6197.94839446[0m [38;2;32;148;243m-6048.75106049[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-6009.28484422[0m [38;2;32;148;243m-6075.42435918[0m [38;2;32;148;243m-6195.95087466[0m [38;2;32;148;243m-6048.18856638[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6009.38784106[0m [38;2;32;148;243m-6076.5575995[0m  [38;2;32;148;243m-6197.94839446[0m [38;2;32;148;243m-6048.75106049[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.32328311[0m [38;2;32;148;243m0.33136092[0m [38;2;32;148;243m0.33288434[0m [38;2;32;148;243m0.33155877[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00087745[0m [38;2;32;148;243m-0.00030477[0m  [38;2;32;148;243m0.00157488[0m  [38;2;32;148;243m0.00408504[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.02560443[0m [38;2;32;148;243m-0.00889343[0m  [38;2;32;148;243m0.04595583[0m  [38;2;32;148;243m0.11920361[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.05894986[0m [38;2;32;148;243m0.05110851[0m [38;2;32;148;243m0.00843622[0m [38;2;32;148;243m0.0193832[0m [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00202018[0m [38;2;32;148;243m0.00175146[0m [38;2;32;148;243m0.0002891[0m  [38;2;32;148;243m0.00066425[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.008961126836871364[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m41[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m42[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m43[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m44[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m45[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-6172.38358228[0m [38;2;32;148;243m-6473.29456144[0m [38;2;32;148;243m-6406.68919584[0m [38;2;32;148;243m-6232.45720644[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6172.72022528[0m [38;2;32;148;243m-6471.1347579[0m  [38;2;32;148;243m-6407.01878873[0m [38;2;32;148;243m-6235.0900963[0m [1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-6172.38358228[0m [38;2;32;148;243m-6473.29456144[0m [38;2;32;148;243m-6406.68919584[0m [38;2;32;148;243m-6232.45720644[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6172.72022528[0m [38;2;32;148;243m-6471.1347579[0m  [38;2;32;148;243m-6407.01878873[0m [38;2;32;148;243m-6235.0900963[0m [1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m2[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m.         [38;2;32;148;243m0.11534778[0m [38;2;32;148;243m1[0m.         [38;2;32;148;243m1[0m.        [1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.33577826[0m [38;2;32;148;243m0.34108362[0m [38;2;32;148;243m0.34569419[0m [38;2;32;148;243m0.34125761[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m1.19948603e-03[0m [38;2;32;148;243m-1.70704191e-05[0m  [38;2;32;148;243m1.29880814e-03[0m  [38;2;32;148;243m1.15764374e-03[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0350016[0m  [38;2;32;148;243m-0.00049812[0m  [38;2;32;148;243m0.03789987[0m  [38;2;32;148;243m0.03378062[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.03684475[0m [38;2;32;148;243m0[0m.         [38;2;32;148;243m0.02928525[0m [38;2;32;148;243m0.01172397[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00126265[0m [38;2;32;148;243m0[0m.         [38;2;32;148;243m0.00100359[0m [38;2;32;148;243m0.00040177[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-0.006980147559740732[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m46[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m47[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m48[0m
>     [38;2;105;105;105m[07/10/23 07:56:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m49[0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:56:33] </span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">checkSU</span><span style="font-weight: bold">(</span>x_hmc<span style="font-weight: bold">)</span>: <span style="font-weight: bold">(</span>tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">3.429e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.484e-16</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">3</span><span style="color: #2094f3; text-decoration-color: #2094f3">.453e-16</span> <span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">2</span><span style="color: #2094f3; text-decoration-color: #2094f3">.351e-18</span> <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">3.442e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.484e-16</span>,  
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #2094f3; text-decoration-color: #2094f3">3.429e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.458e-16</span><span style="font-weight: bold">]</span>, tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">1.072e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.171e-15</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">1</span><span style="color: #2094f3; text-decoration-color: #2094f3">.136e-15</span> <span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="color: #2094f3; text-decoration-color: #2094f3">.386e-17</span> <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">1.143e-15</span>,       
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #2094f3; text-decoration-color: #2094f3">1.157e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.072e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.171e-15</span><span style="font-weight: bold">])</span>                                                                       
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving energy to plots-4dSU3/HMC                                                                        
> </pre>
>
> ![svg](output_11_3.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving logprob to plots-4dSU3/HMC                                                                       
> </pre>
>
> ![svg](output_11_5.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving logdet to plots-4dSU3/HMC                                                                        
> </pre>
>
> ![svg](output_11_7.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving acc to plots-4dSU3/HMC                                                                           
> </pre>
>
> ![svg](output_11_9.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sumlogdet to plots-4dSU3/HMC                                                                     
> </pre>
>
> ![svg](output_11_11.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:56:34] </span>Saving acc_mask to plots-4dSU3/HMC                                                                      
> </pre>
>
> ![svg](output_11_13.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving plaqs to plots-4dSU3/HMC                                                                         
> </pre>
>
> ![svg](output_11_15.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sinQ to plots-4dSU3/HMC                                                                          
> </pre>
>
> ![svg](output_11_17.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving intQ to plots-4dSU3/HMC                                                                          
> </pre>
>
> ![svg](output_11_19.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:56:35] </span>Saving dQint to plots-4dSU3/HMC                                                                         
> </pre>
>
> ![svg](output_11_21.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving dQsin to plots-4dSU3/HMC                                                                         
> </pre>
>
> ![svg](output_11_23.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving loss to plots-4dSU3/HMC                                                                          
> </pre>
>
> ![svg](output_11_25.svg)
>
> </div>

## Evaluation

``` python
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
    nlog=2,
    nprint=5,
    grab=True,
)
xeval = ptExpSU3.trainer.dynamics.unflatten(xeval)
console.log(f"checkSU(x_eval): {g.checkSU(xeval)}")
plot_metrics(history_eval, title='Evaluate', marker='.')
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
>     [38;2;105;105;105m[07/10/23 07:56:45][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m115[0m[2;38;2;144;144;144m][0m - Running [38;2;32;148;243m50[0m steps of eval at [38;2;125;134;151mbeta[0m=[38;2;32;148;243m6[0m[38;2;32;148;243m.0000[0m
>     [38;2;105;105;105m[07/10/23 07:56:45][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m0[0m
>     [38;2;105;105;105m[07/10/23 07:56:46][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m1[0m
>     [38;2;105;105;105m[07/10/23 07:56:46][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m2[0m
>     [38;2;105;105;105m[07/10/23 07:56:46][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m3[0m
>     [38;2;105;105;105m[07/10/23 07:56:47][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m4[0m
>     [38;2;105;105;105m[07/10/23 07:56:47][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m5[0m
>     [38;2;105;105;105m[07/10/23 07:56:48][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m135.13661105[0m [38;2;32;148;243m-33.63307096[0m  [38;2;32;148;243m67.14737865[0m [38;2;32;148;243m-44.94287792[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.12019559[0m [38;2;32;148;243m-33.67474968[0m  [38;2;32;148;243m67.13262028[0m [38;2;32;148;243m-45.06121746[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.11640715[0m [38;2;32;148;243m-33.7029591[0m   [38;2;32;148;243m67.13542667[0m [38;2;32;148;243m-45.15853882[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.12597508[0m [38;2;32;148;243m-33.71582603[0m  [38;2;32;148;243m67.15528762[0m [38;2;32;148;243m-45.23443416[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.15009837[0m [38;2;32;148;243m-33.71282679[0m  [38;2;32;148;243m67.19223232[0m [38;2;32;148;243m-45.28949862[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.24409668[0m [38;2;32;148;243m-33.55745988[0m  [38;2;32;148;243m67.28977981[0m [38;2;32;148;243m-45.05889668[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.35293374[0m [38;2;32;148;243m-33.38888138[0m  [38;2;32;148;243m67.40388327[0m [38;2;32;148;243m-44.8080029[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m135.47661625[0m [38;2;32;148;243m-33.20710601[0m  [38;2;32;148;243m67.53407019[0m [38;2;32;148;243m-44.53600603[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.61403391[0m [38;2;32;148;243m-33.01043184[0m  [38;2;32;148;243m67.68099845[0m [38;2;32;148;243m-44.24220592[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m135.13661105[0m [38;2;32;148;243m-33.63307096[0m  [38;2;32;148;243m67.14737865[0m [38;2;32;148;243m-44.94287792[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.18594798[0m [38;2;32;148;243m-33.64895353[0m  [38;2;32;148;243m67.23191696[0m [38;2;32;148;243m-44.94255748[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.25009615[0m [38;2;32;148;243m-33.65006019[0m  [38;2;32;148;243m67.33406416[0m [38;2;32;148;243m-44.92096816[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.32862172[0m [38;2;32;148;243m-33.63742038[0m  [38;2;32;148;243m67.45280265[0m [38;2;32;148;243m-44.87768554[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.42163919[0m [38;2;32;148;243m-33.61092114[0m  [38;2;32;148;243m67.58849517[0m [38;2;32;148;243m-44.81350169[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.44700618[0m [38;2;32;148;243m-33.47933478[0m  [38;2;32;148;243m67.58664812[0m [38;2;32;148;243m-44.70191131[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.48678158[0m [38;2;32;148;243m-33.33680972[0m  [38;2;32;148;243m67.60182722[0m [38;2;32;148;243m-44.56979511[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m135.54116027[0m [38;2;32;148;243m-33.18273746[0m  [38;2;32;148;243m67.634192[0m   [38;2;32;148;243m-44.4166053[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m135.60984902[0m [38;2;32;148;243m-33.01258723[0m  [38;2;32;148;243m67.68334133[0m [38;2;32;148;243m-44.24196715[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6.57523818e-02[0m [38;2;32;148;243m-2.57961502e-02[0m [38;2;32;148;243m-9.92966725e-02[0m [38;2;32;148;243m-1.18659983e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.33688996e-01[0m [38;2;32;148;243m-5.28989121e-02[0m [38;2;32;148;243m-1.98637494e-01[0m [38;2;32;148;243m-2.37570657e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.02646646e-01[0m [38;2;32;148;243m-7.84056547e-02[0m [38;2;32;148;243m-2.97515030e-01[0m [38;2;32;148;243m-3.56748619e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.71540826e-01[0m [38;2;32;148;243m-1.01905646e-01[0m [38;2;32;148;243m-3.96262845e-01[0m [38;2;32;148;243m-4.75996937e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.02909501e-01[0m [38;2;32;148;243m-7.81251045e-02[0m [38;2;32;148;243m-2.96868313e-01[0m [38;2;32;148;243m-3.56985365e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.33847838e-01[0m [38;2;32;148;243m-5.20716617e-02[0m [38;2;32;148;243m-1.97943952e-01[0m [38;2;32;148;243m-2.38207787e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6.45440187e-02[0m [38;2;32;148;243m-2.43685486e-02[0m [38;2;32;148;243m-1.00121811e-01[0m [38;2;32;148;243m-1.19400726e-01[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m4.18489492e-03[0m  [38;2;32;148;243m2.15538787e-03[0m [38;2;32;148;243m-2.34288059e-03[0m [38;2;32;148;243m-2.38771971e-04[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06575238[0m [38;2;32;148;243m-0.02579615[0m [38;2;32;148;243m-0.09929667[0m [38;2;32;148;243m-0.11865998[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.133689[0m   [38;2;32;148;243m-0.05289891[0m [38;2;32;148;243m-0.19863749[0m [38;2;32;148;243m-0.23757066[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20264665[0m [38;2;32;148;243m-0.07840565[0m [38;2;32;148;243m-0.29751503[0m [38;2;32;148;243m-0.35674862[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27154083[0m [38;2;32;148;243m-0.10190565[0m [38;2;32;148;243m-0.39626285[0m [38;2;32;148;243m-0.47599694[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.06863133[0m [38;2;32;148;243m0.02378054[0m [38;2;32;148;243m0.09939453[0m [38;2;32;148;243m0.11901157[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.13769299[0m [38;2;32;148;243m0.04983398[0m [38;2;32;148;243m0.19831889[0m [38;2;32;148;243m0.23778915[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.20699681[0m [38;2;32;148;243m0.0775371[0m  [38;2;32;148;243m0.29614103[0m [38;2;32;148;243m0.35659621[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.27572572[0m [38;2;32;148;243m0.10406103[0m [38;2;32;148;243m0.39391996[0m [38;2;32;148;243m0.47575816[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6.57523818e-02[0m [38;2;32;148;243m-2.57961502e-02[0m [38;2;32;148;243m-9.92966725e-02[0m [38;2;32;148;243m-1.18659983e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.33688996e-01[0m [38;2;32;148;243m-5.28989121e-02[0m [38;2;32;148;243m-1.98637494e-01[0m [38;2;32;148;243m-2.37570657e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.02646646e-01[0m [38;2;32;148;243m-7.84056547e-02[0m [38;2;32;148;243m-2.97515030e-01[0m [38;2;32;148;243m-3.56748619e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.71540826e-01[0m [38;2;32;148;243m-1.01905646e-01[0m [38;2;32;148;243m-3.96262845e-01[0m [38;2;32;148;243m-4.75996937e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.02909501e-01[0m [38;2;32;148;243m-7.81251045e-02[0m [38;2;32;148;243m-2.96868313e-01[0m [38;2;32;148;243m-3.56985365e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.33847838e-01[0m [38;2;32;148;243m-5.20716617e-02[0m [38;2;32;148;243m-1.97943952e-01[0m [38;2;32;148;243m-2.38207787e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-6.45440187e-02[0m [38;2;32;148;243m-2.43685486e-02[0m [38;2;32;148;243m-1.00121811e-01[0m [38;2;32;148;243m-1.19400726e-01[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m4.18489492e-03[0m  [38;2;32;148;243m2.15538787e-03[0m [38;2;32;148;243m-2.34288059e-03[0m [38;2;32;148;243m-2.38771971e-04[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.6229818[0m  [38;2;32;148;243m0.53768428[0m [38;2;32;148;243m0.58510575[0m [38;2;32;148;243m0.49613324[0m[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.         [38;2;32;148;243m-0.00234288[0m [38;2;32;148;243m-0[0m.        [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00451561[0m [38;2;32;148;243m-0.00117244[0m [38;2;32;148;243m-0.0024079[0m  [38;2;32;148;243m-0.00328763[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00482457[0m [38;2;32;148;243m-0.001806[0m   [38;2;32;148;243m-0.00311093[0m  [38;2;32;148;243m0.00384917[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.14078351[0m [38;2;32;148;243m-0.05270007[0m [38;2;32;148;243m-0.09077836[0m  [38;2;32;148;243m0.11232063[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0.00423973[0m [38;2;32;148;243m0[0m.        [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0.00014529[0m [38;2;32;148;243m0[0m.        [1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-3.176207358041437e-05[0m
>     [38;2;105;105;105m[07/10/23 07:56:48][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m6[0m
>     [38;2;105;105;105m[07/10/23 07:56:48][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m7[0m
>     [38;2;105;105;105m[07/10/23 07:56:48][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m8[0m
>     [38;2;105;105;105m[07/10/23 07:56:49][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m9[0m
>     [38;2;105;105;105m[07/10/23 07:56:49][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m10[0m
>     [38;2;105;105;105m[07/10/23 07:56:49][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-9.56401221e+01[0m  [38;2;32;148;243m3.01519478e-01[0m  [38;2;32;148;243m1.26397106e+02[0m [38;2;32;148;243m-8.75141370e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.56823640e+01[0m  [38;2;32;148;243m1.45612359e-01[0m  [38;2;32;148;243m1.26393890e+02[0m [38;2;32;148;243m-8.73955597e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.57109596e+01[0m  [38;2;32;148;243m3.95985324e-03[0m  [38;2;32;148;243m1.26407219e+02[0m [38;2;32;148;243m-8.72563460e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.57253138e+01[0m [38;2;32;148;243m-1.23224050e-01[0m  [38;2;32;148;243m1.26437241e+02[0m [38;2;32;148;243m-8.70966271e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.57257501e+01[0m [38;2;32;148;243m-2.36135287e-01[0m  [38;2;32;148;243m1.26484182e+02[0m [38;2;32;148;243m-8.69163051e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.56167926e+01[0m [38;2;32;148;243m-1.52561289e-01[0m  [38;2;32;148;243m1.26698581e+02[0m [38;2;32;148;243m-8.69189539e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.54921713e+01[0m [38;2;32;148;243m-5.40582444e-02[0m  [38;2;32;148;243m1.26929478e+02[0m [38;2;32;148;243m-8.69016305e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.53528175e+01[0m  [38;2;32;148;243m5.86356052e-02[0m  [38;2;32;148;243m1.27176913e+02[0m [38;2;32;148;243m-8.68641859e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.52006482e+01[0m  [38;2;32;148;243m1.85718837e-01[0m  [38;2;32;148;243m1.27443679e+02[0m [38;2;32;148;243m-8.68062122e+01[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-9.56401221e+01[0m  [38;2;32;148;243m3.01519478e-01[0m  [38;2;32;148;243m1.26397106e+02[0m [38;2;32;148;243m-8.75141370e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.56248528e+01[0m  [38;2;32;148;243m1.73238470e-01[0m  [38;2;32;148;243m1.26484219e+02[0m [38;2;32;148;243m-8.72781272e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.55952941e+01[0m  [38;2;32;148;243m5.89521523e-02[0m  [38;2;32;148;243m1.26590094e+02[0m [38;2;32;148;243m-8.70213932e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.55488299e+01[0m [38;2;32;148;243m-4.13439677e-02[0m  [38;2;32;148;243m1.26714724e+02[0m [38;2;32;148;243m-8.67439896e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.54873081e+01[0m [38;2;32;148;243m-1.27574436e-01[0m  [38;2;32;148;243m1.26856942e+02[0m [38;2;32;148;243m-8.64458164e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.54413096e+01[0m [38;2;32;148;243m-6.93640504e-02[0m  [38;2;32;148;243m1.26976549e+02[0m [38;2;32;148;243m-8.65662029e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.53811335e+01[0m  [38;2;32;148;243m5.41904615e-03[0m  [38;2;32;148;243m1.27112514e+02[0m [38;2;32;148;243m-8.66661364e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.53069725e+01[0m  [38;2;32;148;243m9.42762298e-02[0m  [38;2;32;148;243m1.27264945e+02[0m [38;2;32;148;243m-8.67460576e+01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.52189787e+01[0m  [38;2;32;148;243m1.94748598e-01[0m  [38;2;32;148;243m1.27435612e+02[0m [38;2;32;148;243m-8.68056112e+01[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05751118[0m [38;2;32;148;243m-0.02762611[0m [38;2;32;148;243m-0.09032856[0m [38;2;32;148;243m-0.11743246[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1156655[0m  [38;2;32;148;243m-0.0549923[0m  [38;2;32;148;243m-0.18287469[0m [38;2;32;148;243m-0.23495288[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17648393[0m [38;2;32;148;243m-0.08188008[0m [38;2;32;148;243m-0.27748311[0m [38;2;32;148;243m-0.35263751[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.23844197[0m [38;2;32;148;243m-0.10856085[0m [38;2;32;148;243m-0.37275912[0m [38;2;32;148;243m-0.47048872[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17548294[0m [38;2;32;148;243m-0.08319724[0m [38;2;32;148;243m-0.27796796[0m [38;2;32;148;243m-0.35275104[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1110378[0m  [38;2;32;148;243m-0.05947729[0m [38;2;32;148;243m-0.18303625[0m [38;2;32;148;243m-0.23549409[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.045845[0m   [38;2;32;148;243m-0.03564062[0m [38;2;32;148;243m-0.0880325[0m  [38;2;32;148;243m-0.11812833[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01833044[0m [38;2;32;148;243m-0.00902976[0m  [38;2;32;148;243m0.00806685[0m [38;2;32;148;243m-0.00060101[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05751118[0m [38;2;32;148;243m-0.02762611[0m [38;2;32;148;243m-0.09032856[0m [38;2;32;148;243m-0.11743246[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1156655[0m  [38;2;32;148;243m-0.0549923[0m  [38;2;32;148;243m-0.18287469[0m [38;2;32;148;243m-0.23495288[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17648393[0m [38;2;32;148;243m-0.08188008[0m [38;2;32;148;243m-0.27748311[0m [38;2;32;148;243m-0.35263751[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.23844197[0m [38;2;32;148;243m-0.10856085[0m [38;2;32;148;243m-0.37275912[0m [38;2;32;148;243m-0.47048872[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.06295902[0m [38;2;32;148;243m0.02536361[0m [38;2;32;148;243m0.09479116[0m [38;2;32;148;243m0.11773768[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.12740417[0m [38;2;32;148;243m0.04908356[0m [38;2;32;148;243m0.18972287[0m [38;2;32;148;243m0.23499463[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.19259697[0m [38;2;32;148;243m0.07292023[0m [38;2;32;148;243m0.28472662[0m [38;2;32;148;243m0.35236039[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.25677241[0m [38;2;32;148;243m0.09953109[0m [38;2;32;148;243m0.38082597[0m [38;2;32;148;243m0.46988771[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05751118[0m [38;2;32;148;243m-0.02762611[0m [38;2;32;148;243m-0.09032856[0m [38;2;32;148;243m-0.11743246[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1156655[0m  [38;2;32;148;243m-0.0549923[0m  [38;2;32;148;243m-0.18287469[0m [38;2;32;148;243m-0.23495288[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17648393[0m [38;2;32;148;243m-0.08188008[0m [38;2;32;148;243m-0.27748311[0m [38;2;32;148;243m-0.35263751[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.23844197[0m [38;2;32;148;243m-0.10856085[0m [38;2;32;148;243m-0.37275912[0m [38;2;32;148;243m-0.47048872[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17548294[0m [38;2;32;148;243m-0.08319724[0m [38;2;32;148;243m-0.27796796[0m [38;2;32;148;243m-0.35275104[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1110378[0m  [38;2;32;148;243m-0.05947729[0m [38;2;32;148;243m-0.18303625[0m [38;2;32;148;243m-0.23549409[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.045845[0m   [38;2;32;148;243m-0.03564062[0m [38;2;32;148;243m-0.0880325[0m  [38;2;32;148;243m-0.11812833[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01833044[0m [38;2;32;148;243m-0.00902976[0m  [38;2;32;148;243m0.00806685[0m [38;2;32;148;243m-0.00060101[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.65629595[0m [38;2;32;148;243m1[0m.         [38;2;32;148;243m0.35398312[0m [38;2;32;148;243m0.49236948[0m[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01833044[0m [38;2;32;148;243m-0.00902976[0m  [38;2;32;148;243m0[0m.         [38;2;32;148;243m-0.00060101[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00429017[0m [38;2;32;148;243m-0.00088047[0m [38;2;32;148;243m-0.0019709[0m  [38;2;32;148;243m-0.00261991[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00470862[0m [38;2;32;148;243m-0.00204245[0m [38;2;32;148;243m-0.00337299[0m  [38;2;32;148;243m0.00397335[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.13739976[0m [38;2;32;148;243m-0.05959964[0m [38;2;32;148;243m-0.09842568[0m  [38;2;32;148;243m0.11594435[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00222099[0m [38;2;32;148;243m0.0004814[0m  [38;2;32;148;243m0[0m.         [38;2;32;148;243m0.00228172[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m7.61122078e-05[0m [38;2;32;148;243m1.64974062e-05[0m [38;2;32;148;243m0.00000000e+00[0m [38;2;32;148;243m7.81931965e-05[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-3.523570691365356e-05[0m
>     [38;2;105;105;105m[07/10/23 07:56:50][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m11[0m
>     [38;2;105;105;105m[07/10/23 07:56:50][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m12[0m
>     [38;2;105;105;105m[07/10/23 07:56:50][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m13[0m
>     [38;2;105;105;105m[07/10/23 07:56:51][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m14[0m
>     [38;2;105;105;105m[07/10/23 07:56:51][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m15[0m
>     [38;2;105;105;105m[07/10/23 07:56:52][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m22.08327295[0m  [38;2;32;148;243m19.0197769[0m  [38;2;32;148;243m-36.65111885[0m  [38;2;32;148;243m35.64933259[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.07697097[0m  [38;2;32;148;243m18.93400173[0m [38;2;32;148;243m-36.74012144[0m  [38;2;32;148;243m35.59790233[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.08385286[0m  [38;2;32;148;243m18.86262503[0m [38;2;32;148;243m-36.81229751[0m  [38;2;32;148;243m35.56680775[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.1041726[0m   [38;2;32;148;243m18.80744513[0m [38;2;32;148;243m-36.86774956[0m  [38;2;32;148;243m35.55670459[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.13787963[0m  [38;2;32;148;243m18.76616913[0m [38;2;32;148;243m-36.9068429[0m   [38;2;32;148;243m35.56661673[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.29806806[0m  [38;2;32;148;243m18.85372515[0m [38;2;32;148;243m-36.78646258[0m  [38;2;32;148;243m35.59589835[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.47129009[0m  [38;2;32;148;243m18.95604273[0m [38;2;32;148;243m-36.64914559[0m  [38;2;32;148;243m35.64578086[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.6563732[0m   [38;2;32;148;243m19.07405443[0m [38;2;32;148;243m-36.494943[0m    [38;2;32;148;243m35.71658607[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.85494135[0m  [38;2;32;148;243m19.20666439[0m [38;2;32;148;243m-36.32391534[0m  [38;2;32;148;243m35.80825486[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m22.08327295[0m  [38;2;32;148;243m19.0197769[0m  [38;2;32;148;243m-36.65111885[0m  [38;2;32;148;243m35.64933259[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.14386161[0m  [38;2;32;148;243m18.96216574[0m [38;2;32;148;243m-36.64341855[0m  [38;2;32;148;243m35.7160735[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m22.21832877[0m  [38;2;32;148;243m18.91860861[0m [38;2;32;148;243m-36.61893687[0m  [38;2;32;148;243m35.80324569[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.30723783[0m  [38;2;32;148;243m18.88893401[0m [38;2;32;148;243m-36.57886993[0m  [38;2;32;148;243m35.91152916[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.41023902[0m  [38;2;32;148;243m18.8735014[0m  [38;2;32;148;243m-36.52280583[0m  [38;2;32;148;243m36.04005679[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.50133881[0m  [38;2;32;148;243m18.93349329[0m [38;2;32;148;243m-36.49758254[0m  [38;2;32;148;243m35.95072537[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.60675344[0m  [38;2;32;148;243m19.00765238[0m [38;2;32;148;243m-36.45586602[0m  [38;2;32;148;243m35.88193507[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.72712714[0m  [38;2;32;148;243m19.09626029[0m [38;2;32;148;243m-36.39758481[0m  [38;2;32;148;243m35.83369873[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m22.86168418[0m  [38;2;32;148;243m19.19884493[0m [38;2;32;148;243m-36.32274159[0m  [38;2;32;148;243m35.80633166[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06689064[0m [38;2;32;148;243m-0.02816401[0m [38;2;32;148;243m-0.09670289[0m [38;2;32;148;243m-0.11817117[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13447591[0m [38;2;32;148;243m-0.05598358[0m [38;2;32;148;243m-0.19336064[0m [38;2;32;148;243m-0.23643794[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20306523[0m [38;2;32;148;243m-0.08148888[0m [38;2;32;148;243m-0.28887963[0m [38;2;32;148;243m-0.35482457[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27235939[0m [38;2;32;148;243m-0.10733227[0m [38;2;32;148;243m-0.38403706[0m [38;2;32;148;243m-0.47344005[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20327075[0m [38;2;32;148;243m-0.07976814[0m [38;2;32;148;243m-0.28888004[0m [38;2;32;148;243m-0.35482702[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13546335[0m [38;2;32;148;243m-0.05160965[0m [38;2;32;148;243m-0.19327958[0m [38;2;32;148;243m-0.23615421[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07075393[0m [38;2;32;148;243m-0.02220586[0m [38;2;32;148;243m-0.09735819[0m [38;2;32;148;243m-0.11711267[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00674283[0m  [38;2;32;148;243m0.00781946[0m [38;2;32;148;243m-0.00117375[0m  [38;2;32;148;243m0.0019232[0m [1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06689064[0m [38;2;32;148;243m-0.02816401[0m [38;2;32;148;243m-0.09670289[0m [38;2;32;148;243m-0.11817117[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13447591[0m [38;2;32;148;243m-0.05598358[0m [38;2;32;148;243m-0.19336064[0m [38;2;32;148;243m-0.23643794[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20306523[0m [38;2;32;148;243m-0.08148888[0m [38;2;32;148;243m-0.28887963[0m [38;2;32;148;243m-0.35482457[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27235939[0m [38;2;32;148;243m-0.10733227[0m [38;2;32;148;243m-0.38403706[0m [38;2;32;148;243m-0.47344005[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.06908864[0m [38;2;32;148;243m0.02756413[0m [38;2;32;148;243m0.09515702[0m [38;2;32;148;243m0.11861303[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.13689604[0m [38;2;32;148;243m0.05572262[0m [38;2;32;148;243m0.19075749[0m [38;2;32;148;243m0.23728585[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.20160546[0m [38;2;32;148;243m0.08512641[0m [38;2;32;148;243m0.28667887[0m [38;2;32;148;243m0.35632739[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.26561656[0m [38;2;32;148;243m0.11515173[0m [38;2;32;148;243m0.38286331[0m [38;2;32;148;243m0.47536325[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06689064[0m [38;2;32;148;243m-0.02816401[0m [38;2;32;148;243m-0.09670289[0m [38;2;32;148;243m-0.11817117[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13447591[0m [38;2;32;148;243m-0.05598358[0m [38;2;32;148;243m-0.19336064[0m [38;2;32;148;243m-0.23643794[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20306523[0m [38;2;32;148;243m-0.08148888[0m [38;2;32;148;243m-0.28887963[0m [38;2;32;148;243m-0.35482457[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27235939[0m [38;2;32;148;243m-0.10733227[0m [38;2;32;148;243m-0.38403706[0m [38;2;32;148;243m-0.47344005[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20327075[0m [38;2;32;148;243m-0.07976814[0m [38;2;32;148;243m-0.28888004[0m [38;2;32;148;243m-0.35482702[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13546335[0m [38;2;32;148;243m-0.05160965[0m [38;2;32;148;243m-0.19327958[0m [38;2;32;148;243m-0.23615421[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07075393[0m [38;2;32;148;243m-0.02220586[0m [38;2;32;148;243m-0.09735819[0m [38;2;32;148;243m-0.11711267[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00674283[0m  [38;2;32;148;243m0.00781946[0m [38;2;32;148;243m-0.00117375[0m  [38;2;32;148;243m0.0019232[0m [1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.45913489[0m [38;2;32;148;243m0.83604902[0m [38;2;32;148;243m0.72009131[0m [38;2;32;148;243m0.85470485[0m[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00674283[0m  [38;2;32;148;243m0.00781946[0m [38;2;32;148;243m-0.00117375[0m  [38;2;32;148;243m0[0m.        [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00410104[0m  [38;2;32;148;243m0.00011216[0m [38;2;32;148;243m-0.00193722[0m [38;2;32;148;243m-0.00172661[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00468274[0m [38;2;32;148;243m-0.00197849[0m [38;2;32;148;243m-0.0033172[0m   [38;2;32;148;243m0.00396172[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.13664465[0m [38;2;32;148;243m-0.05773323[0m [38;2;32;148;243m-0.09679748[0m  [38;2;32;148;243m0.11560499[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00490455[0m [38;2;32;148;243m0.00606124[0m [38;2;32;148;243m0.00438789[0m [38;2;32;148;243m0[0m.        [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00016808[0m [38;2;32;148;243m0.00020772[0m [38;2;32;148;243m0.00015037[0m [38;2;32;148;243m0[0m.        [1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-4.064349363399149e-05[0m
>     [38;2;105;105;105m[07/10/23 07:56:52][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m16[0m
>     [38;2;105;105;105m[07/10/23 07:56:52][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m17[0m
>     [38;2;105;105;105m[07/10/23 07:56:52][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m18[0m
>     [38;2;105;105;105m[07/10/23 07:56:53][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m19[0m
>     [38;2;105;105;105m[07/10/23 07:56:53][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m20[0m
>     [38;2;105;105;105m[07/10/23 07:56:54][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m18.24449011[0m  [38;2;32;148;243m157.51395791[0m [38;2;32;148;243m-208.48354391[0m    [38;2;32;148;243m7.94362976[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m18.131001[0m    [38;2;32;148;243m157.57724448[0m [38;2;32;148;243m-208.48275539[0m    [38;2;32;148;243m7.79977811[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m18.03407439[0m  [38;2;32;148;243m157.65458278[0m [38;2;32;148;243m-208.46456333[0m    [38;2;32;148;243m7.67616995[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.94980559[0m  [38;2;32;148;243m157.74629708[0m [38;2;32;148;243m-208.42972391[0m    [38;2;32;148;243m7.57344121[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.87787847[0m  [38;2;32;148;243m157.85139826[0m [38;2;32;148;243m-208.37822275[0m    [38;2;32;148;243m7.49172438[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.86104423[0m  [38;2;32;148;243m157.84775292[0m [38;2;32;148;243m-208.3583655[0m     [38;2;32;148;243m7.50257305[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.86515279[0m  [38;2;32;148;243m157.85840203[0m [38;2;32;148;243m-208.32204505[0m    [38;2;32;148;243m7.53279089[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.88936298[0m  [38;2;32;148;243m157.8873257[0m  [38;2;32;148;243m-208.26938999[0m    [38;2;32;148;243m7.58293561[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.93276149[0m  [38;2;32;148;243m157.93030545[0m [38;2;32;148;243m-208.20040312[0m    [38;2;32;148;243m7.65313816[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m18.24449011[0m  [38;2;32;148;243m157.51395791[0m [38;2;32;148;243m-208.48354391[0m    [38;2;32;148;243m7.94362976[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m18.18665175[0m  [38;2;32;148;243m157.61341561[0m [38;2;32;148;243m-208.38213623[0m    [38;2;32;148;243m7.91836792[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m18.14357945[0m  [38;2;32;148;243m157.72687855[0m [38;2;32;148;243m-208.26400887[0m    [38;2;32;148;243m7.91352822[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m18.11376441[0m  [38;2;32;148;243m157.85446265[0m [38;2;32;148;243m-208.13028082[0m    [38;2;32;148;243m7.92975756[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m18.09712312[0m  [38;2;32;148;243m157.99621367[0m [38;2;32;148;243m-207.97971005[0m    [38;2;32;148;243m7.96695753[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m18.02461769[0m  [38;2;32;148;243m157.95833194[0m [38;2;32;148;243m-208.05892191[0m    [38;2;32;148;243m7.85916573[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.96958153[0m  [38;2;32;148;243m157.93478819[0m [38;2;32;148;243m-208.12071791[0m    [38;2;32;148;243m7.77120336[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.93172414[0m  [38;2;32;148;243m157.92673721[0m [38;2;32;148;243m-208.16486932[0m    [38;2;32;148;243m7.70347135[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m17.91064328[0m  [38;2;32;148;243m157.93285714[0m [38;2;32;148;243m-208.19159752[0m    [38;2;32;148;243m7.65597587[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05565075[0m [38;2;32;148;243m-0.03617114[0m [38;2;32;148;243m-0.10061915[0m [38;2;32;148;243m-0.1185898[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10950507[0m [38;2;32;148;243m-0.07229577[0m [38;2;32;148;243m-0.20055447[0m [38;2;32;148;243m-0.23735827[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.16395882[0m [38;2;32;148;243m-0.10816557[0m [38;2;32;148;243m-0.29944309[0m [38;2;32;148;243m-0.35631635[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21924465[0m [38;2;32;148;243m-0.14481541[0m [38;2;32;148;243m-0.3985127[0m  [38;2;32;148;243m-0.47523315[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.16357346[0m [38;2;32;148;243m-0.11057902[0m [38;2;32;148;243m-0.29944359[0m [38;2;32;148;243m-0.35659268[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10442873[0m [38;2;32;148;243m-0.07638616[0m [38;2;32;148;243m-0.20132714[0m [38;2;32;148;243m-0.23841247[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.04236116[0m [38;2;32;148;243m-0.0394115[0m  [38;2;32;148;243m-0.10452067[0m [38;2;32;148;243m-0.12053573[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.02211821[0m [38;2;32;148;243m-0.0025517[0m  [38;2;32;148;243m-0.0088056[0m  [38;2;32;148;243m-0.00283771[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05565075[0m [38;2;32;148;243m-0.03617114[0m [38;2;32;148;243m-0.10061915[0m [38;2;32;148;243m-0.1185898[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10950507[0m [38;2;32;148;243m-0.07229577[0m [38;2;32;148;243m-0.20055447[0m [38;2;32;148;243m-0.23735827[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.16395882[0m [38;2;32;148;243m-0.10816557[0m [38;2;32;148;243m-0.29944309[0m [38;2;32;148;243m-0.35631635[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21924465[0m [38;2;32;148;243m-0.14481541[0m [38;2;32;148;243m-0.3985127[0m  [38;2;32;148;243m-0.47523315[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.05567119[0m [38;2;32;148;243m0.03423639[0m [38;2;32;148;243m0.09906911[0m [38;2;32;148;243m0.11864046[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.11481592[0m [38;2;32;148;243m0.06842925[0m [38;2;32;148;243m0.19718556[0m [38;2;32;148;243m0.23682067[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.17688349[0m [38;2;32;148;243m0.10540391[0m [38;2;32;148;243m0.29399203[0m [38;2;32;148;243m0.35469741[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.24136286[0m [38;2;32;148;243m0.14226372[0m [38;2;32;148;243m0.3897071[0m  [38;2;32;148;243m0.47239543[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05565075[0m [38;2;32;148;243m-0.03617114[0m [38;2;32;148;243m-0.10061915[0m [38;2;32;148;243m-0.1185898[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10950507[0m [38;2;32;148;243m-0.07229577[0m [38;2;32;148;243m-0.20055447[0m [38;2;32;148;243m-0.23735827[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.16395882[0m [38;2;32;148;243m-0.10816557[0m [38;2;32;148;243m-0.29944309[0m [38;2;32;148;243m-0.35631635[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21924465[0m [38;2;32;148;243m-0.14481541[0m [38;2;32;148;243m-0.3985127[0m  [38;2;32;148;243m-0.47523315[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.16357346[0m [38;2;32;148;243m-0.11057902[0m [38;2;32;148;243m-0.29944359[0m [38;2;32;148;243m-0.35659268[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10442873[0m [38;2;32;148;243m-0.07638616[0m [38;2;32;148;243m-0.20132714[0m [38;2;32;148;243m-0.23841247[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.04236116[0m [38;2;32;148;243m-0.0394115[0m  [38;2;32;148;243m-0.10452067[0m [38;2;32;148;243m-0.12053573[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.02211821[0m [38;2;32;148;243m-0.0025517[0m  [38;2;32;148;243m-0.0088056[0m  [38;2;32;148;243m-0.00283771[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m.         [38;2;32;148;243m0.65777047[0m [38;2;32;148;243m0.74680857[0m [38;2;32;148;243m1[0m.        [1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.02211821[0m [38;2;32;148;243m-0[0m.         [38;2;32;148;243m-0[0m.         [38;2;32;148;243m-0.00283771[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00307229[0m  [38;2;32;148;243m0.00081471[0m [38;2;32;148;243m-0.00146663[0m [38;2;32;148;243m-0.00150205[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00484404[0m [38;2;32;148;243m-0.00162107[0m [38;2;32;148;243m-0.00354424[0m  [38;2;32;148;243m0.00406079[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.14135151[0m [38;2;32;148;243m-0.04730375[0m [38;2;32;148;243m-0.10342278[0m  [38;2;32;148;243m0.11849601[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00132971[0m [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0.00211542[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m4.55685873e-05[0m [38;2;32;148;243m0.00000000e+00[0m [38;2;32;148;243m0.00000000e+00[0m [38;2;32;148;243m7.24941821e-05[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-4.81144798353746e-05[0m
>     [38;2;105;105;105m[07/10/23 07:56:54][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m21[0m
>     [38;2;105;105;105m[07/10/23 07:56:54][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m22[0m
>     [38;2;105;105;105m[07/10/23 07:56:54][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m23[0m
>     [38;2;105;105;105m[07/10/23 07:56:55][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m24[0m
>     [38;2;105;105;105m[07/10/23 07:56:55][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m25[0m
>     [38;2;105;105;105m[07/10/23 07:56:55][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m103.02841282[0m  [38;2;32;148;243m66.72029544[0m  [38;2;32;148;243m59.90481188[0m [38;2;32;148;243m146.00124109[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.00555319[0m  [38;2;32;148;243m66.74978538[0m  [38;2;32;148;243m59.90840465[0m [38;2;32;148;243m145.85528037[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m102.99730392[0m  [38;2;32;148;243m66.7937695[0m   [38;2;32;148;243m59.92887223[0m [38;2;32;148;243m145.72930134[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.00125737[0m  [38;2;32;148;243m66.85207329[0m  [38;2;32;148;243m59.96577299[0m [38;2;32;148;243m145.62408096[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.01948883[0m  [38;2;32;148;243m66.9247455[0m   [38;2;32;148;243m60.01915062[0m [38;2;32;148;243m145.53920011[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.20998844[0m  [38;2;32;148;243m66.90147596[0m  [38;2;32;148;243m60.11104722[0m [38;2;32;148;243m145.71750439[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.41490006[0m  [38;2;32;148;243m66.89165518[0m  [38;2;32;148;243m60.2198083[0m  [38;2;32;148;243m145.915704[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m103.63465314[0m  [38;2;32;148;243m66.89507612[0m  [38;2;32;148;243m60.34472098[0m [38;2;32;148;243m146.1340541[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m103.86859577[0m  [38;2;32;148;243m66.91086793[0m  [38;2;32;148;243m60.48597197[0m [38;2;32;148;243m146.3725016[0m [1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m103.02841282[0m  [38;2;32;148;243m66.72029544[0m  [38;2;32;148;243m59.90481188[0m [38;2;32;148;243m146.00124109[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.06221521[0m  [38;2;32;148;243m66.78220304[0m  [38;2;32;148;243m60.00541219[0m [38;2;32;148;243m145.97179754[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.11016976[0m  [38;2;32;148;243m66.85855726[0m  [38;2;32;148;243m60.1232324[0m  [38;2;32;148;243m145.96242371[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.17152685[0m  [38;2;32;148;243m66.94940619[0m  [38;2;32;148;243m60.25727658[0m [38;2;32;148;243m145.97369716[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.24708037[0m  [38;2;32;148;243m67.05444773[0m  [38;2;32;148;243m60.40638075[0m [38;2;32;148;243m146.00512372[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.38050002[0m  [38;2;32;148;243m66.99928691[0m  [38;2;32;148;243m60.40429756[0m [38;2;32;148;243m146.06712644[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.52788974[0m  [38;2;32;148;243m66.95744795[0m  [38;2;32;148;243m60.42047951[0m [38;2;32;148;243m146.14933104[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.68961244[0m  [38;2;32;148;243m66.92731634[0m  [38;2;32;148;243m60.45239386[0m [38;2;32;148;243m146.25190874[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m103.86522803[0m  [38;2;32;148;243m66.90645727[0m  [38;2;32;148;243m60.50043445[0m [38;2;32;148;243m146.37474149[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05666201[0m [38;2;32;148;243m-0.03241766[0m [38;2;32;148;243m-0.09700755[0m [38;2;32;148;243m-0.11651717[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.11286583[0m [38;2;32;148;243m-0.06478776[0m [38;2;32;148;243m-0.19436017[0m [38;2;32;148;243m-0.23312237[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17026948[0m [38;2;32;148;243m-0.0973329[0m  [38;2;32;148;243m-0.29150359[0m [38;2;32;148;243m-0.34961621[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.22759154[0m [38;2;32;148;243m-0.12970223[0m [38;2;32;148;243m-0.38723013[0m [38;2;32;148;243m-0.46592362[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17051158[0m [38;2;32;148;243m-0.09781095[0m [38;2;32;148;243m-0.29325034[0m [38;2;32;148;243m-0.34962204[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.11298968[0m [38;2;32;148;243m-0.06579277[0m [38;2;32;148;243m-0.20067121[0m [38;2;32;148;243m-0.23362704[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0549593[0m  [38;2;32;148;243m-0.03224022[0m [38;2;32;148;243m-0.10767288[0m [38;2;32;148;243m-0.11785464[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00336773[0m  [38;2;32;148;243m0.00441065[0m [38;2;32;148;243m-0.01446248[0m [38;2;32;148;243m-0.0022399[0m [1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05666201[0m [38;2;32;148;243m-0.03241766[0m [38;2;32;148;243m-0.09700755[0m [38;2;32;148;243m-0.11651717[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.11286583[0m [38;2;32;148;243m-0.06478776[0m [38;2;32;148;243m-0.19436017[0m [38;2;32;148;243m-0.23312237[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17026948[0m [38;2;32;148;243m-0.0973329[0m  [38;2;32;148;243m-0.29150359[0m [38;2;32;148;243m-0.34961621[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.22759154[0m [38;2;32;148;243m-0.12970223[0m [38;2;32;148;243m-0.38723013[0m [38;2;32;148;243m-0.46592362[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.05707996[0m [38;2;32;148;243m0.03189128[0m [38;2;32;148;243m0.09397978[0m [38;2;32;148;243m0.11630157[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.11460186[0m [38;2;32;148;243m0.06390946[0m [38;2;32;148;243m0.18655891[0m [38;2;32;148;243m0.23229658[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.17263224[0m [38;2;32;148;243m0.09746201[0m [38;2;32;148;243m0.27955724[0m [38;2;32;148;243m0.34806898[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.23095927[0m [38;2;32;148;243m0.13411288[0m [38;2;32;148;243m0.37276764[0m [38;2;32;148;243m0.46368372[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.05666201[0m [38;2;32;148;243m-0.03241766[0m [38;2;32;148;243m-0.09700755[0m [38;2;32;148;243m-0.11651717[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.11286583[0m [38;2;32;148;243m-0.06478776[0m [38;2;32;148;243m-0.19436017[0m [38;2;32;148;243m-0.23312237[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17026948[0m [38;2;32;148;243m-0.0973329[0m  [38;2;32;148;243m-0.29150359[0m [38;2;32;148;243m-0.34961621[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.22759154[0m [38;2;32;148;243m-0.12970223[0m [38;2;32;148;243m-0.38723013[0m [38;2;32;148;243m-0.46592362[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17051158[0m [38;2;32;148;243m-0.09781095[0m [38;2;32;148;243m-0.29325034[0m [38;2;32;148;243m-0.34962204[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.11298968[0m [38;2;32;148;243m-0.06579277[0m [38;2;32;148;243m-0.20067121[0m [38;2;32;148;243m-0.23362704[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0549593[0m  [38;2;32;148;243m-0.03224022[0m [38;2;32;148;243m-0.10767288[0m [38;2;32;148;243m-0.11785464[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00336773[0m  [38;2;32;148;243m0.00441065[0m [38;2;32;148;243m-0.01446248[0m [38;2;32;148;243m-0.0022399[0m [1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.43308762[0m [38;2;32;148;243m0.83013924[0m [38;2;32;148;243m0.55121929[0m [38;2;32;148;243m0.68832071[0m[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0.00441065[0m [38;2;32;148;243m-0.01446248[0m [38;2;32;148;243m-0[0m.        [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.0027481[0m   [38;2;32;148;243m0.00119082[0m [38;2;32;148;243m-0.00158445[0m [38;2;32;148;243m-0.00089441[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00451238[0m [38;2;32;148;243m-0.00182152[0m [38;2;32;148;243m-0.0035942[0m   [38;2;32;148;243m0.00441841[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.13167354[0m [38;2;32;148;243m-0.05315281[0m [38;2;32;148;243m-0.10488051[0m  [38;2;32;148;243m0.12893134[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0.0018939[0m  [38;2;32;148;243m0.00298696[0m [38;2;32;148;243m0[0m.        [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00000000e+00[0m [38;2;32;148;243m6.49030205e-05[0m [38;2;32;148;243m1.02361421e-04[0m [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-3.587542080169492e-05[0m
>     [38;2;105;105;105m[07/10/23 07:56:56][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m26[0m
>     [38;2;105;105;105m[07/10/23 07:56:56][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m27[0m
>     [38;2;105;105;105m[07/10/23 07:56:56][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m28[0m
>     [38;2;105;105;105m[07/10/23 07:56:57][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m29[0m
>     [38;2;105;105;105m[07/10/23 07:56:57][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m30[0m
>     [38;2;105;105;105m[07/10/23 07:56:58][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m   [38;2;32;148;243m5.44430506[0m [38;2;32;148;243m-165.95958335[0m  [38;2;32;148;243m-15.50481522[0m [38;2;32;148;243m-102.40867299[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.43718998[0m [38;2;32;148;243m-166.10277443[0m  [38;2;32;148;243m-15.62779543[0m [38;2;32;148;243m-102.56181693[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.44464905[0m [38;2;32;148;243m-166.22901843[0m  [38;2;32;148;243m-15.7356583[0m  [38;2;32;148;243m-102.69496592[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.46736464[0m [38;2;32;148;243m-166.33990308[0m  [38;2;32;148;243m-15.82781492[0m [38;2;32;148;243m-102.80737464[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.50500891[0m [38;2;32;148;243m-166.43443664[0m  [38;2;32;148;243m-15.90440405[0m [38;2;32;148;243m-102.89983611[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.46719889[0m [38;2;32;148;243m-166.4180316[0m   [38;2;32;148;243m-15.59365182[0m [38;2;32;148;243m-102.80127799[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.44397975[0m [38;2;32;148;243m-166.38658933[0m  [38;2;32;148;243m-15.26972746[0m [38;2;32;148;243m-102.68257058[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.43542179[0m [38;2;32;148;243m-166.34041499[0m  [38;2;32;148;243m-14.9318252[0m  [38;2;32;148;243m-102.54360961[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.44179493[0m [38;2;32;148;243m-166.279924[0m    [38;2;32;148;243m-14.57940301[0m [38;2;32;148;243m-102.38444531[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m   [38;2;32;148;243m5.44430506[0m [38;2;32;148;243m-165.95958335[0m  [38;2;32;148;243m-15.50481522[0m [38;2;32;148;243m-102.40867299[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.50781019[0m [38;2;32;148;243m-166.07558872[0m  [38;2;32;148;243m-15.54001[0m    [38;2;32;148;243m-102.44454202[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.58658104[0m [38;2;32;148;243m-166.17276673[0m  [38;2;32;148;243m-15.55915382[0m [38;2;32;148;243m-102.46024532[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.68270227[0m [38;2;32;148;243m-166.25379651[0m  [38;2;32;148;243m-15.56265621[0m [38;2;32;148;243m-102.45555478[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.79468708[0m [38;2;32;148;243m-166.31696368[0m  [38;2;32;148;243m-15.55021339[0m [38;2;32;148;243m-102.43120679[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.68351855[0m [38;2;32;148;243m-166.33485863[0m  [38;2;32;148;243m-15.3277197[0m  [38;2;32;148;243m-102.44957194[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.58743691[0m [38;2;32;148;243m-166.3402427[0m   [38;2;32;148;243m-15.09050885[0m [38;2;32;148;243m-102.44788906[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.50631117[0m [38;2;32;148;243m-166.33310876[0m  [38;2;32;148;243m-14.83848715[0m [38;2;32;148;243m-102.42597688[0m[1m][0m
>      [1m[[0m   [38;2;32;148;243m5.44094998[0m [38;2;32;148;243m-166.31355254[0m  [38;2;32;148;243m-14.57136671[0m [38;2;32;148;243m-102.38389437[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07062021[0m [38;2;32;148;243m-0.02718572[0m [38;2;32;148;243m-0.08778543[0m [38;2;32;148;243m-0.11727491[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14193199[0m [38;2;32;148;243m-0.0562517[0m  [38;2;32;148;243m-0.17650448[0m [38;2;32;148;243m-0.23472059[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21533763[0m [38;2;32;148;243m-0.08610658[0m [38;2;32;148;243m-0.26515871[0m [38;2;32;148;243m-0.35181986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28967817[0m [38;2;32;148;243m-0.11747296[0m [38;2;32;148;243m-0.35419066[0m [38;2;32;148;243m-0.46862932[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21631966[0m [38;2;32;148;243m-0.08317297[0m [38;2;32;148;243m-0.26593212[0m [38;2;32;148;243m-0.35170605[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14345716[0m [38;2;32;148;243m-0.04634663[0m [38;2;32;148;243m-0.17921861[0m [38;2;32;148;243m-0.23468152[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07088938[0m [38;2;32;148;243m-0.00730623[0m [38;2;32;148;243m-0.09333805[0m [38;2;32;148;243m-0.11763273[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00084495[0m  [38;2;32;148;243m0.03362854[0m [38;2;32;148;243m-0.0080363[0m  [38;2;32;148;243m-0.00055094[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07062021[0m [38;2;32;148;243m-0.02718572[0m [38;2;32;148;243m-0.08778543[0m [38;2;32;148;243m-0.11727491[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14193199[0m [38;2;32;148;243m-0.0562517[0m  [38;2;32;148;243m-0.17650448[0m [38;2;32;148;243m-0.23472059[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21533763[0m [38;2;32;148;243m-0.08610658[0m [38;2;32;148;243m-0.26515871[0m [38;2;32;148;243m-0.35181986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28967817[0m [38;2;32;148;243m-0.11747296[0m [38;2;32;148;243m-0.35419066[0m [38;2;32;148;243m-0.46862932[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.07335851[0m [38;2;32;148;243m0.0343[0m     [38;2;32;148;243m0.08825854[0m [38;2;32;148;243m0.11692327[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.14622101[0m [38;2;32;148;243m0.07112634[0m [38;2;32;148;243m0.17497205[0m [38;2;32;148;243m0.23394779[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21878879[0m [38;2;32;148;243m0.11016673[0m [38;2;32;148;243m0.26085261[0m [38;2;32;148;243m0.35099659[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.29052312[0m [38;2;32;148;243m0.1511015[0m  [38;2;32;148;243m0.34615436[0m [38;2;32;148;243m0.46807838[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07062021[0m [38;2;32;148;243m-0.02718572[0m [38;2;32;148;243m-0.08778543[0m [38;2;32;148;243m-0.11727491[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14193199[0m [38;2;32;148;243m-0.0562517[0m  [38;2;32;148;243m-0.17650448[0m [38;2;32;148;243m-0.23472059[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21533763[0m [38;2;32;148;243m-0.08610658[0m [38;2;32;148;243m-0.26515871[0m [38;2;32;148;243m-0.35181986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28967817[0m [38;2;32;148;243m-0.11747296[0m [38;2;32;148;243m-0.35419066[0m [38;2;32;148;243m-0.46862932[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21631966[0m [38;2;32;148;243m-0.08317297[0m [38;2;32;148;243m-0.26593212[0m [38;2;32;148;243m-0.35170605[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14345716[0m [38;2;32;148;243m-0.04634663[0m [38;2;32;148;243m-0.17921861[0m [38;2;32;148;243m-0.23468152[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07088938[0m [38;2;32;148;243m-0.00730623[0m [38;2;32;148;243m-0.09333805[0m [38;2;32;148;243m-0.11763273[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00084495[0m  [38;2;32;148;243m0.03362854[0m [38;2;32;148;243m-0.0080363[0m  [38;2;32;148;243m-0.00055094[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m.         [38;2;32;148;243m1[0m.         [38;2;32;148;243m0.39319543[0m [38;2;32;148;243m0.97552585[0m[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00084495[0m  [38;2;32;148;243m0.03362854[0m [38;2;32;148;243m-0[0m.         [38;2;32;148;243m-0.00055094[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00267364[0m  [38;2;32;148;243m0.00204628[0m [38;2;32;148;243m-0.00143384[0m [38;2;32;148;243m-0.00045363[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00460961[0m [38;2;32;148;243m-0.00176722[0m [38;2;32;148;243m-0.00379743[0m  [38;2;32;148;243m0.0038541[0m [1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.13451074[0m [38;2;32;148;243m-0.05156846[0m [38;2;32;148;243m-0.11081098[0m  [38;2;32;148;243m0.11246456[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00099198[0m [38;2;32;148;243m0.00188247[0m [38;2;32;148;243m0[0m.         [38;2;32;148;243m0.00811582[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m3.39945351e-05[0m [38;2;32;148;243m6.45112363e-05[0m [38;2;32;148;243m0.00000000e+00[0m [38;2;32;148;243m2.78124805e-04[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-4.729706796001338e-05[0m
>     [38;2;105;105;105m[07/10/23 07:56:58][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m31[0m
>     [38;2;105;105;105m[07/10/23 07:56:58][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m32[0m
>     [38;2;105;105;105m[07/10/23 07:56:59][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m33[0m
>     [38;2;105;105;105m[07/10/23 07:56:59][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m34[0m
>     [38;2;105;105;105m[07/10/23 07:56:59][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m35[0m
>     [38;2;105;105;105m[07/10/23 07:57:00][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-38.24109866[0m   [38;2;32;148;243m5.55612112[0m  [38;2;32;148;243m-1.76277452[0m [38;2;32;148;243m-52.84427962[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.32994306[0m   [38;2;32;148;243m5.55140781[0m  [38;2;32;148;243m-1.64083871[0m [38;2;32;148;243m-52.86777806[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.40531362[0m   [38;2;32;148;243m5.56129168[0m  [38;2;32;148;243m-1.50469869[0m [38;2;32;148;243m-52.87194866[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.46792643[0m   [38;2;32;148;243m5.58535052[0m  [38;2;32;148;243m-1.35394915[0m [38;2;32;148;243m-52.85704699[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.51648618[0m   [38;2;32;148;243m5.62468415[0m  [38;2;32;148;243m-1.18861717[0m [38;2;32;148;243m-52.82292239[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.31153675[0m   [38;2;32;148;243m5.73641202[0m  [38;2;32;148;243m-1.11288606[0m [38;2;32;148;243m-52.65606878[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.09224637[0m   [38;2;32;148;243m5.8627936[0m   [38;2;32;148;243m-1.02476863[0m [38;2;32;148;243m-52.46855894[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-37.85867931[0m   [38;2;32;148;243m6.00371097[0m  [38;2;32;148;243m-0.92164249[0m [38;2;32;148;243m-52.26006541[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-37.6106359[0m    [38;2;32;148;243m6.15897688[0m  [38;2;32;148;243m-0.80273488[0m [38;2;32;148;243m-52.03065188[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m-38.24109866[0m   [38;2;32;148;243m5.55612112[0m  [38;2;32;148;243m-1.76277452[0m [38;2;32;148;243m-52.84427962[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.26222741[0m   [38;2;32;148;243m5.57653363[0m  [38;2;32;148;243m-1.56187095[0m [38;2;32;148;243m-52.7515782[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-38.26973224[0m   [38;2;32;148;243m5.61095297[0m  [38;2;32;148;243m-1.34509313[0m [38;2;32;148;243m-52.63971354[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.26380994[0m   [38;2;32;148;243m5.65981386[0m  [38;2;32;148;243m-1.11211757[0m [38;2;32;148;243m-52.50928377[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.2439666[0m    [38;2;32;148;243m5.72255661[0m  [38;2;32;148;243m-0.86315277[0m [38;2;32;148;243m-52.35968199[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-38.10907575[0m   [38;2;32;148;243m5.81183388[0m  [38;2;32;148;243m-0.87027295[0m [38;2;32;148;243m-52.30873805[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-37.96184935[0m   [38;2;32;148;243m5.91481407[0m  [38;2;32;148;243m-0.86351973[0m [38;2;32;148;243m-52.23756461[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-37.80125051[0m   [38;2;32;148;243m6.03101643[0m  [38;2;32;148;243m-0.84187415[0m [38;2;32;148;243m-52.14581443[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-37.62641755[0m   [38;2;32;148;243m6.15964415[0m  [38;2;32;148;243m-0.80499075[0m [38;2;32;148;243m-52.03352906[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06771565[0m [38;2;32;148;243m-0.02512582[0m [38;2;32;148;243m-0.07896776[0m [38;2;32;148;243m-0.11619986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13558138[0m [38;2;32;148;243m-0.04966129[0m [38;2;32;148;243m-0.15960556[0m [38;2;32;148;243m-0.23223511[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20411649[0m [38;2;32;148;243m-0.07446334[0m [38;2;32;148;243m-0.24183158[0m [38;2;32;148;243m-0.34776322[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27251958[0m [38;2;32;148;243m-0.09787246[0m [38;2;32;148;243m-0.3254644[0m  [38;2;32;148;243m-0.4632404[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.202461[0m   [38;2;32;148;243m-0.07542186[0m [38;2;32;148;243m-0.24261311[0m [38;2;32;148;243m-0.34733073[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13039702[0m [38;2;32;148;243m-0.05202047[0m [38;2;32;148;243m-0.1612489[0m  [38;2;32;148;243m-0.23099433[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0574288[0m  [38;2;32;148;243m-0.02730546[0m [38;2;32;148;243m-0.07976834[0m [38;2;32;148;243m-0.11425098[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01578164[0m [38;2;32;148;243m-0.00066727[0m  [38;2;32;148;243m0.00225587[0m  [38;2;32;148;243m0.00287718[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06771565[0m [38;2;32;148;243m-0.02512582[0m [38;2;32;148;243m-0.07896776[0m [38;2;32;148;243m-0.11619986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13558138[0m [38;2;32;148;243m-0.04966129[0m [38;2;32;148;243m-0.15960556[0m [38;2;32;148;243m-0.23223511[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20411649[0m [38;2;32;148;243m-0.07446334[0m [38;2;32;148;243m-0.24183158[0m [38;2;32;148;243m-0.34776322[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27251958[0m [38;2;32;148;243m-0.09787246[0m [38;2;32;148;243m-0.3254644[0m  [38;2;32;148;243m-0.4632404[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.07005858[0m [38;2;32;148;243m0.0224506[0m  [38;2;32;148;243m0.08285129[0m [38;2;32;148;243m0.11590967[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.14212256[0m [38;2;32;148;243m0.04585199[0m [38;2;32;148;243m0.1642155[0m  [38;2;32;148;243m0.23224607[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21509078[0m [38;2;32;148;243m0.070567[0m   [38;2;32;148;243m0.24569606[0m [38;2;32;148;243m0.34898942[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.28830122[0m [38;2;32;148;243m0.09720519[0m [38;2;32;148;243m0.32772027[0m [38;2;32;148;243m0.46611758[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06771565[0m [38;2;32;148;243m-0.02512582[0m [38;2;32;148;243m-0.07896776[0m [38;2;32;148;243m-0.11619986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13558138[0m [38;2;32;148;243m-0.04966129[0m [38;2;32;148;243m-0.15960556[0m [38;2;32;148;243m-0.23223511[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20411649[0m [38;2;32;148;243m-0.07446334[0m [38;2;32;148;243m-0.24183158[0m [38;2;32;148;243m-0.34776322[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27251958[0m [38;2;32;148;243m-0.09787246[0m [38;2;32;148;243m-0.3254644[0m  [38;2;32;148;243m-0.4632404[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.202461[0m   [38;2;32;148;243m-0.07542186[0m [38;2;32;148;243m-0.24261311[0m [38;2;32;148;243m-0.34733073[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13039702[0m [38;2;32;148;243m-0.05202047[0m [38;2;32;148;243m-0.1612489[0m  [38;2;32;148;243m-0.23099433[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0574288[0m  [38;2;32;148;243m-0.02730546[0m [38;2;32;148;243m-0.07976834[0m [38;2;32;148;243m-0.11425098[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01578164[0m [38;2;32;148;243m-0.00066727[0m  [38;2;32;148;243m0.00225587[0m  [38;2;32;148;243m0.00287718[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.54081332[0m [38;2;32;148;243m0.54688156[0m [38;2;32;148;243m0.3837424[0m  [38;2;32;148;243m0.4445243[0m [1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01578164[0m [38;2;32;148;243m-0.00066727[0m  [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m. [38;2;32;148;243m0[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-2.40489594e-03[0m  [38;2;32;148;243m2.42568817e-03[0m [38;2;32;148;243m-1.22128286e-03[0m  [38;2;32;148;243m3.33506631e-05[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.0044297[0m  [38;2;32;148;243m-0.00151298[0m [38;2;32;148;243m-0.00355408[0m  [38;2;32;148;243m0.00326773[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.12926073[0m [38;2;32;148;243m-0.04414946[0m [38;2;32;148;243m-0.10370987[0m  [38;2;32;148;243m0.09535414[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00126963[0m [38;2;32;148;243m0.00183703[0m [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m4.35095927e-05[0m [38;2;32;148;243m6.29539451e-05[0m [38;2;32;148;243m0.00000000e+00[0m [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-2.710630203148566e-05[0m
>     [38;2;105;105;105m[07/10/23 07:57:00][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m36[0m
>     [38;2;105;105;105m[07/10/23 07:57:01][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m37[0m
>     [38;2;105;105;105m[07/10/23 07:57:01][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m38[0m
>     [38;2;105;105;105m[07/10/23 07:57:02][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m39[0m
>     [38;2;105;105;105m[07/10/23 07:57:02][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m40[0m
>     [38;2;105;105;105m[07/10/23 07:57:02][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m37.53083778[0m [38;2;32;148;243m-61.47612458[0m [38;2;32;148;243m194.49383362[0m [38;2;32;148;243m-75.26659131[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.41314729[0m [38;2;32;148;243m-61.56796393[0m [38;2;32;148;243m194.42693791[0m [38;2;32;148;243m-75.35635768[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.30997375[0m [38;2;32;148;243m-61.64422221[0m [38;2;32;148;243m194.37883272[0m [38;2;32;148;243m-75.42621902[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.21778242[0m [38;2;32;148;243m-61.70354439[0m [38;2;32;148;243m194.3457032[0m  [38;2;32;148;243m-75.47606722[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.1380567[0m  [38;2;32;148;243m-61.74728792[0m [38;2;32;148;243m194.32642745[0m [38;2;32;148;243m-75.5059038[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m37.18193878[0m [38;2;32;148;243m-61.7853854[0m  [38;2;32;148;243m194.2253633[0m  [38;2;32;148;243m-75.27512306[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.24120223[0m [38;2;32;148;243m-61.80516899[0m [38;2;32;148;243m194.14110165[0m [38;2;32;148;243m-75.02533298[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.31483923[0m [38;2;32;148;243m-61.80731799[0m [38;2;32;148;243m194.0737638[0m  [38;2;32;148;243m-74.75607289[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.40107025[0m [38;2;32;148;243m-61.7919109[0m  [38;2;32;148;243m194.02513639[0m [38;2;32;148;243m-74.46727477[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m37.53083778[0m [38;2;32;148;243m-61.47612458[0m [38;2;32;148;243m194.49383362[0m [38;2;32;148;243m-75.26659131[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.48417588[0m [38;2;32;148;243m-61.54316591[0m [38;2;32;148;243m194.51489672[0m [38;2;32;148;243m-75.23936232[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.45182683[0m [38;2;32;148;243m-61.5952443[0m  [38;2;32;148;243m194.55346786[0m [38;2;32;148;243m-75.19268807[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.43217035[0m [38;2;32;148;243m-61.63209287[0m [38;2;32;148;243m194.60743084[0m [38;2;32;148;243m-75.1262975[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m37.42601492[0m [38;2;32;148;243m-61.65399555[0m [38;2;32;148;243m194.6764139[0m  [38;2;32;148;243m-75.03984482[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.3959529[0m  [38;2;32;148;243m-61.71622516[0m [38;2;32;148;243m194.48553365[0m [38;2;32;148;243m-74.92522514[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.38070855[0m [38;2;32;148;243m-61.76334278[0m [38;2;32;148;243m194.31072335[0m [38;2;32;148;243m-74.79112617[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.37999823[0m [38;2;32;148;243m-61.79563195[0m [38;2;32;148;243m194.15223251[0m [38;2;32;148;243m-74.63721938[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m37.39333859[0m [38;2;32;148;243m-61.81332954[0m [38;2;32;148;243m194.01100144[0m [38;2;32;148;243m-74.46363194[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07102859[0m [38;2;32;148;243m-0.02479802[0m [38;2;32;148;243m-0.08795881[0m [38;2;32;148;243m-0.11699536[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14185308[0m [38;2;32;148;243m-0.04897791[0m [38;2;32;148;243m-0.17463513[0m [38;2;32;148;243m-0.23353095[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21438792[0m [38;2;32;148;243m-0.07145152[0m [38;2;32;148;243m-0.26172765[0m [38;2;32;148;243m-0.34976972[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28795822[0m [38;2;32;148;243m-0.09329237[0m [38;2;32;148;243m-0.34998644[0m [38;2;32;148;243m-0.46605898[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21401412[0m [38;2;32;148;243m-0.06916024[0m [38;2;32;148;243m-0.26017036[0m [38;2;32;148;243m-0.34989792[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13950632[0m [38;2;32;148;243m-0.04182621[0m [38;2;32;148;243m-0.16962171[0m [38;2;32;148;243m-0.23420681[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06515899[0m [38;2;32;148;243m-0.01168603[0m [38;2;32;148;243m-0.07846871[0m [38;2;32;148;243m-0.11885351[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00773165[0m  [38;2;32;148;243m0.02141864[0m  [38;2;32;148;243m0.01413494[0m [38;2;32;148;243m-0.00364284[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07102859[0m [38;2;32;148;243m-0.02479802[0m [38;2;32;148;243m-0.08795881[0m [38;2;32;148;243m-0.11699536[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14185308[0m [38;2;32;148;243m-0.04897791[0m [38;2;32;148;243m-0.17463513[0m [38;2;32;148;243m-0.23353095[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21438792[0m [38;2;32;148;243m-0.07145152[0m [38;2;32;148;243m-0.26172765[0m [38;2;32;148;243m-0.34976972[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28795822[0m [38;2;32;148;243m-0.09329237[0m [38;2;32;148;243m-0.34998644[0m [38;2;32;148;243m-0.46605898[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.0739441[0m  [38;2;32;148;243m0.02413213[0m [38;2;32;148;243m0.08981609[0m [38;2;32;148;243m0.11616105[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.1484519[0m  [38;2;32;148;243m0.05146617[0m [38;2;32;148;243m0.18036474[0m [38;2;32;148;243m0.23185217[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.22279923[0m [38;2;32;148;243m0.08160634[0m [38;2;32;148;243m0.27151774[0m [38;2;32;148;243m0.34720547[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.29568987[0m [38;2;32;148;243m0.11471101[0m [38;2;32;148;243m0.36412139[0m [38;2;32;148;243m0.46241614[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07102859[0m [38;2;32;148;243m-0.02479802[0m [38;2;32;148;243m-0.08795881[0m [38;2;32;148;243m-0.11699536[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14185308[0m [38;2;32;148;243m-0.04897791[0m [38;2;32;148;243m-0.17463513[0m [38;2;32;148;243m-0.23353095[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21438792[0m [38;2;32;148;243m-0.07145152[0m [38;2;32;148;243m-0.26172765[0m [38;2;32;148;243m-0.34976972[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28795822[0m [38;2;32;148;243m-0.09329237[0m [38;2;32;148;243m-0.34998644[0m [38;2;32;148;243m-0.46605898[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21401412[0m [38;2;32;148;243m-0.06916024[0m [38;2;32;148;243m-0.26017036[0m [38;2;32;148;243m-0.34989792[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.13950632[0m [38;2;32;148;243m-0.04182621[0m [38;2;32;148;243m-0.16962171[0m [38;2;32;148;243m-0.23420681[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.06515899[0m [38;2;32;148;243m-0.01168603[0m [38;2;32;148;243m-0.07846871[0m [38;2;32;148;243m-0.11885351[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00773165[0m  [38;2;32;148;243m0.02141864[0m  [38;2;32;148;243m0.01413494[0m [38;2;32;148;243m-0.00364284[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m.        [38;2;32;148;243m1[0m.        [38;2;32;148;243m1[0m.        [38;2;32;148;243m0.4480012[0m[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00773165[0m  [38;2;32;148;243m0.02141864[0m  [38;2;32;148;243m0.01413494[0m [38;2;32;148;243m-0[0m.        [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m0[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00176544[0m  [38;2;32;148;243m0.00272047[0m [38;2;32;148;243m-0.00086076[0m  [38;2;32;148;243m0.00036219[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00437903[0m [38;2;32;148;243m-0.00162554[0m [38;2;32;148;243m-0.00408489[0m  [38;2;32;148;243m0.00312903[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.12778234[0m [38;2;32;148;243m-0.047434[0m   [38;2;32;148;243m-0.11919906[0m  [38;2;32;148;243m0.09130668[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00039654[0m [38;2;32;148;243m0.00091485[0m [38;2;32;148;243m0.003272[0m   [38;2;32;148;243m0[0m.        [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.35893336e-05[0m [38;2;32;148;243m3.13513107e-05[0m [38;2;32;148;243m1.12129821e-04[0m [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-4.9186839696357544e-05[0m
>     [38;2;105;105;105m[07/10/23 07:57:02][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m41[0m
>     [38;2;105;105;105m[07/10/23 07:57:03][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m42[0m
>     [38;2;105;105;105m[07/10/23 07:57:03][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m43[0m
>     [38;2;105;105;105m[07/10/23 07:57:04][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m44[0m
>     [38;2;105;105;105m[07/10/23 07:57:04][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m45[0m
>     [38;2;105;105;105m[07/10/23 07:57:04][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m-59.62601426[0m  [38;2;32;148;243m-58.699203[0m     [38;2;32;148;243m59.46849932[0m [38;2;32;148;243m-172.36747803[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.5591947[0m   [38;2;32;148;243m-58.72665761[0m   [38;2;32;148;243m59.54762304[0m [38;2;32;148;243m-172.43679004[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.478034[0m    [38;2;32;148;243m-58.73975672[0m   [38;2;32;148;243m59.6428672[0m  [38;2;32;148;243m-172.48654935[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.38301564[0m  [38;2;32;148;243m-58.73852857[0m   [38;2;32;148;243m59.75374459[0m [38;2;32;148;243m-172.51625033[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.27389385[0m  [38;2;32;148;243m-58.72284798[0m   [38;2;32;148;243m59.88025131[0m [38;2;32;148;243m-172.52596124[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.11103216[0m  [38;2;32;148;243m-58.75758363[0m   [38;2;32;148;243m59.96074514[0m [38;2;32;148;243m-172.40155126[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.93446541[0m  [38;2;32;148;243m-58.777547[0m     [38;2;32;148;243m60.05743232[0m [38;2;32;148;243m-172.25787331[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.74060986[0m  [38;2;32;148;243m-58.78243594[0m   [38;2;32;148;243m60.17052679[0m [38;2;32;148;243m-172.09459919[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.52728845[0m  [38;2;32;148;243m-58.7714395[0m    [38;2;32;148;243m60.30047494[0m [38;2;32;148;243m-171.91225775[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m-59.62601426[0m  [38;2;32;148;243m-58.699203[0m     [38;2;32;148;243m59.46849932[0m [38;2;32;148;243m-172.36747803[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.48467085[0m  [38;2;32;148;243m-58.69961143[0m   [38;2;32;148;243m59.63720058[0m [38;2;32;148;243m-172.32092147[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.32967507[0m  [38;2;32;148;243m-58.68565043[0m   [38;2;32;148;243m59.82288137[0m [38;2;32;148;243m-172.25539508[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-59.16351451[0m  [38;2;32;148;243m-58.6582891[0m    [38;2;32;148;243m60.02368863[0m [38;2;32;148;243m-172.17018998[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.98479033[0m  [38;2;32;148;243m-58.61718232[0m   [38;2;32;148;243m60.23996164[0m [38;2;32;148;243m-172.06529574[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.89165025[0m  [38;2;32;148;243m-58.67767335[0m   [38;2;32;148;243m60.22960664[0m [38;2;32;148;243m-172.05530265[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.78459027[0m  [38;2;32;148;243m-58.72376757[0m   [38;2;32;148;243m60.23384326[0m [38;2;32;148;243m-172.02592269[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.66232782[0m  [38;2;32;148;243m-58.75541252[0m   [38;2;32;148;243m60.2532672[0m  [38;2;32;148;243m-171.97623719[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-58.52427607[0m  [38;2;32;148;243m-58.77275329[0m   [38;2;32;148;243m60.28906012[0m [38;2;32;148;243m-171.9071523[0m [1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07452385[0m [38;2;32;148;243m-0.02704618[0m [38;2;32;148;243m-0.08957755[0m [38;2;32;148;243m-0.11586856[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14835894[0m [38;2;32;148;243m-0.05410629[0m [38;2;32;148;243m-0.18001417[0m [38;2;32;148;243m-0.23115426[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21950113[0m [38;2;32;148;243m-0.08023947[0m [38;2;32;148;243m-0.26994404[0m [38;2;32;148;243m-0.34606035[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28910352[0m [38;2;32;148;243m-0.10566565[0m [38;2;32;148;243m-0.35971033[0m [38;2;32;148;243m-0.46066551[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21938191[0m [38;2;32;148;243m-0.07991027[0m [38;2;32;148;243m-0.2688615[0m  [38;2;32;148;243m-0.34624861[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14987513[0m [38;2;32;148;243m-0.05377943[0m [38;2;32;148;243m-0.17641094[0m [38;2;32;148;243m-0.23195062[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07828204[0m [38;2;32;148;243m-0.02702341[0m [38;2;32;148;243m-0.08274041[0m [38;2;32;148;243m-0.118362[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00301238[0m  [38;2;32;148;243m0.0013138[0m   [38;2;32;148;243m0.01141482[0m [38;2;32;148;243m-0.00510544[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07452385[0m [38;2;32;148;243m-0.02704618[0m [38;2;32;148;243m-0.08957755[0m [38;2;32;148;243m-0.11586856[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14835894[0m [38;2;32;148;243m-0.05410629[0m [38;2;32;148;243m-0.18001417[0m [38;2;32;148;243m-0.23115426[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21950113[0m [38;2;32;148;243m-0.08023947[0m [38;2;32;148;243m-0.26994404[0m [38;2;32;148;243m-0.34606035[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28910352[0m [38;2;32;148;243m-0.10566565[0m [38;2;32;148;243m-0.35971033[0m [38;2;32;148;243m-0.46066551[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.06972161[0m [38;2;32;148;243m0.02575538[0m [38;2;32;148;243m0.09084883[0m [38;2;32;148;243m0.1144169[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.13922839[0m [38;2;32;148;243m0.05188622[0m [38;2;32;148;243m0.18329939[0m [38;2;32;148;243m0.22871488[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21082148[0m [38;2;32;148;243m0.07864224[0m [38;2;32;148;243m0.27696992[0m [38;2;32;148;243m0.3423035[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.28609114[0m [38;2;32;148;243m0.10697945[0m [38;2;32;148;243m0.37112515[0m [38;2;32;148;243m0.45556006[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07452385[0m [38;2;32;148;243m-0.02704618[0m [38;2;32;148;243m-0.08957755[0m [38;2;32;148;243m-0.11586856[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14835894[0m [38;2;32;148;243m-0.05410629[0m [38;2;32;148;243m-0.18001417[0m [38;2;32;148;243m-0.23115426[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21950113[0m [38;2;32;148;243m-0.08023947[0m [38;2;32;148;243m-0.26994404[0m [38;2;32;148;243m-0.34606035[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.28910352[0m [38;2;32;148;243m-0.10566565[0m [38;2;32;148;243m-0.35971033[0m [38;2;32;148;243m-0.46066551[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21938191[0m [38;2;32;148;243m-0.07991027[0m [38;2;32;148;243m-0.2688615[0m  [38;2;32;148;243m-0.34624861[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.14987513[0m [38;2;32;148;243m-0.05377943[0m [38;2;32;148;243m-0.17641094[0m [38;2;32;148;243m-0.23195062[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.07828204[0m [38;2;32;148;243m-0.02702341[0m [38;2;32;148;243m-0.08274041[0m [38;2;32;148;243m-0.118362[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00301238[0m  [38;2;32;148;243m0.0013138[0m   [38;2;32;148;243m0.01141482[0m [38;2;32;148;243m-0.00510544[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.33229299[0m [38;2;32;148;243m1[0m.         [38;2;32;148;243m0.44018473[0m [38;2;32;148;243m0.63107805[0m[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0[0m.          [38;2;32;148;243m0.0013138[0m   [38;2;32;148;243m0.01141482[0m [38;2;32;148;243m-0.00510544[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m0[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00092452[0m  [38;2;32;148;243m0.00331762[0m [38;2;32;148;243m-0.00037819[0m  [38;2;32;148;243m0.00054205[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00458282[0m [38;2;32;148;243m-0.00182782[0m [38;2;32;148;243m-0.0036927[0m   [38;2;32;148;243m0.00349978[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.13372905[0m [38;2;32;148;243m-0.05333676[0m [38;2;32;148;243m-0.10775497[0m  [38;2;32;148;243m0.10212541[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0.00644556[0m [38;2;32;148;243m0.00171661[0m [38;2;32;148;243m0.00336905[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00000000e+00[0m [38;2;32;148;243m2.20885899e-04[0m [38;2;32;148;243m5.88271294e-05[0m [38;2;32;148;243m1.15455355e-04[0m[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-3.394048083099028e-05[0m
>     [38;2;105;105;105m[07/10/23 07:57:04][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m46[0m
>     [38;2;105;105;105m[07/10/23 07:57:05][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m47[0m
>     [38;2;105;105;105m[07/10/23 07:57:05][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m48[0m
>     [38;2;105;105;105m[07/10/23 07:57:06][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mexperiment.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m119[0m[2;38;2;144;144;144m][0m - STEP: [38;2;32;148;243m49[0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:06] </span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">checkSU</span><span style="font-weight: bold">(</span>x_eval<span style="font-weight: bold">)</span>: <span style="font-weight: bold">(</span>tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">3.845e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.993e-16</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">3</span><span style="color: #2094f3; text-decoration-color: #2094f3">.923e-16</span> <span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">6</span><span style="color: #2094f3; text-decoration-color: #2094f3">.436e-18</span> <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">3.955e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.845e-16</span>, 
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #2094f3; text-decoration-color: #2094f3">3.901e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.993e-16</span><span style="font-weight: bold">]</span>, tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">1.405e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.560e-15</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">1</span><span style="color: #2094f3; text-decoration-color: #2094f3">.496e-15</span> <span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">6</span><span style="color: #2094f3; text-decoration-color: #2094f3">.503e-17</span> <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">1.560e-15</span>,       
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #2094f3; text-decoration-color: #2094f3">1.405e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.511e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.507e-15</span><span style="font-weight: bold">])</span>                                                                       
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving energy to plots-4dSU3/Evaluate                                                                   
> </pre>
>
> ![svg](output_13_3.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving logprob to plots-4dSU3/Evaluate                                                                  
> </pre>
>
> ![svg](output_13_5.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:07] </span>Saving logdet to plots-4dSU3/Evaluate                                                                   
> </pre>
>
> ![svg](output_13_7.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sldf to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_9.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sldb to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_11.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sld to plots-4dSU3/Evaluate                                                                      
> </pre>
>
> ![svg](output_13_13.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:08] </span>Saving xeps to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_15.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving veps to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_17.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving acc to plots-4dSU3/Evaluate                                                                      
> </pre>
>
> ![svg](output_13_19.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sumlogdet to plots-4dSU3/Evaluate                                                                
> </pre>
>
> ![svg](output_13_21.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:09] </span>Saving beta to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_23.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving acc_mask to plots-4dSU3/Evaluate                                                                 
> </pre>
>
> ![svg](output_13_25.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving plaqs to plots-4dSU3/Evaluate                                                                    
> </pre>
>
> ![svg](output_13_27.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sinQ to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_29.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:10] </span>Saving intQ to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_31.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving dQint to plots-4dSU3/Evaluate                                                                    
> </pre>
>
> ![svg](output_13_33.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving dQsin to plots-4dSU3/Evaluate                                                                    
> </pre>
>
> ![svg](output_13_35.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:11] </span>Saving loss to plots-4dSU3/Evaluate                                                                     
> </pre>
>
> ![svg](output_13_37.svg)
>
> </div>

## Training

``` python
history = {}
x = state.x
for step in range(50):
    console.log(f'TRAIN STEP: {step}')
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
console.log(f"checkSU(x_train): {g.checkSU(x)}")
plot_metrics(history, title='train', marker='.')
```

> [!TIP]
>
> ### <span class="dim-text">`output`:</span>
>
> <div class="cell-output cell-output-display">
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:15] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">0</span>                                                                                           
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:16] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">1</span>                                                                                           
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">2</span>                                                                                           
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:17][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m281.36521385[0m [38;2;32;148;243m-105.97415137[0m  [38;2;32;148;243m-69.97486138[0m  [38;2;32;148;243m-68.7094104[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m263.24054833[0m [38;2;32;148;243m-109.07253574[0m  [38;2;32;148;243m-86.39965892[0m  [38;2;32;148;243m-81.32577003[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.1725606[0m  [38;2;32;148;243m-109.04339532[0m  [38;2;32;148;243m-86.52495783[0m  [38;2;32;148;243m-81.36519938[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.12190109[0m [38;2;32;148;243m-109.00002499[0m  [38;2;32;148;243m-86.63542547[0m  [38;2;32;148;243m-81.39024754[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.08826659[0m [38;2;32;148;243m-108.94247239[0m  [38;2;32;148;243m-86.72756853[0m  [38;2;32;148;243m-81.40051327[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.15071961[0m [38;2;32;148;243m-108.77269274[0m  [38;2;32;148;243m-86.75297969[0m  [38;2;32;148;243m-81.33529329[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.23101691[0m [38;2;32;148;243m-108.59060157[0m  [38;2;32;148;243m-86.762429[0m    [38;2;32;148;243m-81.25362457[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.32955656[0m [38;2;32;148;243m-108.39687575[0m  [38;2;32;148;243m-86.75589511[0m  [38;2;32;148;243m-81.15387713[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.44688414[0m [38;2;32;148;243m-108.19113989[0m  [38;2;32;148;243m-86.73311909[0m  [38;2;32;148;243m-81.03775212[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m281.36521385[0m [38;2;32;148;243m-105.97415137[0m  [38;2;32;148;243m-69.97486138[0m  [38;2;32;148;243m-68.7094104[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m263.34046617[0m [38;2;32;148;243m-109.01072961[0m  [38;2;32;148;243m-86.38264168[0m  [38;2;32;148;243m-81.24452066[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.37307865[0m [38;2;32;148;243m-108.92456325[0m  [38;2;32;148;243m-86.49242126[0m  [38;2;32;148;243m-81.20801101[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.42366401[0m [38;2;32;148;243m-108.82480411[0m  [38;2;32;148;243m-86.58754075[0m  [38;2;32;148;243m-81.15657432[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.49344316[0m [38;2;32;148;243m-108.71126581[0m  [38;2;32;148;243m-86.66610532[0m  [38;2;32;148;243m-81.09002842[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.45194401[0m [38;2;32;148;243m-108.59723981[0m  [38;2;32;148;243m-86.7045229[0m   [38;2;32;148;243m-81.10170009[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.42889533[0m [38;2;32;148;243m-108.46916199[0m  [38;2;32;148;243m-86.72820528[0m  [38;2;32;148;243m-81.098189[0m  [1m][0m
>      [1m[[0m [38;2;32;148;243m263.42369473[0m [38;2;32;148;243m-108.32747357[0m  [38;2;32;148;243m-86.73779558[0m  [38;2;32;148;243m-81.07912734[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m263.43579227[0m [38;2;32;148;243m-108.17161087[0m  [38;2;32;148;243m-86.73344347[0m  [38;2;32;148;243m-81.04469161[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.99178394e-02[0m [38;2;32;148;243m-6.18061332e-02[0m [38;2;32;148;243m-1.70172453e-02[0m [38;2;32;148;243m-8.12493680e-02[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.00518048e-01[0m [38;2;32;148;243m-1.18832073e-01[0m [38;2;32;148;243m-3.25365743e-02[0m [38;2;32;148;243m-1.57188371e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.01762924e-01[0m [38;2;32;148;243m-1.75220878e-01[0m [38;2;32;148;243m-4.78847161e-02[0m [38;2;32;148;243m-2.33673222e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-4.05176570e-01[0m [38;2;32;148;243m-2.31206581e-01[0m [38;2;32;148;243m-6.14632088e-02[0m [38;2;32;148;243m-3.10484854e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.01224398e-01[0m [38;2;32;148;243m-1.75452932e-01[0m [38;2;32;148;243m-4.84567891e-02[0m [38;2;32;148;243m-2.33593200e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.97878419e-01[0m [38;2;32;148;243m-1.21439581e-01[0m [38;2;32;148;243m-3.42237191e-02[0m [38;2;32;148;243m-1.55435574e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.41381732e-02[0m [38;2;32;148;243m-6.94021838e-02[0m [38;2;32;148;243m-1.80995342e-02[0m [38;2;32;148;243m-7.47497902e-02[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m1.10918724e-02[0m [38;2;32;148;243m-1.95290252e-02[0m  [38;2;32;148;243m3.24386329e-04[0m  [38;2;32;148;243m6.93949633e-03[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09991784[0m [38;2;32;148;243m-0.06180613[0m [38;2;32;148;243m-0.01701725[0m [38;2;32;148;243m-0.08124937[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20051805[0m [38;2;32;148;243m-0.11883207[0m [38;2;32;148;243m-0.03253657[0m [38;2;32;148;243m-0.15718837[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30176292[0m [38;2;32;148;243m-0.17522088[0m [38;2;32;148;243m-0.04788472[0m [38;2;32;148;243m-0.23367322[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40517657[0m [38;2;32;148;243m-0.23120658[0m [38;2;32;148;243m-0.06146321[0m [38;2;32;148;243m-0.31048485[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10395217[0m [38;2;32;148;243m0.05575365[0m [38;2;32;148;243m0.01300642[0m [38;2;32;148;243m0.07689165[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.20729815[0m [38;2;32;148;243m0.109767[0m   [38;2;32;148;243m0.02723949[0m [38;2;32;148;243m0.15504928[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.3110384[0m  [38;2;32;148;243m0.1618044[0m  [38;2;32;148;243m0.04336367[0m [38;2;32;148;243m0.23573506[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.41626844[0m [38;2;32;148;243m0.21167756[0m [38;2;32;148;243m0.0617876[0m  [38;2;32;148;243m0.31742435[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.99178394e-02[0m [38;2;32;148;243m-6.18061332e-02[0m [38;2;32;148;243m-1.70172453e-02[0m [38;2;32;148;243m-8.12493680e-02[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.00518048e-01[0m [38;2;32;148;243m-1.18832073e-01[0m [38;2;32;148;243m-3.25365743e-02[0m [38;2;32;148;243m-1.57188371e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.01762924e-01[0m [38;2;32;148;243m-1.75220878e-01[0m [38;2;32;148;243m-4.78847161e-02[0m [38;2;32;148;243m-2.33673222e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-4.05176570e-01[0m [38;2;32;148;243m-2.31206581e-01[0m [38;2;32;148;243m-6.14632088e-02[0m [38;2;32;148;243m-3.10484854e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.01224398e-01[0m [38;2;32;148;243m-1.75452932e-01[0m [38;2;32;148;243m-4.84567891e-02[0m [38;2;32;148;243m-2.33593200e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.97878419e-01[0m [38;2;32;148;243m-1.21439581e-01[0m [38;2;32;148;243m-3.42237191e-02[0m [38;2;32;148;243m-1.55435574e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.41381732e-02[0m [38;2;32;148;243m-6.94021838e-02[0m [38;2;32;148;243m-1.80995342e-02[0m [38;2;32;148;243m-7.47497902e-02[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m1.10918724e-02[0m [38;2;32;148;243m-1.95290252e-02[0m  [38;2;32;148;243m3.24386329e-04[0m  [38;2;32;148;243m6.93949633e-03[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01109187[0m [38;2;32;148;243m-0.01952903[0m  [38;2;32;148;243m0.00032439[0m  [38;2;32;148;243m0.0069395[0m [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.214976548064738e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.0061887[0m   [38;2;32;148;243m0.00500194[0m  [38;2;32;148;243m0.00397052[0m  [38;2;32;148;243m0.00286354[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00160074[0m  [38;2;32;148;243m0.00051351[0m [38;2;32;148;243m-0.00282033[0m [38;2;32;148;243m-0.00232394[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0467105[0m   [38;2;32;148;243m0.01498444[0m [38;2;32;148;243m-0.08229858[0m [38;2;32;148;243m-0.06781372[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00360113[0m [38;2;32;148;243m0.00411262[0m [38;2;32;148;243m0.00122385[0m [38;2;32;148;243m0.01062494[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.23408791e-04[0m [38;2;32;148;243m1.40937158e-04[0m [38;2;32;148;243m4.19407924e-05[0m [38;2;32;148;243m3.64110788e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:17] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">3</span>                                                                                           
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">4</span>                                                                                           
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:18][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m135.11825502[0m [38;2;32;148;243m-185.23125091[0m   [38;2;32;148;243m39.49801865[0m  [38;2;32;148;243m131.48643148[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.29012806[0m [38;2;32;148;243m-187.76677412[0m   [38;2;32;148;243m21.57755422[0m  [38;2;32;148;243m118.32075786[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.24121464[0m [38;2;32;148;243m-187.81407012[0m   [38;2;32;148;243m21.54918557[0m  [38;2;32;148;243m118.22618093[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.21203826[0m [38;2;32;148;243m-187.84774093[0m   [38;2;32;148;243m21.53445154[0m  [38;2;32;148;243m118.14697838[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.20235487[0m [38;2;32;148;243m-187.86847227[0m   [38;2;32;148;243m21.53313956[0m  [38;2;32;148;243m118.08177774[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.33755546[0m [38;2;32;148;243m-187.73695847[0m   [38;2;32;148;243m21.5237841[0m   [38;2;32;148;243m118.21081333[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.49320029[0m [38;2;32;148;243m-187.59196912[0m   [38;2;32;148;243m21.52970039[0m  [38;2;32;148;243m118.35271288[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.67025311[0m [38;2;32;148;243m-187.43381451[0m   [38;2;32;148;243m21.54991672[0m  [38;2;32;148;243m118.5124691[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m116.86681924[0m [38;2;32;148;243m-187.26307202[0m   [38;2;32;148;243m21.58393788[0m  [38;2;32;148;243m118.68765025[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m135.11825502[0m [38;2;32;148;243m-185.23125091[0m   [38;2;32;148;243m39.49801865[0m  [38;2;32;148;243m131.48643148[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.39238634[0m [38;2;32;148;243m-187.712692[0m     [38;2;32;148;243m21.61699055[0m  [38;2;32;148;243m118.39725088[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.45523291[0m [38;2;32;148;243m-187.70870488[0m   [38;2;32;148;243m21.64805552[0m  [38;2;32;148;243m118.37787194[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.53809899[0m [38;2;32;148;243m-187.69063165[0m   [38;2;32;148;243m21.69314918[0m  [38;2;32;148;243m118.37350562[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.6405493[0m  [38;2;32;148;243m-187.65779592[0m   [38;2;32;148;243m21.75214543[0m  [38;2;32;148;243m118.38564832[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.66329968[0m [38;2;32;148;243m-187.58144768[0m   [38;2;32;148;243m21.68175728[0m  [38;2;32;148;243m118.43634456[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.70589156[0m [38;2;32;148;243m-187.49115407[0m   [38;2;32;148;243m21.62822818[0m  [38;2;32;148;243m118.50107276[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m116.76922139[0m [38;2;32;148;243m-187.38620402[0m   [38;2;32;148;243m21.58955162[0m  [38;2;32;148;243m118.5821923[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m116.85193341[0m [38;2;32;148;243m-187.26621096[0m   [38;2;32;148;243m21.56478423[0m  [38;2;32;148;243m118.67845245[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10225828[0m [38;2;32;148;243m-0.05408212[0m [38;2;32;148;243m-0.03943634[0m [38;2;32;148;243m-0.07649302[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21401826[0m [38;2;32;148;243m-0.10536524[0m [38;2;32;148;243m-0.09886995[0m [38;2;32;148;243m-0.151691[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32606074[0m [38;2;32;148;243m-0.15710928[0m [38;2;32;148;243m-0.15869764[0m [38;2;32;148;243m-0.22652724[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43819443[0m [38;2;32;148;243m-0.21067635[0m [38;2;32;148;243m-0.21900587[0m [38;2;32;148;243m-0.30387058[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32574422[0m [38;2;32;148;243m-0.15551078[0m [38;2;32;148;243m-0.15797318[0m [38;2;32;148;243m-0.22553124[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21269127[0m [38;2;32;148;243m-0.10081506[0m [38;2;32;148;243m-0.09852779[0m [38;2;32;148;243m-0.14835988[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09896827[0m [38;2;32;148;243m-0.04761049[0m [38;2;32;148;243m-0.0396349[0m  [38;2;32;148;243m-0.0697232[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0.01488583[0m  [38;2;32;148;243m0.00313894[0m  [38;2;32;148;243m0.01915366[0m  [38;2;32;148;243m0.00919779[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10225828[0m [38;2;32;148;243m-0.05408212[0m [38;2;32;148;243m-0.03943634[0m [38;2;32;148;243m-0.07649302[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21401826[0m [38;2;32;148;243m-0.10536524[0m [38;2;32;148;243m-0.09886995[0m [38;2;32;148;243m-0.151691[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32606074[0m [38;2;32;148;243m-0.15710928[0m [38;2;32;148;243m-0.15869764[0m [38;2;32;148;243m-0.22652724[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43819443[0m [38;2;32;148;243m-0.21067635[0m [38;2;32;148;243m-0.21900587[0m [38;2;32;148;243m-0.30387058[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.11245021[0m [38;2;32;148;243m0.05516556[0m [38;2;32;148;243m0.06103269[0m [38;2;32;148;243m0.07833934[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.22550316[0m [38;2;32;148;243m0.10986129[0m [38;2;32;148;243m0.12047808[0m [38;2;32;148;243m0.1555107[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.33922616[0m [38;2;32;148;243m0.16306585[0m [38;2;32;148;243m0.17937097[0m [38;2;32;148;243m0.23414738[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.45308026[0m [38;2;32;148;243m0.21381529[0m [38;2;32;148;243m0.23815953[0m [38;2;32;148;243m0.31306837[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10225828[0m [38;2;32;148;243m-0.05408212[0m [38;2;32;148;243m-0.03943634[0m [38;2;32;148;243m-0.07649302[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21401826[0m [38;2;32;148;243m-0.10536524[0m [38;2;32;148;243m-0.09886995[0m [38;2;32;148;243m-0.151691[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32606074[0m [38;2;32;148;243m-0.15710928[0m [38;2;32;148;243m-0.15869764[0m [38;2;32;148;243m-0.22652724[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43819443[0m [38;2;32;148;243m-0.21067635[0m [38;2;32;148;243m-0.21900587[0m [38;2;32;148;243m-0.30387058[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32574422[0m [38;2;32;148;243m-0.15551078[0m [38;2;32;148;243m-0.15797318[0m [38;2;32;148;243m-0.22553124[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21269127[0m [38;2;32;148;243m-0.10081506[0m [38;2;32;148;243m-0.09852779[0m [38;2;32;148;243m-0.14835988[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09896827[0m [38;2;32;148;243m-0.04761049[0m [38;2;32;148;243m-0.0396349[0m  [38;2;32;148;243m-0.0697232[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0.01488583[0m  [38;2;32;148;243m0.00313894[0m  [38;2;32;148;243m0.01915366[0m  [38;2;32;148;243m0.00919779[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.01488583[0m [38;2;32;148;243m0.00313894[0m [38;2;32;148;243m0.01915366[0m [38;2;32;148;243m0.00919779[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.311273598207552e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00625617[0m  [38;2;32;148;243m0.00515697[0m  [38;2;32;148;243m0.00406816[0m  [38;2;32;148;243m0.00275435[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00169669[0m  [38;2;32;148;243m0.00027787[0m [38;2;32;148;243m-0.00307472[0m [38;2;32;148;243m-0.00243068[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04951016[0m  [38;2;32;148;243m0.00810845[0m [38;2;32;148;243m-0.08972183[0m [38;2;32;148;243m-0.0709284[0m [1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.0091121[0m  [38;2;32;148;243m0.00485531[0m [38;2;32;148;243m0.00373837[0m [38;2;32;148;243m0.00530226[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00031227[0m [38;2;32;148;243m0.00016639[0m [38;2;32;148;243m0.00012811[0m [38;2;32;148;243m0.00018171[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:18] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">5</span>                                                                                           
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">6</span>                                                                                           
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:18][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m228.01629364[0m [38;2;32;148;243m-281.98565237[0m  [38;2;32;148;243m-20.12491938[0m  [38;2;32;148;243m-81.93169814[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.54372772[0m [38;2;32;148;243m-284.81511328[0m  [38;2;32;148;243m-37.93562912[0m  [38;2;32;148;243m-94.82221044[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.53314206[0m [38;2;32;148;243m-284.86321231[0m  [38;2;32;148;243m-37.98213952[0m  [38;2;32;148;243m-94.7657871[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m209.53988816[0m [38;2;32;148;243m-284.89900259[0m  [38;2;32;148;243m-38.01533459[0m  [38;2;32;148;243m-94.6929395[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m209.56277464[0m [38;2;32;148;243m-284.92278355[0m  [38;2;32;148;243m-38.03459123[0m  [38;2;32;148;243m-94.60412808[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.52202739[0m [38;2;32;148;243m-284.81794281[0m  [38;2;32;148;243m-37.98911718[0m  [38;2;32;148;243m-94.44784165[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.49797583[0m [38;2;32;148;243m-284.69947243[0m  [38;2;32;148;243m-37.9326838[0m   [38;2;32;148;243m-94.27544262[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.49108176[0m [38;2;32;148;243m-284.56770054[0m  [38;2;32;148;243m-37.864695[0m    [38;2;32;148;243m-94.08767484[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.50091155[0m [38;2;32;148;243m-284.42249184[0m  [38;2;32;148;243m-37.78261324[0m  [38;2;32;148;243m-93.88473302[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m228.01629364[0m [38;2;32;148;243m-281.98565237[0m  [38;2;32;148;243m-20.12491938[0m  [38;2;32;148;243m-81.93169814[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.64717122[0m [38;2;32;148;243m-284.76978578[0m  [38;2;32;148;243m-37.89875392[0m  [38;2;32;148;243m-94.73644574[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.73537504[0m [38;2;32;148;243m-284.77002068[0m  [38;2;32;148;243m-37.89616835[0m  [38;2;32;148;243m-94.59029472[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.84174293[0m [38;2;32;148;243m-284.75652536[0m  [38;2;32;148;243m-37.88147942[0m  [38;2;32;148;243m-94.42815844[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.962672[0m   [38;2;32;148;243m-284.7292213[0m   [38;2;32;148;243m-37.85268786[0m  [38;2;32;148;243m-94.25053909[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.82588679[0m [38;2;32;148;243m-284.6766064[0m   [38;2;32;148;243m-37.85487886[0m  [38;2;32;148;243m-94.18370178[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.70632398[0m [38;2;32;148;243m-284.6116521[0m   [38;2;32;148;243m-37.84305335[0m  [38;2;32;148;243m-94.10091561[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.60393282[0m [38;2;32;148;243m-284.53535682[0m  [38;2;32;148;243m-37.81768024[0m  [38;2;32;148;243m-94.00238516[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m209.51820993[0m [38;2;32;148;243m-284.44869923[0m  [38;2;32;148;243m-37.7788019[0m   [38;2;32;148;243m-93.88832557[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1034435[0m  [38;2;32;148;243m-0.0453275[0m  [38;2;32;148;243m-0.0368752[0m  [38;2;32;148;243m-0.0857647[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20223298[0m [38;2;32;148;243m-0.09319163[0m [38;2;32;148;243m-0.08597117[0m [38;2;32;148;243m-0.17549238[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30185477[0m [38;2;32;148;243m-0.14247724[0m [38;2;32;148;243m-0.13385516[0m [38;2;32;148;243m-0.26478106[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39989736[0m [38;2;32;148;243m-0.19356225[0m [38;2;32;148;243m-0.18190337[0m [38;2;32;148;243m-0.35358898[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3038594[0m  [38;2;32;148;243m-0.14133641[0m [38;2;32;148;243m-0.13423832[0m [38;2;32;148;243m-0.26413986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20834815[0m [38;2;32;148;243m-0.08782033[0m [38;2;32;148;243m-0.08963045[0m [38;2;32;148;243m-0.174527[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.11285106[0m [38;2;32;148;243m-0.03234373[0m [38;2;32;148;243m-0.04701475[0m [38;2;32;148;243m-0.08528968[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.01729838[0m  [38;2;32;148;243m0.02620739[0m [38;2;32;148;243m-0.00381135[0m  [38;2;32;148;243m0.00359256[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1034435[0m  [38;2;32;148;243m-0.0453275[0m  [38;2;32;148;243m-0.0368752[0m  [38;2;32;148;243m-0.0857647[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20223298[0m [38;2;32;148;243m-0.09319163[0m [38;2;32;148;243m-0.08597117[0m [38;2;32;148;243m-0.17549238[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30185477[0m [38;2;32;148;243m-0.14247724[0m [38;2;32;148;243m-0.13385516[0m [38;2;32;148;243m-0.26478106[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39989736[0m [38;2;32;148;243m-0.19356225[0m [38;2;32;148;243m-0.18190337[0m [38;2;32;148;243m-0.35358898[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.09603796[0m [38;2;32;148;243m0.05222585[0m [38;2;32;148;243m0.04766505[0m [38;2;32;148;243m0.08944912[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.19154921[0m [38;2;32;148;243m0.10574193[0m [38;2;32;148;243m0.09227292[0m [38;2;32;148;243m0.17906198[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.2870463[0m  [38;2;32;148;243m0.16121852[0m [38;2;32;148;243m0.13488861[0m [38;2;32;148;243m0.2682993[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.38259897[0m [38;2;32;148;243m0.21976964[0m [38;2;32;148;243m0.17809202[0m [38;2;32;148;243m0.35718154[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1034435[0m  [38;2;32;148;243m-0.0453275[0m  [38;2;32;148;243m-0.0368752[0m  [38;2;32;148;243m-0.0857647[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20223298[0m [38;2;32;148;243m-0.09319163[0m [38;2;32;148;243m-0.08597117[0m [38;2;32;148;243m-0.17549238[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30185477[0m [38;2;32;148;243m-0.14247724[0m [38;2;32;148;243m-0.13385516[0m [38;2;32;148;243m-0.26478106[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39989736[0m [38;2;32;148;243m-0.19356225[0m [38;2;32;148;243m-0.18190337[0m [38;2;32;148;243m-0.35358898[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3038594[0m  [38;2;32;148;243m-0.14133641[0m [38;2;32;148;243m-0.13423832[0m [38;2;32;148;243m-0.26413986[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20834815[0m [38;2;32;148;243m-0.08782033[0m [38;2;32;148;243m-0.08963045[0m [38;2;32;148;243m-0.174527[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.11285106[0m [38;2;32;148;243m-0.03234373[0m [38;2;32;148;243m-0.04701475[0m [38;2;32;148;243m-0.08528968[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.01729838[0m  [38;2;32;148;243m0.02620739[0m [38;2;32;148;243m-0.00381135[0m  [38;2;32;148;243m0.00359256[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.01729838[0m  [38;2;32;148;243m0.02620739[0m [38;2;32;148;243m-0.00381135[0m  [38;2;32;148;243m0.00359256[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.28664314669938e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00595627[0m  [38;2;32;148;243m0.00563947[0m  [38;2;32;148;243m0.00427224[0m  [38;2;32;148;243m0.00266021[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0015684[0m   [38;2;32;148;243m0.000394[0m   [38;2;32;148;243m-0.00288902[0m [38;2;32;148;243m-0.00230002[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04576675[0m  [38;2;32;148;243m0.01149709[0m [38;2;32;148;243m-0.08430301[0m [38;2;32;148;243m-0.06711567[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00569303[0m [38;2;32;148;243m0.00018414[0m [38;2;32;148;243m0.00430204[0m [38;2;32;148;243m0.0048331[0m [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.95097159e-04[0m [38;2;32;148;243m6.31025768e-06[0m [38;2;32;148;243m1.47428747e-04[0m [38;2;32;148;243m1.65627571e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">7</span>                                                                                           
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:19] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">8</span>                                                                                           
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:19][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m58.07198815[0m [38;2;32;148;243m-254.25276437[0m [38;2;32;148;243m-172.2577486[0m    [38;2;32;148;243m29.085534[0m  [1m][0m
>      [1m[[0m  [38;2;32;148;243m38.58967365[0m [38;2;32;148;243m-256.77756509[0m [38;2;32;148;243m-189.63209473[0m   [38;2;32;148;243m16.9107765[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m38.54131984[0m [38;2;32;148;243m-256.74101555[0m [38;2;32;148;243m-189.65157019[0m   [38;2;32;148;243m16.86084166[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.51063821[0m [38;2;32;148;243m-256.69100156[0m [38;2;32;148;243m-189.65346392[0m   [38;2;32;148;243m16.82947871[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.49779564[0m [38;2;32;148;243m-256.63323532[0m [38;2;32;148;243m-189.63822452[0m   [38;2;32;148;243m16.81279717[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.44790192[0m [38;2;32;148;243m-256.74596217[0m [38;2;32;148;243m-189.52847699[0m   [38;2;32;148;243m16.85991159[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.41672204[0m [38;2;32;148;243m-256.84462705[0m [38;2;32;148;243m-189.40387487[0m   [38;2;32;148;243m16.9229498[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m38.40415924[0m [38;2;32;148;243m-256.92936903[0m [38;2;32;148;243m-189.26495267[0m   [38;2;32;148;243m17.00162406[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.40968237[0m [38;2;32;148;243m-257.00031104[0m [38;2;32;148;243m-189.11191882[0m   [38;2;32;148;243m17.09386957[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m58.07198815[0m [38;2;32;148;243m-254.25276437[0m [38;2;32;148;243m-172.2577486[0m    [38;2;32;148;243m29.085534[0m  [1m][0m
>      [1m[[0m  [38;2;32;148;243m38.69108049[0m [38;2;32;148;243m-256.73219519[0m [38;2;32;148;243m-189.60853741[0m   [38;2;32;148;243m16.99823438[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.74834371[0m [38;2;32;148;243m-256.65132718[0m [38;2;32;148;243m-189.60253545[0m   [38;2;32;148;243m17.02556407[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.82415398[0m [38;2;32;148;243m-256.55706059[0m [38;2;32;148;243m-189.58177739[0m   [38;2;32;148;243m17.06940503[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.91900959[0m [38;2;32;148;243m-256.45221112[0m [38;2;32;148;243m-189.54598199[0m   [38;2;32;148;243m17.12813164[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.76089447[0m [38;2;32;148;243m-256.61505054[0m [38;2;32;148;243m-189.45640459[0m   [38;2;32;148;243m17.09995904[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.62166943[0m [38;2;32;148;243m-256.76307375[0m [38;2;32;148;243m-189.35122961[0m   [38;2;32;148;243m17.08707163[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.50106519[0m [38;2;32;148;243m-256.89602957[0m [38;2;32;148;243m-189.2319903[0m    [38;2;32;148;243m17.08934889[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m38.39833993[0m [38;2;32;148;243m-257.01404255[0m [38;2;32;148;243m-189.09974009[0m   [38;2;32;148;243m17.1062827[0m [1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10140685[0m [38;2;32;148;243m-0.0453699[0m  [38;2;32;148;243m-0.02355732[0m [38;2;32;148;243m-0.08745788[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20702387[0m [38;2;32;148;243m-0.08968837[0m [38;2;32;148;243m-0.04903474[0m [38;2;32;148;243m-0.16472241[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31351577[0m [38;2;32;148;243m-0.13394096[0m [38;2;32;148;243m-0.07168653[0m [38;2;32;148;243m-0.23992632[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.42121395[0m [38;2;32;148;243m-0.18102421[0m [38;2;32;148;243m-0.09224252[0m [38;2;32;148;243m-0.31533447[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31299255[0m [38;2;32;148;243m-0.13091163[0m [38;2;32;148;243m-0.0720724[0m  [38;2;32;148;243m-0.24004744[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20494739[0m [38;2;32;148;243m-0.08155329[0m [38;2;32;148;243m-0.05264525[0m [38;2;32;148;243m-0.16412183[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09690595[0m [38;2;32;148;243m-0.03333946[0m [38;2;32;148;243m-0.03296237[0m [38;2;32;148;243m-0.08772482[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01134244[0m  [38;2;32;148;243m0.01373151[0m [38;2;32;148;243m-0.01217873[0m [38;2;32;148;243m-0.01241313[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10140685[0m [38;2;32;148;243m-0.0453699[0m  [38;2;32;148;243m-0.02355732[0m [38;2;32;148;243m-0.08745788[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20702387[0m [38;2;32;148;243m-0.08968837[0m [38;2;32;148;243m-0.04903474[0m [38;2;32;148;243m-0.16472241[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31351577[0m [38;2;32;148;243m-0.13394096[0m [38;2;32;148;243m-0.07168653[0m [38;2;32;148;243m-0.23992632[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.42121395[0m [38;2;32;148;243m-0.18102421[0m [38;2;32;148;243m-0.09224252[0m [38;2;32;148;243m-0.31533447[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.1082214[0m  [38;2;32;148;243m0.05011257[0m [38;2;32;148;243m0.02017012[0m [38;2;32;148;243m0.07528702[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21626657[0m [38;2;32;148;243m0.09947091[0m [38;2;32;148;243m0.03959727[0m [38;2;32;148;243m0.15121264[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.324308[0m   [38;2;32;148;243m0.14768475[0m [38;2;32;148;243m0.05928015[0m [38;2;32;148;243m0.22760964[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.43255639[0m [38;2;32;148;243m0.19475572[0m [38;2;32;148;243m0.08006379[0m [38;2;32;148;243m0.30292134[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10140685[0m [38;2;32;148;243m-0.0453699[0m  [38;2;32;148;243m-0.02355732[0m [38;2;32;148;243m-0.08745788[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20702387[0m [38;2;32;148;243m-0.08968837[0m [38;2;32;148;243m-0.04903474[0m [38;2;32;148;243m-0.16472241[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31351577[0m [38;2;32;148;243m-0.13394096[0m [38;2;32;148;243m-0.07168653[0m [38;2;32;148;243m-0.23992632[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.42121395[0m [38;2;32;148;243m-0.18102421[0m [38;2;32;148;243m-0.09224252[0m [38;2;32;148;243m-0.31533447[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31299255[0m [38;2;32;148;243m-0.13091163[0m [38;2;32;148;243m-0.0720724[0m  [38;2;32;148;243m-0.24004744[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20494739[0m [38;2;32;148;243m-0.08155329[0m [38;2;32;148;243m-0.05264525[0m [38;2;32;148;243m-0.16412183[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09690595[0m [38;2;32;148;243m-0.03333946[0m [38;2;32;148;243m-0.03296237[0m [38;2;32;148;243m-0.08772482[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01134244[0m  [38;2;32;148;243m0.01373151[0m [38;2;32;148;243m-0.01217873[0m [38;2;32;148;243m-0.01241313[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01134244[0m  [38;2;32;148;243m0.01373151[0m [38;2;32;148;243m-0.01217873[0m [38;2;32;148;243m-0.01241313[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.173678843041342e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00589752[0m  [38;2;32;148;243m0.00556446[0m  [38;2;32;148;243m0.00453206[0m  [38;2;32;148;243m0.00283429[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0013992[0m   [38;2;32;148;243m0.00034847[0m [38;2;32;148;243m-0.0030403[0m  [38;2;32;148;243m-0.00230766[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04082942[0m  [38;2;32;148;243m0.01016853[0m [38;2;32;148;243m-0.08871753[0m [38;2;32;148;243m-0.06733875[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00390699[0m [38;2;32;148;243m0.00238289[0m [38;2;32;148;243m0.00855818[0m [38;2;32;148;243m0.00216934[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.33890400e-04[0m [38;2;32;148;243m8.16602253e-05[0m [38;2;32;148;243m2.93284326e-04[0m [38;2;32;148;243m7.43419571e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">9</span>                                                                                           
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:20] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">10</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:20][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m165.78526004[0m [38;2;32;148;243m-158.66660112[0m [38;2;32;148;243m-150.84648678[0m [38;2;32;148;243m-277.1795612[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m147.39212947[0m [38;2;32;148;243m-160.8891048[0m  [38;2;32;148;243m-167.54447157[0m [38;2;32;148;243m-290.5208167[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m147.17045246[0m [38;2;32;148;243m-160.83928327[0m [38;2;32;148;243m-167.60853844[0m [38;2;32;148;243m-290.55458752[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m146.96450675[0m [38;2;32;148;243m-160.77671527[0m [38;2;32;148;243m-167.65852461[0m [38;2;32;148;243m-290.57214886[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m146.77478032[0m [38;2;32;148;243m-160.700923[0m   [38;2;32;148;243m-167.69478487[0m [38;2;32;148;243m-290.5741648[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m146.98532096[0m [38;2;32;148;243m-160.72826923[0m [38;2;32;148;243m-167.77220345[0m [38;2;32;148;243m-290.53358274[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.2134995[0m  [38;2;32;148;243m-160.74296164[0m [38;2;32;148;243m-167.83705697[0m [38;2;32;148;243m-290.4768103[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m147.45923281[0m [38;2;32;148;243m-160.74258706[0m [38;2;32;148;243m-167.88795065[0m [38;2;32;148;243m-290.40486434[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.72274485[0m [38;2;32;148;243m-160.72719985[0m [38;2;32;148;243m-167.92486955[0m [38;2;32;148;243m-290.31919424[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m165.78526004[0m [38;2;32;148;243m-158.66660112[0m [38;2;32;148;243m-150.84648678[0m [38;2;32;148;243m-277.1795612[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m147.49634341[0m [38;2;32;148;243m-160.85234158[0m [38;2;32;148;243m-167.52064545[0m [38;2;32;148;243m-290.43531158[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.37500476[0m [38;2;32;148;243m-160.76971169[0m [38;2;32;148;243m-167.55289547[0m [38;2;32;148;243m-290.38673672[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.27085091[0m [38;2;32;148;243m-160.67383095[0m [38;2;32;148;243m-167.570758[0m   [38;2;32;148;243m-290.32414945[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.18408477[0m [38;2;32;148;243m-160.56418795[0m [38;2;32;148;243m-167.57497862[0m [38;2;32;148;243m-290.24798665[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.29124479[0m [38;2;32;148;243m-160.627483[0m   [38;2;32;148;243m-167.68155488[0m [38;2;32;148;243m-290.28472817[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.41620449[0m [38;2;32;148;243m-160.67938552[0m [38;2;32;148;243m-167.77259251[0m [38;2;32;148;243m-290.30666732[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.55913924[0m [38;2;32;148;243m-160.7151849[0m  [38;2;32;148;243m-167.84958544[0m [38;2;32;148;243m-290.31370793[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m147.71994084[0m [38;2;32;148;243m-160.73486739[0m [38;2;32;148;243m-167.91182803[0m [38;2;32;148;243m-290.30589063[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10421394[0m [38;2;32;148;243m-0.03676322[0m [38;2;32;148;243m-0.02382612[0m [38;2;32;148;243m-0.08550512[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20455229[0m [38;2;32;148;243m-0.06957158[0m [38;2;32;148;243m-0.05564296[0m [38;2;32;148;243m-0.1678508[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30634415[0m [38;2;32;148;243m-0.10288432[0m [38;2;32;148;243m-0.08776661[0m [38;2;32;148;243m-0.24799941[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40930446[0m [38;2;32;148;243m-0.13673505[0m [38;2;32;148;243m-0.11980625[0m [38;2;32;148;243m-0.32617815[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30592382[0m [38;2;32;148;243m-0.10078623[0m [38;2;32;148;243m-0.09064857[0m [38;2;32;148;243m-0.24885457[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20270499[0m [38;2;32;148;243m-0.06357612[0m [38;2;32;148;243m-0.06446446[0m [38;2;32;148;243m-0.17014297[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09990643[0m [38;2;32;148;243m-0.02740216[0m [38;2;32;148;243m-0.03836522[0m [38;2;32;148;243m-0.09115642[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00280402[0m  [38;2;32;148;243m0.00766754[0m [38;2;32;148;243m-0.01304152[0m [38;2;32;148;243m-0.01330361[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10421394[0m [38;2;32;148;243m-0.03676322[0m [38;2;32;148;243m-0.02382612[0m [38;2;32;148;243m-0.08550512[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20455229[0m [38;2;32;148;243m-0.06957158[0m [38;2;32;148;243m-0.05564296[0m [38;2;32;148;243m-0.1678508[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30634415[0m [38;2;32;148;243m-0.10288432[0m [38;2;32;148;243m-0.08776661[0m [38;2;32;148;243m-0.24799941[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40930446[0m [38;2;32;148;243m-0.13673505[0m [38;2;32;148;243m-0.11980625[0m [38;2;32;148;243m-0.32617815[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10338063[0m [38;2;32;148;243m0.03594882[0m [38;2;32;148;243m0.02915768[0m [38;2;32;148;243m0.07732358[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.20659947[0m [38;2;32;148;243m0.07315893[0m [38;2;32;148;243m0.05534179[0m [38;2;32;148;243m0.15603518[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.30939802[0m [38;2;32;148;243m0.10933289[0m [38;2;32;148;243m0.08144103[0m [38;2;32;148;243m0.23502173[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.41210847[0m [38;2;32;148;243m0.14440259[0m [38;2;32;148;243m0.10676473[0m [38;2;32;148;243m0.31287454[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10421394[0m [38;2;32;148;243m-0.03676322[0m [38;2;32;148;243m-0.02382612[0m [38;2;32;148;243m-0.08550512[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20455229[0m [38;2;32;148;243m-0.06957158[0m [38;2;32;148;243m-0.05564296[0m [38;2;32;148;243m-0.1678508[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30634415[0m [38;2;32;148;243m-0.10288432[0m [38;2;32;148;243m-0.08776661[0m [38;2;32;148;243m-0.24799941[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40930446[0m [38;2;32;148;243m-0.13673505[0m [38;2;32;148;243m-0.11980625[0m [38;2;32;148;243m-0.32617815[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30592382[0m [38;2;32;148;243m-0.10078623[0m [38;2;32;148;243m-0.09064857[0m [38;2;32;148;243m-0.24885457[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20270499[0m [38;2;32;148;243m-0.06357612[0m [38;2;32;148;243m-0.06446446[0m [38;2;32;148;243m-0.17014297[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09990643[0m [38;2;32;148;243m-0.02740216[0m [38;2;32;148;243m-0.03836522[0m [38;2;32;148;243m-0.09115642[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00280402[0m  [38;2;32;148;243m0.00766754[0m [38;2;32;148;243m-0.01304152[0m [38;2;32;148;243m-0.01330361[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00280402[0m  [38;2;32;148;243m0.00766754[0m [38;2;32;148;243m-0.01304152[0m [38;2;32;148;243m-0.01330361[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.176952518904433e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00560066[0m  [38;2;32;148;243m0.00583437[0m  [38;2;32;148;243m0.00445981[0m  [38;2;32;148;243m0.00269613[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00164278[0m  [38;2;32;148;243m0.00039318[0m [38;2;32;148;243m-0.0029024[0m  [38;2;32;148;243m-0.00230664[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04793717[0m  [38;2;32;148;243m0.01147332[0m [38;2;32;148;243m-0.08469338[0m [38;2;32;148;243m-0.06730904[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00583948[0m [38;2;32;148;243m0.00029422[0m [38;2;32;148;243m0.00308553[0m [38;2;32;148;243m0.00448383[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m2.00115985e-04[0m [38;2;32;148;243m1.00829077e-05[0m [38;2;32;148;243m1.05739392e-04[0m [38;2;32;148;243m1.53658415e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">11</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:21] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">12</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:21][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m188.20505965[0m  [38;2;32;148;243m-58.29885144[0m  [38;2;32;148;243m-98.18029968[0m  [38;2;32;148;243m-54.39963063[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.51710167[0m  [38;2;32;148;243m-61.06483545[0m [38;2;32;148;243m-116.01752595[0m  [38;2;32;148;243m-68.13930872[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.22367662[0m  [38;2;32;148;243m-61.02729971[0m [38;2;32;148;243m-116.12317278[0m  [38;2;32;148;243m-68.35212217[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m169.94726353[0m  [38;2;32;148;243m-60.97502657[0m [38;2;32;148;243m-116.21178215[0m  [38;2;32;148;243m-68.54917981[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m169.6874372[0m   [38;2;32;148;243m-60.90769105[0m [38;2;32;148;243m-116.28237697[0m  [38;2;32;148;243m-68.73040709[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m169.88756791[0m  [38;2;32;148;243m-60.60725102[0m [38;2;32;148;243m-116.19644423[0m  [38;2;32;148;243m-68.68839189[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.10515006[0m  [38;2;32;148;243m-60.29350582[0m [38;2;32;148;243m-116.09583329[0m  [38;2;32;148;243m-68.63115605[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.33946017[0m  [38;2;32;148;243m-59.96648684[0m [38;2;32;148;243m-115.98075856[0m  [38;2;32;148;243m-68.55935268[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.590336[0m    [38;2;32;148;243m-59.62290977[0m [38;2;32;148;243m-115.85112333[0m  [38;2;32;148;243m-68.47245672[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m188.20505965[0m  [38;2;32;148;243m-58.29885144[0m  [38;2;32;148;243m-98.18029968[0m  [38;2;32;148;243m-54.39963063[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.61883772[0m  [38;2;32;148;243m-60.99860325[0m [38;2;32;148;243m-115.99267874[0m  [38;2;32;148;243m-68.05494583[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.42464918[0m  [38;2;32;148;243m-60.88261255[0m [38;2;32;148;243m-116.07186839[0m  [38;2;32;148;243m-68.18439678[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.24744052[0m  [38;2;32;148;243m-60.75120359[0m [38;2;32;148;243m-116.1365266[0m   [38;2;32;148;243m-68.29608566[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.08701925[0m  [38;2;32;148;243m-60.60473632[0m [38;2;32;148;243m-116.18640188[0m  [38;2;32;148;243m-68.3914714[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m170.18679771[0m  [38;2;32;148;243m-60.38249412[0m [38;2;32;148;243m-116.11953711[0m  [38;2;32;148;243m-68.43541968[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.30402635[0m  [38;2;32;148;243m-60.14605486[0m [38;2;32;148;243m-116.03781314[0m  [38;2;32;148;243m-68.46434924[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.43897919[0m  [38;2;32;148;243m-59.8957125[0m  [38;2;32;148;243m-115.94174113[0m  [38;2;32;148;243m-68.47993056[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m170.5910721[0m   [38;2;32;148;243m-59.63004352[0m [38;2;32;148;243m-115.83059649[0m  [38;2;32;148;243m-68.48065411[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10173606[0m [38;2;32;148;243m-0.06623219[0m [38;2;32;148;243m-0.02484721[0m [38;2;32;148;243m-0.08436289[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20097256[0m [38;2;32;148;243m-0.14468717[0m [38;2;32;148;243m-0.05130439[0m [38;2;32;148;243m-0.16772539[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30017699[0m [38;2;32;148;243m-0.22382299[0m [38;2;32;148;243m-0.07525555[0m [38;2;32;148;243m-0.25309416[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39958205[0m [38;2;32;148;243m-0.30295473[0m [38;2;32;148;243m-0.09597509[0m [38;2;32;148;243m-0.33893569[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2992298[0m  [38;2;32;148;243m-0.22475689[0m [38;2;32;148;243m-0.07690712[0m [38;2;32;148;243m-0.25297222[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19887629[0m [38;2;32;148;243m-0.14745096[0m [38;2;32;148;243m-0.05802014[0m [38;2;32;148;243m-0.16680681[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09951902[0m [38;2;32;148;243m-0.07077434[0m [38;2;32;148;243m-0.03901743[0m [38;2;32;148;243m-0.07942211[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00073611[0m  [38;2;32;148;243m0.00713375[0m [38;2;32;148;243m-0.02052685[0m  [38;2;32;148;243m0.00819739[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10173606[0m [38;2;32;148;243m-0.06623219[0m [38;2;32;148;243m-0.02484721[0m [38;2;32;148;243m-0.08436289[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20097256[0m [38;2;32;148;243m-0.14468717[0m [38;2;32;148;243m-0.05130439[0m [38;2;32;148;243m-0.16772539[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30017699[0m [38;2;32;148;243m-0.22382299[0m [38;2;32;148;243m-0.07525555[0m [38;2;32;148;243m-0.25309416[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39958205[0m [38;2;32;148;243m-0.30295473[0m [38;2;32;148;243m-0.09597509[0m [38;2;32;148;243m-0.33893569[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10035224[0m [38;2;32;148;243m0.07819784[0m [38;2;32;148;243m0.01906797[0m [38;2;32;148;243m0.08596348[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.20070576[0m [38;2;32;148;243m0.15550378[0m [38;2;32;148;243m0.03795495[0m [38;2;32;148;243m0.17212888[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.30006303[0m [38;2;32;148;243m0.2321804[0m  [38;2;32;148;243m0.05695766[0m [38;2;32;148;243m0.25951358[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.39884594[0m [38;2;32;148;243m0.31008849[0m [38;2;32;148;243m0.07544825[0m [38;2;32;148;243m0.34713308[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10173606[0m [38;2;32;148;243m-0.06623219[0m [38;2;32;148;243m-0.02484721[0m [38;2;32;148;243m-0.08436289[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20097256[0m [38;2;32;148;243m-0.14468717[0m [38;2;32;148;243m-0.05130439[0m [38;2;32;148;243m-0.16772539[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30017699[0m [38;2;32;148;243m-0.22382299[0m [38;2;32;148;243m-0.07525555[0m [38;2;32;148;243m-0.25309416[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39958205[0m [38;2;32;148;243m-0.30295473[0m [38;2;32;148;243m-0.09597509[0m [38;2;32;148;243m-0.33893569[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2992298[0m  [38;2;32;148;243m-0.22475689[0m [38;2;32;148;243m-0.07690712[0m [38;2;32;148;243m-0.25297222[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19887629[0m [38;2;32;148;243m-0.14745096[0m [38;2;32;148;243m-0.05802014[0m [38;2;32;148;243m-0.16680681[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09951902[0m [38;2;32;148;243m-0.07077434[0m [38;2;32;148;243m-0.03901743[0m [38;2;32;148;243m-0.07942211[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00073611[0m  [38;2;32;148;243m0.00713375[0m [38;2;32;148;243m-0.02052685[0m  [38;2;32;148;243m0.00819739[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00073611[0m  [38;2;32;148;243m0.00713375[0m [38;2;32;148;243m-0.02052685[0m  [38;2;32;148;243m0.00819739[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.275198253756771e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00565215[0m  [38;2;32;148;243m0.00597338[0m  [38;2;32;148;243m0.00471459[0m  [38;2;32;148;243m0.00279347[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m1.64529269e-03[0m  [38;2;32;148;243m5.24680611e-05[0m [38;2;32;148;243m-2.97667902e-03[0m [38;2;32;148;243m-2.65912988e-03[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04801046[0m  [38;2;32;148;243m0.00153104[0m [38;2;32;148;243m-0.08686098[0m [38;2;32;148;243m-0.07759474[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00065245[0m [38;2;32;148;243m0.00969285[0m [38;2;32;148;243m0.00241307[0m [38;2;32;148;243m0.0080844[0m [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m2.23591304e-05[0m [38;2;32;148;243m3.32168887e-04[0m [38;2;32;148;243m8.26946588e-05[0m [38;2;32;148;243m2.77047880e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">13</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:22] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">14</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:22][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m269.43140031[0m [38;2;32;148;243m-78.92513987[0m  [38;2;32;148;243m32.34792115[0m [38;2;32;148;243m113.77488245[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.52566438[0m [38;2;32;148;243m-83.0934488[0m   [38;2;32;148;243m14.569342[0m   [38;2;32;148;243m100.76386849[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.51054733[0m [38;2;32;148;243m-83.03842171[0m  [38;2;32;148;243m14.48931617[0m [38;2;32;148;243m100.64609339[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.51146935[0m [38;2;32;148;243m-82.96954958[0m  [38;2;32;148;243m14.42328181[0m [38;2;32;148;243m100.54350263[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.52862743[0m [38;2;32;148;243m-82.88674771[0m  [38;2;32;148;243m14.37082778[0m [38;2;32;148;243m100.4575535[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m250.58961371[0m [38;2;32;148;243m-83.01337977[0m  [38;2;32;148;243m14.46234277[0m [38;2;32;148;243m100.48350857[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.66604833[0m [38;2;32;148;243m-83.12516719[0m  [38;2;32;148;243m14.5679644[0m  [38;2;32;148;243m100.52532797[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.75806246[0m [38;2;32;148;243m-83.22390956[0m  [38;2;32;148;243m14.68536563[0m [38;2;32;148;243m100.58193137[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.86680332[0m [38;2;32;148;243m-83.30966923[0m  [38;2;32;148;243m14.81512122[0m [38;2;32;148;243m100.65259476[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m269.43140031[0m [38;2;32;148;243m-78.92513987[0m  [38;2;32;148;243m32.34792115[0m [38;2;32;148;243m113.77488245[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.61992512[0m [38;2;32;148;243m-83.05632952[0m  [38;2;32;148;243m14.6020026[0m  [38;2;32;148;243m100.85210239[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.69159598[0m [38;2;32;148;243m-82.96117521[0m  [38;2;32;148;243m14.57372486[0m [38;2;32;148;243m100.81378112[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.77872241[0m [38;2;32;148;243m-82.85454899[0m  [38;2;32;148;243m14.56012068[0m [38;2;32;148;243m100.79105526[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.88195949[0m [38;2;32;148;243m-82.73377657[0m  [38;2;32;148;243m14.55859979[0m [38;2;32;148;243m100.78363819[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.8571183[0m  [38;2;32;148;243m-82.89666386[0m  [38;2;32;148;243m14.59964566[0m [38;2;32;148;243m100.7307771[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m250.84671938[0m [38;2;32;148;243m-83.04389711[0m  [38;2;32;148;243m14.65428118[0m [38;2;32;148;243m100.69242974[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m250.85183089[0m [38;2;32;148;243m-83.17870223[0m  [38;2;32;148;243m14.72263822[0m [38;2;32;148;243m100.6694879[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m250.87622229[0m [38;2;32;148;243m-83.30090252[0m  [38;2;32;148;243m14.80466478[0m [38;2;32;148;243m100.66222142[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09426074[0m [38;2;32;148;243m-0.03711928[0m [38;2;32;148;243m-0.0326606[0m  [38;2;32;148;243m-0.08823389[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18104865[0m [38;2;32;148;243m-0.0772465[0m  [38;2;32;148;243m-0.08440869[0m [38;2;32;148;243m-0.16768772[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.26725305[0m [38;2;32;148;243m-0.11500059[0m [38;2;32;148;243m-0.13683887[0m [38;2;32;148;243m-0.24755263[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.35333206[0m [38;2;32;148;243m-0.15297115[0m [38;2;32;148;243m-0.18777202[0m [38;2;32;148;243m-0.32608469[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.26750458[0m [38;2;32;148;243m-0.11671591[0m [38;2;32;148;243m-0.13730289[0m [38;2;32;148;243m-0.24726853[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18067105[0m [38;2;32;148;243m-0.08127008[0m [38;2;32;148;243m-0.08631679[0m [38;2;32;148;243m-0.16710177[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09376843[0m [38;2;32;148;243m-0.04520733[0m [38;2;32;148;243m-0.03727259[0m [38;2;32;148;243m-0.08755653[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00941898[0m [38;2;32;148;243m-0.0087667[0m   [38;2;32;148;243m0.01045645[0m [38;2;32;148;243m-0.00962666[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09426074[0m [38;2;32;148;243m-0.03711928[0m [38;2;32;148;243m-0.0326606[0m  [38;2;32;148;243m-0.08823389[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18104865[0m [38;2;32;148;243m-0.0772465[0m  [38;2;32;148;243m-0.08440869[0m [38;2;32;148;243m-0.16768772[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.26725305[0m [38;2;32;148;243m-0.11500059[0m [38;2;32;148;243m-0.13683887[0m [38;2;32;148;243m-0.24755263[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.35333206[0m [38;2;32;148;243m-0.15297115[0m [38;2;32;148;243m-0.18777202[0m [38;2;32;148;243m-0.32608469[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.08582748[0m [38;2;32;148;243m0.03625524[0m [38;2;32;148;243m0.05046912[0m [38;2;32;148;243m0.07881616[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.17266101[0m [38;2;32;148;243m0.07170107[0m [38;2;32;148;243m0.10145523[0m [38;2;32;148;243m0.15898292[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.25956363[0m [38;2;32;148;243m0.10776382[0m [38;2;32;148;243m0.15049943[0m [38;2;32;148;243m0.23852816[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.34391308[0m [38;2;32;148;243m0.14420444[0m [38;2;32;148;243m0.19822846[0m [38;2;32;148;243m0.31645803[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09426074[0m [38;2;32;148;243m-0.03711928[0m [38;2;32;148;243m-0.0326606[0m  [38;2;32;148;243m-0.08823389[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18104865[0m [38;2;32;148;243m-0.0772465[0m  [38;2;32;148;243m-0.08440869[0m [38;2;32;148;243m-0.16768772[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.26725305[0m [38;2;32;148;243m-0.11500059[0m [38;2;32;148;243m-0.13683887[0m [38;2;32;148;243m-0.24755263[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.35333206[0m [38;2;32;148;243m-0.15297115[0m [38;2;32;148;243m-0.18777202[0m [38;2;32;148;243m-0.32608469[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.26750458[0m [38;2;32;148;243m-0.11671591[0m [38;2;32;148;243m-0.13730289[0m [38;2;32;148;243m-0.24726853[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18067105[0m [38;2;32;148;243m-0.08127008[0m [38;2;32;148;243m-0.08631679[0m [38;2;32;148;243m-0.16710177[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09376843[0m [38;2;32;148;243m-0.04520733[0m [38;2;32;148;243m-0.03727259[0m [38;2;32;148;243m-0.08755653[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00941898[0m [38;2;32;148;243m-0.0087667[0m   [38;2;32;148;243m0.01045645[0m [38;2;32;148;243m-0.00962666[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00941898[0m [38;2;32;148;243m-0.0087667[0m   [38;2;32;148;243m0.01045645[0m [38;2;32;148;243m-0.00962666[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.288034143291682e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00545839[0m  [38;2;32;148;243m0.00615163[0m  [38;2;32;148;243m0.0047064[0m   [38;2;32;148;243m0.00285683[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00169113[0m  [38;2;32;148;243m0.00022278[0m [38;2;32;148;243m-0.00285758[0m [38;2;32;148;243m-0.00260122[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04934804[0m  [38;2;32;148;243m0.00650095[0m [38;2;32;148;243m-0.08338556[0m [38;2;32;148;243m-0.07590503[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00654428[0m [38;2;32;148;243m0.00071274[0m [38;2;32;148;243m0.00170216[0m [38;2;32;148;243m0.00521421[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m2.24269016e-04[0m [38;2;32;148;243m2.44251559e-05[0m [38;2;32;148;243m5.83322748e-05[0m [38;2;32;148;243m1.78688118e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">15</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:23] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">16</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:23][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m242.09111141[0m [38;2;32;148;243m-167.78069176[0m  [38;2;32;148;243m-86.79127201[0m   [38;2;32;148;243m-1.71409733[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.42479069[0m [38;2;32;148;243m-170.8975103[0m  [38;2;32;148;243m-104.09194289[0m  [38;2;32;148;243m-15.06808349[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.32836168[0m [38;2;32;148;243m-171.01752667[0m [38;2;32;148;243m-104.26316404[0m  [38;2;32;148;243m-15.13329744[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.25220575[0m [38;2;32;148;243m-171.12311605[0m [38;2;32;148;243m-104.41971763[0m  [38;2;32;148;243m-15.18359855[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.19583803[0m [38;2;32;148;243m-171.21411865[0m [38;2;32;148;243m-104.56149024[0m  [38;2;32;148;243m-15.21926476[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.23732721[0m [38;2;32;148;243m-171.07503408[0m [38;2;32;148;243m-104.45847981[0m  [38;2;32;148;243m-15.15402874[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.2982538[0m  [38;2;32;148;243m-170.91796681[0m [38;2;32;148;243m-104.34035922[0m  [38;2;32;148;243m-15.07306938[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.37934147[0m [38;2;32;148;243m-170.74700095[0m [38;2;32;148;243m-104.20937461[0m  [38;2;32;148;243m-14.97519778[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.48054702[0m [38;2;32;148;243m-170.56258876[0m [38;2;32;148;243m-104.06378562[0m  [38;2;32;148;243m-14.86043875[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m242.09111141[0m [38;2;32;148;243m-167.78069176[0m  [38;2;32;148;243m-86.79127201[0m   [38;2;32;148;243m-1.71409733[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.52786528[0m [38;2;32;148;243m-170.83137805[0m [38;2;32;148;243m-104.0718157[0m   [38;2;32;148;243m-14.98056815[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.54494286[0m [38;2;32;148;243m-170.88452904[0m [38;2;32;148;243m-104.2145884[0m   [38;2;32;148;243m-14.96318368[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.58203094[0m [38;2;32;148;243m-170.92414762[0m [38;2;32;148;243m-104.34489498[0m  [38;2;32;148;243m-14.93017888[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.63913253[0m [38;2;32;148;243m-170.95000627[0m [38;2;32;148;243m-104.46195817[0m  [38;2;32;148;243m-14.8814643[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m223.56698162[0m [38;2;32;148;243m-170.87777937[0m [38;2;32;148;243m-104.38605005[0m  [38;2;32;148;243m-14.90105953[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.51387208[0m [38;2;32;148;243m-170.79101929[0m [38;2;32;148;243m-104.29603278[0m  [38;2;32;148;243m-14.90494681[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.48004487[0m [38;2;32;148;243m-170.6902614[0m  [38;2;32;148;243m-104.19119749[0m  [38;2;32;148;243m-14.89268621[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m223.46577315[0m [38;2;32;148;243m-170.57526562[0m [38;2;32;148;243m-104.07214949[0m  [38;2;32;148;243m-14.86422353[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10307459[0m [38;2;32;148;243m-0.06613225[0m [38;2;32;148;243m-0.02012719[0m [38;2;32;148;243m-0.08751534[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21658118[0m [38;2;32;148;243m-0.13299763[0m [38;2;32;148;243m-0.04857564[0m [38;2;32;148;243m-0.17011376[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32982519[0m [38;2;32;148;243m-0.19896844[0m [38;2;32;148;243m-0.07482265[0m [38;2;32;148;243m-0.25341967[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.4432945[0m  [38;2;32;148;243m-0.26411238[0m [38;2;32;148;243m-0.09953207[0m [38;2;32;148;243m-0.33780046[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32965441[0m [38;2;32;148;243m-0.19725471[0m [38;2;32;148;243m-0.07242976[0m [38;2;32;148;243m-0.25296921[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21561828[0m [38;2;32;148;243m-0.12694752[0m [38;2;32;148;243m-0.04432644[0m [38;2;32;148;243m-0.16812257[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1007034[0m  [38;2;32;148;243m-0.05673955[0m [38;2;32;148;243m-0.01817712[0m [38;2;32;148;243m-0.08251156[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01477387[0m  [38;2;32;148;243m0.01267686[0m  [38;2;32;148;243m0.00836387[0m  [38;2;32;148;243m0.00378478[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10307459[0m [38;2;32;148;243m-0.06613225[0m [38;2;32;148;243m-0.02012719[0m [38;2;32;148;243m-0.08751534[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21658118[0m [38;2;32;148;243m-0.13299763[0m [38;2;32;148;243m-0.04857564[0m [38;2;32;148;243m-0.17011376[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32982519[0m [38;2;32;148;243m-0.19896844[0m [38;2;32;148;243m-0.07482265[0m [38;2;32;148;243m-0.25341967[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.4432945[0m  [38;2;32;148;243m-0.26411238[0m [38;2;32;148;243m-0.09953207[0m [38;2;32;148;243m-0.33780046[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.1136401[0m  [38;2;32;148;243m0.06685767[0m [38;2;32;148;243m0.02710231[0m [38;2;32;148;243m0.08483125[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.22767622[0m [38;2;32;148;243m0.13716486[0m [38;2;32;148;243m0.05520562[0m [38;2;32;148;243m0.1696779[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.34259111[0m [38;2;32;148;243m0.20737283[0m [38;2;32;148;243m0.08135495[0m [38;2;32;148;243m0.2552889[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.45806837[0m [38;2;32;148;243m0.27678924[0m [38;2;32;148;243m0.10789594[0m [38;2;32;148;243m0.34158524[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10307459[0m [38;2;32;148;243m-0.06613225[0m [38;2;32;148;243m-0.02012719[0m [38;2;32;148;243m-0.08751534[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21658118[0m [38;2;32;148;243m-0.13299763[0m [38;2;32;148;243m-0.04857564[0m [38;2;32;148;243m-0.17011376[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32982519[0m [38;2;32;148;243m-0.19896844[0m [38;2;32;148;243m-0.07482265[0m [38;2;32;148;243m-0.25341967[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.4432945[0m  [38;2;32;148;243m-0.26411238[0m [38;2;32;148;243m-0.09953207[0m [38;2;32;148;243m-0.33780046[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32965441[0m [38;2;32;148;243m-0.19725471[0m [38;2;32;148;243m-0.07242976[0m [38;2;32;148;243m-0.25296921[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21561828[0m [38;2;32;148;243m-0.12694752[0m [38;2;32;148;243m-0.04432644[0m [38;2;32;148;243m-0.16812257[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1007034[0m  [38;2;32;148;243m-0.05673955[0m [38;2;32;148;243m-0.01817712[0m [38;2;32;148;243m-0.08251156[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01477387[0m  [38;2;32;148;243m0.01267686[0m  [38;2;32;148;243m0.00836387[0m  [38;2;32;148;243m0.00378478[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.01477387[0m [38;2;32;148;243m0.01267686[0m [38;2;32;148;243m0.00836387[0m [38;2;32;148;243m0.00378478[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.286711824528633e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00537043[0m  [38;2;32;148;243m0.00654723[0m  [38;2;32;148;243m0.0046871[0m   [38;2;32;148;243m0.00286848[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0014679[0m   [38;2;32;148;243m0.00035176[0m [38;2;32;148;243m-0.00312108[0m [38;2;32;148;243m-0.00251439[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04283408[0m  [38;2;32;148;243m0.01026443[0m [38;2;32;148;243m-0.09107459[0m [38;2;32;148;243m-0.07337109[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00142679[0m [38;2;32;148;243m0.00507351[0m [38;2;32;148;243m0.00434687[0m [38;2;32;148;243m0.00288916[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m4.88952829e-05[0m [38;2;32;148;243m1.73866583e-04[0m [38;2;32;148;243m1.48964780e-04[0m [38;2;32;148;243m9.90099206e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">17</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:24] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">18</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:24][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m51.81364669[0m [38;2;32;148;243m-276.10671185[0m  [38;2;32;148;243m-98.60682104[0m  [38;2;32;148;243m-25.36370694[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.66548524[0m [38;2;32;148;243m-279.85187586[0m [38;2;32;148;243m-116.04757011[0m  [38;2;32;148;243m-37.27420888[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.52539767[0m [38;2;32;148;243m-279.95203661[0m [38;2;32;148;243m-116.18756296[0m  [38;2;32;148;243m-37.35219963[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.40266348[0m [38;2;32;148;243m-280.0364063[0m  [38;2;32;148;243m-116.31266057[0m  [38;2;32;148;243m-37.41251265[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.2975883[0m  [38;2;32;148;243m-280.10401611[0m [38;2;32;148;243m-116.42371203[0m  [38;2;32;148;243m-37.45611136[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.49407266[0m [38;2;32;148;243m-279.90120261[0m [38;2;32;148;243m-116.42109307[0m  [38;2;32;148;243m-37.38052488[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.70561809[0m [38;2;32;148;243m-279.68370345[0m [38;2;32;148;243m-116.40390229[0m  [38;2;32;148;243m-37.28787851[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.9320363[0m  [38;2;32;148;243m-279.45238646[0m [38;2;32;148;243m-116.3725013[0m   [38;2;32;148;243m-37.17660581[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m33.17450914[0m [38;2;32;148;243m-279.20755491[0m [38;2;32;148;243m-116.32766274[0m  [38;2;32;148;243m-37.04654594[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m51.81364669[0m [38;2;32;148;243m-276.10671185[0m  [38;2;32;148;243m-98.60682104[0m  [38;2;32;148;243m-25.36370694[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.76189104[0m [38;2;32;148;243m-279.82955433[0m [38;2;32;148;243m-116.02372562[0m  [38;2;32;148;243m-37.18231376[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.71911899[0m [38;2;32;148;243m-279.90477912[0m [38;2;32;148;243m-116.13872567[0m  [38;2;32;148;243m-37.17473898[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.693671[0m   [38;2;32;148;243m-279.96520078[0m [38;2;32;148;243m-116.2393825[0m   [38;2;32;148;243m-37.1508859[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m32.68587057[0m [38;2;32;148;243m-280.00999821[0m [38;2;32;148;243m-116.32586348[0m  [38;2;32;148;243m-37.11121438[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m32.78485093[0m [38;2;32;148;243m-279.8297702[0m  [38;2;32;148;243m-116.34805175[0m  [38;2;32;148;243m-37.1188895[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m32.89988208[0m [38;2;32;148;243m-279.63562624[0m [38;2;32;148;243m-116.35767199[0m  [38;2;32;148;243m-37.11027228[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m33.03055209[0m [38;2;32;148;243m-279.42760771[0m [38;2;32;148;243m-116.3539882[0m   [38;2;32;148;243m-37.08474785[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m33.17727777[0m [38;2;32;148;243m-279.20475654[0m [38;2;32;148;243m-116.33262468[0m  [38;2;32;148;243m-37.04216334[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0964058[0m  [38;2;32;148;243m-0.02232153[0m [38;2;32;148;243m-0.02384449[0m [38;2;32;148;243m-0.09189513[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19372132[0m [38;2;32;148;243m-0.04725749[0m [38;2;32;148;243m-0.04883729[0m [38;2;32;148;243m-0.17746065[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29100752[0m [38;2;32;148;243m-0.07120551[0m [38;2;32;148;243m-0.07327806[0m [38;2;32;148;243m-0.26162675[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.38828227[0m [38;2;32;148;243m-0.0940179[0m  [38;2;32;148;243m-0.09784855[0m [38;2;32;148;243m-0.34489698[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29077828[0m [38;2;32;148;243m-0.07143241[0m [38;2;32;148;243m-0.07304131[0m [38;2;32;148;243m-0.26163538[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19426399[0m [38;2;32;148;243m-0.0480772[0m  [38;2;32;148;243m-0.0462303[0m  [38;2;32;148;243m-0.17760622[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09851579[0m [38;2;32;148;243m-0.02477875[0m [38;2;32;148;243m-0.0185131[0m  [38;2;32;148;243m-0.09185795[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00276863[0m [38;2;32;148;243m-0.00279837[0m  [38;2;32;148;243m0.00496193[0m [38;2;32;148;243m-0.0043826[0m [1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0964058[0m  [38;2;32;148;243m-0.02232153[0m [38;2;32;148;243m-0.02384449[0m [38;2;32;148;243m-0.09189513[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19372132[0m [38;2;32;148;243m-0.04725749[0m [38;2;32;148;243m-0.04883729[0m [38;2;32;148;243m-0.17746065[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29100752[0m [38;2;32;148;243m-0.07120551[0m [38;2;32;148;243m-0.07327806[0m [38;2;32;148;243m-0.26162675[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.38828227[0m [38;2;32;148;243m-0.0940179[0m  [38;2;32;148;243m-0.09784855[0m [38;2;32;148;243m-0.34489698[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.09750399[0m [38;2;32;148;243m0.02258548[0m [38;2;32;148;243m0.02480724[0m [38;2;32;148;243m0.0832616[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.19401828[0m [38;2;32;148;243m0.04594069[0m [38;2;32;148;243m0.05161825[0m [38;2;32;148;243m0.16729076[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.28976648[0m [38;2;32;148;243m0.06923915[0m [38;2;32;148;243m0.07933545[0m [38;2;32;148;243m0.25303903[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.38551364[0m [38;2;32;148;243m0.09121953[0m [38;2;32;148;243m0.10281049[0m [38;2;32;148;243m0.34051439[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0964058[0m  [38;2;32;148;243m-0.02232153[0m [38;2;32;148;243m-0.02384449[0m [38;2;32;148;243m-0.09189513[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19372132[0m [38;2;32;148;243m-0.04725749[0m [38;2;32;148;243m-0.04883729[0m [38;2;32;148;243m-0.17746065[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29100752[0m [38;2;32;148;243m-0.07120551[0m [38;2;32;148;243m-0.07327806[0m [38;2;32;148;243m-0.26162675[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.38828227[0m [38;2;32;148;243m-0.0940179[0m  [38;2;32;148;243m-0.09784855[0m [38;2;32;148;243m-0.34489698[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29077828[0m [38;2;32;148;243m-0.07143241[0m [38;2;32;148;243m-0.07304131[0m [38;2;32;148;243m-0.26163538[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19426399[0m [38;2;32;148;243m-0.0480772[0m  [38;2;32;148;243m-0.0462303[0m  [38;2;32;148;243m-0.17760622[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09851579[0m [38;2;32;148;243m-0.02477875[0m [38;2;32;148;243m-0.0185131[0m  [38;2;32;148;243m-0.09185795[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00276863[0m [38;2;32;148;243m-0.00279837[0m  [38;2;32;148;243m0.00496193[0m [38;2;32;148;243m-0.0043826[0m [1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00276863[0m [38;2;32;148;243m-0.00279837[0m  [38;2;32;148;243m0.00496193[0m [38;2;32;148;243m-0.0043826[0m [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.167386663631978e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00517545[0m  [38;2;32;148;243m0.00667307[0m  [38;2;32;148;243m0.00507462[0m  [38;2;32;148;243m0.00313774[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00132024[0m  [38;2;32;148;243m0.00044875[0m [38;2;32;148;243m-0.002744[0m   [38;2;32;148;243m-0.00245262[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.03852533[0m  [38;2;32;148;243m0.01309472[0m [38;2;32;148;243m-0.08007142[0m [38;2;32;148;243m-0.0715688[0m [1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m7.28081614e-03[0m [38;2;32;148;243m6.66048468e-06[0m [38;2;32;148;243m4.84043390e-03[0m [38;2;32;148;243m3.36405003e-03[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m2.49509636e-04[0m [38;2;32;148;243m2.28251212e-07[0m [38;2;32;148;243m1.65879055e-04[0m [38;2;32;148;243m1.15284177e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">19</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:25] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">20</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:25][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m123.0916713[0m  [38;2;32;148;243m-368.39530166[0m   [38;2;32;148;243m33.58539175[0m  [38;2;32;148;243m-34.38175277[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.52159903[0m [38;2;32;148;243m-371.9245264[0m    [38;2;32;148;243m16.25876955[0m  [38;2;32;148;243m-46.83411547[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.49818912[0m [38;2;32;148;243m-371.99175699[0m   [38;2;32;148;243m16.25033885[0m  [38;2;32;148;243m-46.90479335[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.49279164[0m [38;2;32;148;243m-372.04895247[0m   [38;2;32;148;243m16.25593112[0m  [38;2;32;148;243m-46.95916726[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.5056771[0m  [38;2;32;148;243m-372.0929779[0m    [38;2;32;148;243m16.27542596[0m  [38;2;32;148;243m-46.99594976[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.58569049[0m [38;2;32;148;243m-372.03068088[0m   [38;2;32;148;243m16.33807126[0m  [38;2;32;148;243m-46.87584595[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.68441208[0m [38;2;32;148;243m-371.95640792[0m   [38;2;32;148;243m16.41538247[0m  [38;2;32;148;243m-46.74498565[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.80248382[0m [38;2;32;148;243m-371.86939321[0m   [38;2;32;148;243m16.50716943[0m  [38;2;32;148;243m-46.60248449[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.94033489[0m [38;2;32;148;243m-371.76738451[0m   [38;2;32;148;243m16.61384277[0m  [38;2;32;148;243m-46.44620817[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m123.0916713[0m  [38;2;32;148;243m-368.39530166[0m   [38;2;32;148;243m33.58539175[0m  [38;2;32;148;243m-34.38175277[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.62409357[0m [38;2;32;148;243m-371.88212063[0m   [38;2;32;148;243m16.27096835[0m  [38;2;32;148;243m-46.74644487[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.71044289[0m [38;2;32;148;243m-371.89183418[0m   [38;2;32;148;243m16.2822088[0m   [38;2;32;148;243m-46.73869286[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.81537939[0m [38;2;32;148;243m-371.88864557[0m   [38;2;32;148;243m16.30635963[0m  [38;2;32;148;243m-46.71547364[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.93916614[0m [38;2;32;148;243m-371.8720773[0m    [38;2;32;148;243m16.34372148[0m  [38;2;32;148;243m-46.6760762[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m103.90804543[0m [38;2;32;148;243m-371.87005391[0m   [38;2;32;148;243m16.38865795[0m  [38;2;32;148;243m-46.63093433[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.89474182[0m [38;2;32;148;243m-371.85431278[0m   [38;2;32;148;243m16.44826651[0m  [38;2;32;148;243m-46.57274808[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.90017294[0m [38;2;32;148;243m-371.82514128[0m   [38;2;32;148;243m16.52183202[0m  [38;2;32;148;243m-46.50121273[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m103.92583081[0m [38;2;32;148;243m-371.78192182[0m   [38;2;32;148;243m16.61100638[0m  [38;2;32;148;243m-46.4154334[0m [1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10249453[0m [38;2;32;148;243m-0.04240577[0m [38;2;32;148;243m-0.0121988[0m  [38;2;32;148;243m-0.0876706[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21225377[0m [38;2;32;148;243m-0.09992281[0m [38;2;32;148;243m-0.03186996[0m [38;2;32;148;243m-0.16610049[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32258775[0m [38;2;32;148;243m-0.1603069[0m  [38;2;32;148;243m-0.05042851[0m [38;2;32;148;243m-0.24369362[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43348904[0m [38;2;32;148;243m-0.2209006[0m  [38;2;32;148;243m-0.06829553[0m [38;2;32;148;243m-0.31987357[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32235494[0m [38;2;32;148;243m-0.16062696[0m [38;2;32;148;243m-0.05058669[0m [38;2;32;148;243m-0.24491162[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21032974[0m [38;2;32;148;243m-0.10209513[0m [38;2;32;148;243m-0.03288404[0m [38;2;32;148;243m-0.17223757[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09768913[0m [38;2;32;148;243m-0.04425193[0m [38;2;32;148;243m-0.01466259[0m [38;2;32;148;243m-0.10127176[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01450408[0m  [38;2;32;148;243m0.01453731[0m  [38;2;32;148;243m0.0028364[0m  [38;2;32;148;243m-0.03077477[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10249453[0m [38;2;32;148;243m-0.04240577[0m [38;2;32;148;243m-0.0121988[0m  [38;2;32;148;243m-0.0876706[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21225377[0m [38;2;32;148;243m-0.09992281[0m [38;2;32;148;243m-0.03186996[0m [38;2;32;148;243m-0.16610049[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32258775[0m [38;2;32;148;243m-0.1603069[0m  [38;2;32;148;243m-0.05042851[0m [38;2;32;148;243m-0.24369362[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43348904[0m [38;2;32;148;243m-0.2209006[0m  [38;2;32;148;243m-0.06829553[0m [38;2;32;148;243m-0.31987357[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.11113411[0m [38;2;32;148;243m0.06027364[0m [38;2;32;148;243m0.01770883[0m [38;2;32;148;243m0.07496195[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.2231593[0m  [38;2;32;148;243m0.11880547[0m [38;2;32;148;243m0.03541149[0m [38;2;32;148;243m0.147636[0m  [1m][0m
>      [1m[[0m[38;2;32;148;243m0.33579992[0m [38;2;32;148;243m0.17664867[0m [38;2;32;148;243m0.05363294[0m [38;2;32;148;243m0.2186018[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.44799312[0m [38;2;32;148;243m0.23543791[0m [38;2;32;148;243m0.07113192[0m [38;2;32;148;243m0.2890988[0m [1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10249453[0m [38;2;32;148;243m-0.04240577[0m [38;2;32;148;243m-0.0121988[0m  [38;2;32;148;243m-0.0876706[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21225377[0m [38;2;32;148;243m-0.09992281[0m [38;2;32;148;243m-0.03186996[0m [38;2;32;148;243m-0.16610049[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32258775[0m [38;2;32;148;243m-0.1603069[0m  [38;2;32;148;243m-0.05042851[0m [38;2;32;148;243m-0.24369362[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43348904[0m [38;2;32;148;243m-0.2209006[0m  [38;2;32;148;243m-0.06829553[0m [38;2;32;148;243m-0.31987357[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32235494[0m [38;2;32;148;243m-0.16062696[0m [38;2;32;148;243m-0.05058669[0m [38;2;32;148;243m-0.24491162[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21032974[0m [38;2;32;148;243m-0.10209513[0m [38;2;32;148;243m-0.03288404[0m [38;2;32;148;243m-0.17223757[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09768913[0m [38;2;32;148;243m-0.04425193[0m [38;2;32;148;243m-0.01466259[0m [38;2;32;148;243m-0.10127176[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01450408[0m  [38;2;32;148;243m0.01453731[0m  [38;2;32;148;243m0.0028364[0m  [38;2;32;148;243m-0.03077477[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01450408[0m  [38;2;32;148;243m0.01453731[0m  [38;2;32;148;243m0.0028364[0m  [38;2;32;148;243m-0.03077477[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.258356095911575e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00516376[0m  [38;2;32;148;243m0.00676524[0m  [38;2;32;148;243m0.00537446[0m  [38;2;32;148;243m0.00314183[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00137951[0m  [38;2;32;148;243m0.00029458[0m [38;2;32;148;243m-0.00284336[0m [38;2;32;148;243m-0.00246712[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0402548[0m   [38;2;32;148;243m0.00859586[0m [38;2;32;148;243m-0.08297054[0m [38;2;32;148;243m-0.07199194[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.01094545[0m [38;2;32;148;243m0.00412099[0m [38;2;32;148;243m0.00106453[0m [38;2;32;148;243m0.00203444[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m3.75094787e-04[0m [38;2;32;148;243m1.41224095e-04[0m [38;2;32;148;243m3.64808917e-05[0m [38;2;32;148;243m6.97190191e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">21</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:26] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">22</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:26][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m262.81068213[0m [38;2;32;148;243m-221.53230965[0m  [38;2;32;148;243m-85.89856373[0m [38;2;32;148;243m-124.36697529[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.0466881[0m  [38;2;32;148;243m-225.56388349[0m [38;2;32;148;243m-103.67656562[0m [38;2;32;148;243m-137.33725402[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m243.97921278[0m [38;2;32;148;243m-225.5258324[0m  [38;2;32;148;243m-103.81282295[0m [38;2;32;148;243m-137.34152247[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m243.9295502[0m  [38;2;32;148;243m-225.4739091[0m  [38;2;32;148;243m-103.93755735[0m [38;2;32;148;243m-137.33042661[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m243.89742265[0m [38;2;32;148;243m-225.40839491[0m [38;2;32;148;243m-104.04975512[0m [38;2;32;148;243m-137.30418556[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.13875959[0m [38;2;32;148;243m-225.26466851[0m [38;2;32;148;243m-103.9470228[0m  [38;2;32;148;243m-137.18562375[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.39817786[0m [38;2;32;148;243m-225.10681087[0m [38;2;32;148;243m-103.83079731[0m [38;2;32;148;243m-137.05137395[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.67516855[0m [38;2;32;148;243m-224.93451782[0m [38;2;32;148;243m-103.70118309[0m [38;2;32;148;243m-136.90126371[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.97020268[0m [38;2;32;148;243m-224.74824293[0m [38;2;32;148;243m-103.55778437[0m [38;2;32;148;243m-136.73499005[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m262.81068213[0m [38;2;32;148;243m-221.53230965[0m  [38;2;32;148;243m-85.89856373[0m [38;2;32;148;243m-124.36697529[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.14750674[0m [38;2;32;148;243m-225.54050097[0m [38;2;32;148;243m-103.65166962[0m [38;2;32;148;243m-137.2466387[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m244.18664367[0m [38;2;32;148;243m-225.48139629[0m [38;2;32;148;243m-103.75803865[0m [38;2;32;148;243m-137.16439118[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.24333905[0m [38;2;32;148;243m-225.40797576[0m [38;2;32;148;243m-103.85609319[0m [38;2;32;148;243m-137.06719523[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.31720171[0m [38;2;32;148;243m-225.31967898[0m [38;2;32;148;243m-103.94353926[0m [38;2;32;148;243m-136.95526835[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.45239498[0m [38;2;32;148;243m-225.19812353[0m [38;2;32;148;243m-103.86575155[0m [38;2;32;148;243m-136.9220462[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m244.60599805[0m [38;2;32;148;243m-225.06183681[0m [38;2;32;148;243m-103.77287375[0m [38;2;32;148;243m-136.87291166[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.77792238[0m [38;2;32;148;243m-224.91099768[0m [38;2;32;148;243m-103.66528316[0m [38;2;32;148;243m-136.80800178[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m244.96821729[0m [38;2;32;148;243m-224.74552135[0m [38;2;32;148;243m-103.54270102[0m [38;2;32;148;243m-136.72752315[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10081864[0m [38;2;32;148;243m-0.02338253[0m [38;2;32;148;243m-0.024896[0m   [38;2;32;148;243m-0.09061532[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20743089[0m [38;2;32;148;243m-0.04443612[0m [38;2;32;148;243m-0.0547843[0m  [38;2;32;148;243m-0.17713129[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31378886[0m [38;2;32;148;243m-0.06593334[0m [38;2;32;148;243m-0.08146416[0m [38;2;32;148;243m-0.26323138[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.41977907[0m [38;2;32;148;243m-0.08871593[0m [38;2;32;148;243m-0.10621586[0m [38;2;32;148;243m-0.34891721[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3136354[0m  [38;2;32;148;243m-0.06654498[0m [38;2;32;148;243m-0.08127125[0m [38;2;32;148;243m-0.26357756[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20782019[0m [38;2;32;148;243m-0.04497407[0m [38;2;32;148;243m-0.05792356[0m [38;2;32;148;243m-0.17846229[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10275383[0m [38;2;32;148;243m-0.02352013[0m [38;2;32;148;243m-0.03589992[0m [38;2;32;148;243m-0.09326193[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00198538[0m [38;2;32;148;243m-0.00272157[0m [38;2;32;148;243m-0.01508335[0m [38;2;32;148;243m-0.0074669[0m [1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10081864[0m [38;2;32;148;243m-0.02338253[0m [38;2;32;148;243m-0.024896[0m   [38;2;32;148;243m-0.09061532[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20743089[0m [38;2;32;148;243m-0.04443612[0m [38;2;32;148;243m-0.0547843[0m  [38;2;32;148;243m-0.17713129[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31378886[0m [38;2;32;148;243m-0.06593334[0m [38;2;32;148;243m-0.08146416[0m [38;2;32;148;243m-0.26323138[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.41977907[0m [38;2;32;148;243m-0.08871593[0m [38;2;32;148;243m-0.10621586[0m [38;2;32;148;243m-0.34891721[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10614367[0m [38;2;32;148;243m0.02217095[0m [38;2;32;148;243m0.02494461[0m [38;2;32;148;243m0.08533966[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21195887[0m [38;2;32;148;243m0.04374187[0m [38;2;32;148;243m0.0482923[0m  [38;2;32;148;243m0.17045493[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.31702524[0m [38;2;32;148;243m0.0651958[0m  [38;2;32;148;243m0.07031594[0m [38;2;32;148;243m0.25565528[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.42176445[0m [38;2;32;148;243m0.08599436[0m [38;2;32;148;243m0.09113251[0m [38;2;32;148;243m0.34145031[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10081864[0m [38;2;32;148;243m-0.02338253[0m [38;2;32;148;243m-0.024896[0m   [38;2;32;148;243m-0.09061532[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20743089[0m [38;2;32;148;243m-0.04443612[0m [38;2;32;148;243m-0.0547843[0m  [38;2;32;148;243m-0.17713129[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31378886[0m [38;2;32;148;243m-0.06593334[0m [38;2;32;148;243m-0.08146416[0m [38;2;32;148;243m-0.26323138[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.41977907[0m [38;2;32;148;243m-0.08871593[0m [38;2;32;148;243m-0.10621586[0m [38;2;32;148;243m-0.34891721[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3136354[0m  [38;2;32;148;243m-0.06654498[0m [38;2;32;148;243m-0.08127125[0m [38;2;32;148;243m-0.26357756[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20782019[0m [38;2;32;148;243m-0.04497407[0m [38;2;32;148;243m-0.05792356[0m [38;2;32;148;243m-0.17846229[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10275383[0m [38;2;32;148;243m-0.02352013[0m [38;2;32;148;243m-0.03589992[0m [38;2;32;148;243m-0.09326193[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00198538[0m [38;2;32;148;243m-0.00272157[0m [38;2;32;148;243m-0.01508335[0m [38;2;32;148;243m-0.0074669[0m [1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00198538[0m [38;2;32;148;243m-0.00272157[0m [38;2;32;148;243m-0.01508335[0m [38;2;32;148;243m-0.0074669[0m [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.203359341330946e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00490634[0m  [38;2;32;148;243m0.00699433[0m  [38;2;32;148;243m0.00557357[0m  [38;2;32;148;243m0.00328262[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00137312[0m  [38;2;32;148;243m0.00036101[0m [38;2;32;148;243m-0.0028145[0m  [38;2;32;148;243m-0.00263894[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04006824[0m  [38;2;32;148;243m0.01053447[0m [38;2;32;148;243m-0.0821286[0m  [38;2;32;148;243m-0.07700553[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00260182[0m [38;2;32;148;243m0.00359335[0m [38;2;32;148;243m0.00417618[0m [38;2;32;148;243m0.00081342[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m8.91629785e-05[0m [38;2;32;148;243m1.23142058e-04[0m [38;2;32;148;243m1.43115374e-04[0m [38;2;32;148;243m2.78753748e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">23</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:27] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">24</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:27][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m53.29549314[0m  [38;2;32;148;243m-58.76586695[0m [38;2;32;148;243m-244.18392685[0m  [38;2;32;148;243m-90.47824155[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.57995877[0m  [38;2;32;148;243m-62.69294363[0m [38;2;32;148;243m-262.28875795[0m [38;2;32;148;243m-104.3813051[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m34.47438861[0m  [38;2;32;148;243m-62.83474894[0m [38;2;32;148;243m-262.34322091[0m [38;2;32;148;243m-104.45432688[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.3856592[0m   [38;2;32;148;243m-62.9622492[0m  [38;2;32;148;243m-262.38208482[0m [38;2;32;148;243m-104.51525265[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.31450599[0m  [38;2;32;148;243m-63.07517595[0m [38;2;32;148;243m-262.40400217[0m [38;2;32;148;243m-104.56176039[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.47641167[0m  [38;2;32;148;243m-63.00415612[0m [38;2;32;148;243m-262.27248643[0m [38;2;32;148;243m-104.62725646[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.65607385[0m  [38;2;32;148;243m-62.91995838[0m [38;2;32;148;243m-262.12895522[0m [38;2;32;148;243m-104.676091[0m  [1m][0m
>      [1m[[0m  [38;2;32;148;243m34.85403748[0m  [38;2;32;148;243m-62.81988525[0m [38;2;32;148;243m-261.97383069[0m [38;2;32;148;243m-104.70795625[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m35.07040043[0m  [38;2;32;148;243m-62.70322397[0m [38;2;32;148;243m-261.80467069[0m [38;2;32;148;243m-104.72640427[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m53.29549314[0m  [38;2;32;148;243m-58.76586695[0m [38;2;32;148;243m-244.18392685[0m  [38;2;32;148;243m-90.47824155[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.68201044[0m  [38;2;32;148;243m-62.64211138[0m [38;2;32;148;243m-262.267517[0m   [38;2;32;148;243m-104.30220296[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.67827523[0m  [38;2;32;148;243m-62.71990458[0m [38;2;32;148;243m-262.29424895[0m [38;2;32;148;243m-104.29103486[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.69104203[0m  [38;2;32;148;243m-62.78368168[0m [38;2;32;148;243m-262.30681754[0m [38;2;32;148;243m-104.26609628[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.72116752[0m  [38;2;32;148;243m-62.83294363[0m [38;2;32;148;243m-262.30487786[0m [38;2;32;148;243m-104.22593336[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.78192114[0m  [38;2;32;148;243m-62.82211379[0m [38;2;32;148;243m-262.19717421[0m [38;2;32;148;243m-104.3771953[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m34.85994349[0m  [38;2;32;148;243m-62.79763485[0m [38;2;32;148;243m-262.07585442[0m [38;2;32;148;243m-104.51235589[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m34.95536497[0m  [38;2;32;148;243m-62.75964481[0m [38;2;32;148;243m-261.9409464[0m  [38;2;32;148;243m-104.63123031[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m35.06841863[0m  [38;2;32;148;243m-62.70751068[0m [38;2;32;148;243m-261.79191975[0m [38;2;32;148;243m-104.73542137[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10205166[0m [38;2;32;148;243m-0.05083225[0m [38;2;32;148;243m-0.02124095[0m [38;2;32;148;243m-0.07910213[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20388662[0m [38;2;32;148;243m-0.11484436[0m [38;2;32;148;243m-0.04897196[0m [38;2;32;148;243m-0.16329202[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30538284[0m [38;2;32;148;243m-0.17856752[0m [38;2;32;148;243m-0.07526728[0m [38;2;32;148;243m-0.24915637[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40666153[0m [38;2;32;148;243m-0.24223232[0m [38;2;32;148;243m-0.09912432[0m [38;2;32;148;243m-0.33582703[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30550947[0m [38;2;32;148;243m-0.18204233[0m [38;2;32;148;243m-0.07531222[0m [38;2;32;148;243m-0.25006116[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20386965[0m [38;2;32;148;243m-0.12232353[0m [38;2;32;148;243m-0.0531008[0m  [38;2;32;148;243m-0.16373511[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10132749[0m [38;2;32;148;243m-0.06024043[0m [38;2;32;148;243m-0.03288429[0m [38;2;32;148;243m-0.07672595[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.0019818[0m   [38;2;32;148;243m0.00428671[0m [38;2;32;148;243m-0.01275094[0m  [38;2;32;148;243m0.0090171[0m [1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10205166[0m [38;2;32;148;243m-0.05083225[0m [38;2;32;148;243m-0.02124095[0m [38;2;32;148;243m-0.07910213[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20388662[0m [38;2;32;148;243m-0.11484436[0m [38;2;32;148;243m-0.04897196[0m [38;2;32;148;243m-0.16329202[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30538284[0m [38;2;32;148;243m-0.17856752[0m [38;2;32;148;243m-0.07526728[0m [38;2;32;148;243m-0.24915637[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40666153[0m [38;2;32;148;243m-0.24223232[0m [38;2;32;148;243m-0.09912432[0m [38;2;32;148;243m-0.33582703[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10115205[0m [38;2;32;148;243m0.06018998[0m [38;2;32;148;243m0.0238121[0m  [38;2;32;148;243m0.08576588[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.20279188[0m [38;2;32;148;243m0.11990879[0m [38;2;32;148;243m0.04602352[0m [38;2;32;148;243m0.17209193[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.30533404[0m [38;2;32;148;243m0.18199188[0m [38;2;32;148;243m0.06624003[0m [38;2;32;148;243m0.25910108[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.40864333[0m [38;2;32;148;243m0.24651903[0m [38;2;32;148;243m0.08637338[0m [38;2;32;148;243m0.34484413[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10205166[0m [38;2;32;148;243m-0.05083225[0m [38;2;32;148;243m-0.02124095[0m [38;2;32;148;243m-0.07910213[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20388662[0m [38;2;32;148;243m-0.11484436[0m [38;2;32;148;243m-0.04897196[0m [38;2;32;148;243m-0.16329202[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30538284[0m [38;2;32;148;243m-0.17856752[0m [38;2;32;148;243m-0.07526728[0m [38;2;32;148;243m-0.24915637[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40666153[0m [38;2;32;148;243m-0.24223232[0m [38;2;32;148;243m-0.09912432[0m [38;2;32;148;243m-0.33582703[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30550947[0m [38;2;32;148;243m-0.18204233[0m [38;2;32;148;243m-0.07531222[0m [38;2;32;148;243m-0.25006116[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20386965[0m [38;2;32;148;243m-0.12232353[0m [38;2;32;148;243m-0.0531008[0m  [38;2;32;148;243m-0.16373511[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10132749[0m [38;2;32;148;243m-0.06024043[0m [38;2;32;148;243m-0.03288429[0m [38;2;32;148;243m-0.07672595[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.0019818[0m   [38;2;32;148;243m0.00428671[0m [38;2;32;148;243m-0.01275094[0m  [38;2;32;148;243m0.0090171[0m [1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0019818[0m   [38;2;32;148;243m0.00428671[0m [38;2;32;148;243m-0.01275094[0m  [38;2;32;148;243m0.0090171[0m [1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.199918126428102e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00464369[0m  [38;2;32;148;243m0.00702943[0m  [38;2;32;148;243m0.00568062[0m  [38;2;32;148;243m0.00344332[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00160992[0m  [38;2;32;148;243m0.0003247[0m  [38;2;32;148;243m-0.00290013[0m [38;2;32;148;243m-0.00234132[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04697816[0m  [38;2;32;148;243m0.00947497[0m [38;2;32;148;243m-0.08462732[0m [38;2;32;148;243m-0.06832074[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00490065[0m [38;2;32;148;243m0.00466015[0m [38;2;32;148;243m0.00849919[0m [38;2;32;148;243m0.00734464[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00016794[0m [38;2;32;148;243m0.0001597[0m  [38;2;32;148;243m0.00029126[0m [38;2;32;148;243m0.0002517[0m [1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">25</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:28] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">26</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:28][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m145.56268985[0m [38;2;32;148;243m-250.90061932[0m   [38;2;32;148;243m58.73238516[0m  [38;2;32;148;243m-73.52151453[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.27648138[0m [38;2;32;148;243m-254.11582955[0m   [38;2;32;148;243m40.89072731[0m  [38;2;32;148;243m-87.21799879[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.24154442[0m [38;2;32;148;243m-254.20254897[0m   [38;2;32;148;243m40.83512463[0m  [38;2;32;148;243m-87.32131832[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.2226123[0m  [38;2;32;148;243m-254.27667566[0m   [38;2;32;148;243m40.79727028[0m  [38;2;32;148;243m-87.41032588[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.22042926[0m [38;2;32;148;243m-254.33751424[0m   [38;2;32;148;243m40.77621894[0m  [38;2;32;148;243m-87.48500866[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.28697941[0m [38;2;32;148;243m-254.27357731[0m   [38;2;32;148;243m40.81122568[0m  [38;2;32;148;243m-87.29897074[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.37044421[0m [38;2;32;148;243m-254.19652122[0m   [38;2;32;148;243m40.86115183[0m  [38;2;32;148;243m-87.09686641[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.4704619[0m  [38;2;32;148;243m-254.10639685[0m   [38;2;32;148;243m40.9265964[0m   [38;2;32;148;243m-86.87836757[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.58721126[0m [38;2;32;148;243m-254.00235699[0m   [38;2;32;148;243m41.00782063[0m  [38;2;32;148;243m-86.64472854[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m145.56268985[0m [38;2;32;148;243m-250.90061932[0m   [38;2;32;148;243m58.73238516[0m  [38;2;32;148;243m-73.52151453[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.36641975[0m [38;2;32;148;243m-254.092928[0m     [38;2;32;148;243m40.90759934[0m  [38;2;32;148;243m-87.1375361[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m126.42468866[0m [38;2;32;148;243m-254.14419272[0m   [38;2;32;148;243m40.86953976[0m  [38;2;32;148;243m-87.1632774[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m126.50060579[0m [38;2;32;148;243m-254.18180556[0m   [38;2;32;148;243m40.84666217[0m  [38;2;32;148;243m-87.17378558[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.59386801[0m [38;2;32;148;243m-254.20585361[0m   [38;2;32;148;243m40.83913233[0m  [38;2;32;148;243m-87.16897545[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.56524509[0m [38;2;32;148;243m-254.18121627[0m   [38;2;32;148;243m40.86224539[0m  [38;2;32;148;243m-87.06220381[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.55353401[0m [38;2;32;148;243m-254.14693932[0m   [38;2;32;148;243m40.90091196[0m  [38;2;32;148;243m-86.94020992[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.55866354[0m [38;2;32;148;243m-254.10066525[0m   [38;2;32;148;243m40.95487408[0m  [38;2;32;148;243m-86.80282657[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m126.58095921[0m [38;2;32;148;243m-254.03758118[0m   [38;2;32;148;243m41.02396249[0m  [38;2;32;148;243m-86.65028484[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.08993836[0m [38;2;32;148;243m-0.02290155[0m [38;2;32;148;243m-0.01687203[0m [38;2;32;148;243m-0.0804627[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18314424[0m [38;2;32;148;243m-0.05835625[0m [38;2;32;148;243m-0.03441513[0m [38;2;32;148;243m-0.15804092[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27799349[0m [38;2;32;148;243m-0.0948701[0m  [38;2;32;148;243m-0.04939189[0m [38;2;32;148;243m-0.23654031[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.37343874[0m [38;2;32;148;243m-0.13166064[0m [38;2;32;148;243m-0.06291339[0m [38;2;32;148;243m-0.31603321[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27826568[0m [38;2;32;148;243m-0.09236104[0m [38;2;32;148;243m-0.05101971[0m [38;2;32;148;243m-0.23676693[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1830898[0m  [38;2;32;148;243m-0.0495819[0m  [38;2;32;148;243m-0.03976013[0m [38;2;32;148;243m-0.15665649[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.08820164[0m [38;2;32;148;243m-0.0057316[0m  [38;2;32;148;243m-0.02827768[0m [38;2;32;148;243m-0.075541[0m  [1m][0m
>      [1m[[0m [38;2;32;148;243m0.00625205[0m  [38;2;32;148;243m0.03522419[0m [38;2;32;148;243m-0.01614186[0m  [38;2;32;148;243m0.00555631[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.08993836[0m [38;2;32;148;243m-0.02290155[0m [38;2;32;148;243m-0.01687203[0m [38;2;32;148;243m-0.0804627[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18314424[0m [38;2;32;148;243m-0.05835625[0m [38;2;32;148;243m-0.03441513[0m [38;2;32;148;243m-0.15804092[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27799349[0m [38;2;32;148;243m-0.0948701[0m  [38;2;32;148;243m-0.04939189[0m [38;2;32;148;243m-0.23654031[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.37343874[0m [38;2;32;148;243m-0.13166064[0m [38;2;32;148;243m-0.06291339[0m [38;2;32;148;243m-0.31603321[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.09517307[0m [38;2;32;148;243m0.03929959[0m [38;2;32;148;243m0.01189369[0m [38;2;32;148;243m0.07926628[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.19034894[0m [38;2;32;148;243m0.08207873[0m [38;2;32;148;243m0.02315326[0m [38;2;32;148;243m0.15937672[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.2852371[0m  [38;2;32;148;243m0.12592904[0m [38;2;32;148;243m0.03463571[0m [38;2;32;148;243m0.24049221[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.37969079[0m [38;2;32;148;243m0.16688482[0m [38;2;32;148;243m0.04677153[0m [38;2;32;148;243m0.32158951[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.08993836[0m [38;2;32;148;243m-0.02290155[0m [38;2;32;148;243m-0.01687203[0m [38;2;32;148;243m-0.0804627[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18314424[0m [38;2;32;148;243m-0.05835625[0m [38;2;32;148;243m-0.03441513[0m [38;2;32;148;243m-0.15804092[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27799349[0m [38;2;32;148;243m-0.0948701[0m  [38;2;32;148;243m-0.04939189[0m [38;2;32;148;243m-0.23654031[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.37343874[0m [38;2;32;148;243m-0.13166064[0m [38;2;32;148;243m-0.06291339[0m [38;2;32;148;243m-0.31603321[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27826568[0m [38;2;32;148;243m-0.09236104[0m [38;2;32;148;243m-0.05101971[0m [38;2;32;148;243m-0.23676693[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.1830898[0m  [38;2;32;148;243m-0.0495819[0m  [38;2;32;148;243m-0.03976013[0m [38;2;32;148;243m-0.15665649[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.08820164[0m [38;2;32;148;243m-0.0057316[0m  [38;2;32;148;243m-0.02827768[0m [38;2;32;148;243m-0.075541[0m  [1m][0m
>      [1m[[0m [38;2;32;148;243m0.00625205[0m  [38;2;32;148;243m0.03522419[0m [38;2;32;148;243m-0.01614186[0m  [38;2;32;148;243m0.00555631[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00625205[0m  [38;2;32;148;243m0.03522419[0m [38;2;32;148;243m-0.01614186[0m  [38;2;32;148;243m0.00555631[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.29218625393476e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00441948[0m  [38;2;32;148;243m0.00714393[0m  [38;2;32;148;243m0.00583822[0m  [38;2;32;148;243m0.00372959[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00167111[0m  [38;2;32;148;243m0.00027887[0m [38;2;32;148;243m-0.00268236[0m [38;2;32;148;243m-0.00261393[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04876376[0m  [38;2;32;148;243m0.00813762[0m [38;2;32;148;243m-0.07827249[0m [38;2;32;148;243m-0.07627567[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00064774[0m [38;2;32;148;243m0.00074784[0m [38;2;32;148;243m0.00113602[0m [38;2;32;148;243m0.00841213[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m2.21975696e-05[0m [38;2;32;148;243m2.56281275e-05[0m [38;2;32;148;243m3.89306736e-05[0m [38;2;32;148;243m2.88279188e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">27</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:29] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">28</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:29][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m171.15931665[0m   [38;2;32;148;243m15.3638884[0m  [38;2;32;148;243m-201.77147589[0m  [38;2;32;148;243m-93.17897621[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m151.88147376[0m   [38;2;32;148;243m11.79418133[0m [38;2;32;148;243m-218.98901622[0m [38;2;32;148;243m-106.07141897[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m151.85071746[0m   [38;2;32;148;243m11.78524517[0m [38;2;32;148;243m-218.9201588[0m  [38;2;32;148;243m-106.10069885[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m151.83872619[0m   [38;2;32;148;243m11.78844414[0m [38;2;32;148;243m-218.83784101[0m [38;2;32;148;243m-106.11383911[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m151.84627233[0m   [38;2;32;148;243m11.80707467[0m [38;2;32;148;243m-218.74149497[0m [38;2;32;148;243m-106.1110771[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m151.9395231[0m    [38;2;32;148;243m11.7743522[0m  [38;2;32;148;243m-218.71824208[0m [38;2;32;148;243m-105.92181133[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.05147054[0m   [38;2;32;148;243m11.75387696[0m [38;2;32;148;243m-218.67960987[0m [38;2;32;148;243m-105.71477454[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.18197841[0m   [38;2;32;148;243m11.74386838[0m [38;2;32;148;243m-218.62650409[0m [38;2;32;148;243m-105.49266828[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.33148059[0m   [38;2;32;148;243m11.74512414[0m [38;2;32;148;243m-218.56000431[0m [38;2;32;148;243m-105.25598557[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m171.15931665[0m   [38;2;32;148;243m15.3638884[0m  [38;2;32;148;243m-201.77147589[0m  [38;2;32;148;243m-93.17897621[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m151.98643516[0m   [38;2;32;148;243m11.82994944[0m [38;2;32;148;243m-218.9758738[0m  [38;2;32;148;243m-105.98897581[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.06660101[0m   [38;2;32;148;243m11.85246911[0m [38;2;32;148;243m-218.88795818[0m [38;2;32;148;243m-105.93663796[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.1655778[0m    [38;2;32;148;243m11.88824259[0m [38;2;32;148;243m-218.78497323[0m [38;2;32;148;243m-105.86945202[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.28377714[0m   [38;2;32;148;243m11.93838745[0m [38;2;32;148;243m-218.6679944[0m  [38;2;32;148;243m-105.78800194[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.26684222[0m   [38;2;32;148;243m11.87633823[0m [38;2;32;148;243m-218.66502018[0m [38;2;32;148;243m-105.67750282[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.26917284[0m   [38;2;32;148;243m11.82832167[0m [38;2;32;148;243m-218.64756474[0m [38;2;32;148;243m-105.55123251[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.29036215[0m   [38;2;32;148;243m11.79368232[0m [38;2;32;148;243m-218.61564241[0m [38;2;32;148;243m-105.40984507[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m152.33036729[0m   [38;2;32;148;243m11.77296525[0m [38;2;32;148;243m-218.56916079[0m [38;2;32;148;243m-105.25359312[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10496139[0m [38;2;32;148;243m-0.03576811[0m [38;2;32;148;243m-0.01314242[0m [38;2;32;148;243m-0.08244316[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21588355[0m [38;2;32;148;243m-0.06722394[0m [38;2;32;148;243m-0.03220063[0m [38;2;32;148;243m-0.16406089[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32685161[0m [38;2;32;148;243m-0.09979845[0m [38;2;32;148;243m-0.05286778[0m [38;2;32;148;243m-0.24438709[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43750481[0m [38;2;32;148;243m-0.13131278[0m [38;2;32;148;243m-0.07350057[0m [38;2;32;148;243m-0.32307516[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32731913[0m [38;2;32;148;243m-0.10198602[0m [38;2;32;148;243m-0.05322189[0m [38;2;32;148;243m-0.24430852[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2177023[0m  [38;2;32;148;243m-0.07444471[0m [38;2;32;148;243m-0.03204513[0m [38;2;32;148;243m-0.16354203[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10838375[0m [38;2;32;148;243m-0.04981394[0m [38;2;32;148;243m-0.01086168[0m [38;2;32;148;243m-0.0828232[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0.0011133[0m  [38;2;32;148;243m-0.02784111[0m  [38;2;32;148;243m0.00915648[0m [38;2;32;148;243m-0.00239244[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10496139[0m [38;2;32;148;243m-0.03576811[0m [38;2;32;148;243m-0.01314242[0m [38;2;32;148;243m-0.08244316[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21588355[0m [38;2;32;148;243m-0.06722394[0m [38;2;32;148;243m-0.03220063[0m [38;2;32;148;243m-0.16406089[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32685161[0m [38;2;32;148;243m-0.09979845[0m [38;2;32;148;243m-0.05286778[0m [38;2;32;148;243m-0.24438709[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43750481[0m [38;2;32;148;243m-0.13131278[0m [38;2;32;148;243m-0.07350057[0m [38;2;32;148;243m-0.32307516[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.11018568[0m [38;2;32;148;243m0.02932676[0m [38;2;32;148;243m0.02027868[0m [38;2;32;148;243m0.07876664[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21980251[0m [38;2;32;148;243m0.05686807[0m [38;2;32;148;243m0.04145544[0m [38;2;32;148;243m0.15953313[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.32912106[0m [38;2;32;148;243m0.08149885[0m [38;2;32;148;243m0.0626389[0m  [38;2;32;148;243m0.24025196[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.43861811[0m [38;2;32;148;243m0.10347167[0m [38;2;32;148;243m0.08265705[0m [38;2;32;148;243m0.32068272[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10496139[0m [38;2;32;148;243m-0.03576811[0m [38;2;32;148;243m-0.01314242[0m [38;2;32;148;243m-0.08244316[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21588355[0m [38;2;32;148;243m-0.06722394[0m [38;2;32;148;243m-0.03220063[0m [38;2;32;148;243m-0.16406089[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32685161[0m [38;2;32;148;243m-0.09979845[0m [38;2;32;148;243m-0.05286778[0m [38;2;32;148;243m-0.24438709[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43750481[0m [38;2;32;148;243m-0.13131278[0m [38;2;32;148;243m-0.07350057[0m [38;2;32;148;243m-0.32307516[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32731913[0m [38;2;32;148;243m-0.10198602[0m [38;2;32;148;243m-0.05322189[0m [38;2;32;148;243m-0.24430852[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2177023[0m  [38;2;32;148;243m-0.07444471[0m [38;2;32;148;243m-0.03204513[0m [38;2;32;148;243m-0.16354203[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10838375[0m [38;2;32;148;243m-0.04981394[0m [38;2;32;148;243m-0.01086168[0m [38;2;32;148;243m-0.0828232[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0.0011133[0m  [38;2;32;148;243m-0.02784111[0m  [38;2;32;148;243m0.00915648[0m [38;2;32;148;243m-0.00239244[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0011133[0m  [38;2;32;148;243m-0.02784111[0m  [38;2;32;148;243m0.00915648[0m [38;2;32;148;243m-0.00239244[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.292362240828847e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00431864[0m  [38;2;32;148;243m0.00723347[0m  [38;2;32;148;243m0.00602306[0m  [38;2;32;148;243m0.00394892[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00173084[0m  [38;2;32;148;243m0.0003554[0m  [38;2;32;148;243m-0.00279759[0m [38;2;32;148;243m-0.00244461[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.05050673[0m  [38;2;32;148;243m0.01037071[0m [38;2;32;148;243m-0.08163518[0m [38;2;32;148;243m-0.07133497[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00456065[0m [38;2;32;148;243m0.00044[0m    [38;2;32;148;243m0.00186585[0m [38;2;32;148;243m0.00040742[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.56291176e-04[0m [38;2;32;148;243m1.50785662e-05[0m [38;2;32;148;243m6.39416478e-05[0m [38;2;32;148;243m1.39619870e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">29</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:30] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">30</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:30][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m184.34203524[0m [38;2;32;148;243m-103.47477517[0m  [38;2;32;148;243m-46.02384416[0m  [38;2;32;148;243m178.62854614[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.39374778[0m [38;2;32;148;243m-106.90385803[0m  [38;2;32;148;243m-64.75457232[0m  [38;2;32;148;243m165.33959469[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.26513605[0m [38;2;32;148;243m-106.92917044[0m  [38;2;32;148;243m-64.84065599[0m  [38;2;32;148;243m165.37948522[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.15602636[0m [38;2;32;148;243m-106.93950103[0m  [38;2;32;148;243m-64.91302965[0m  [38;2;32;148;243m165.43531311[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.06680062[0m [38;2;32;148;243m-106.93442205[0m  [38;2;32;148;243m-64.97232719[0m  [38;2;32;148;243m165.50844993[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.32277271[0m [38;2;32;148;243m-106.8860604[0m   [38;2;32;148;243m-64.97473127[0m  [38;2;32;148;243m165.60670896[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.59922593[0m [38;2;32;148;243m-106.82227902[0m  [38;2;32;148;243m-64.96399892[0m  [38;2;32;148;243m165.72146782[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.89629594[0m [38;2;32;148;243m-106.74322528[0m  [38;2;32;148;243m-64.93942739[0m  [38;2;32;148;243m165.85356863[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m165.21408407[0m [38;2;32;148;243m-106.64911548[0m  [38;2;32;148;243m-64.90107888[0m  [38;2;32;148;243m166.00337226[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m184.34203524[0m [38;2;32;148;243m-103.47477517[0m  [38;2;32;148;243m-46.02384416[0m  [38;2;32;148;243m178.62854614[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.49729319[0m [38;2;32;148;243m-106.88829307[0m  [38;2;32;148;243m-64.7255811[0m   [38;2;32;148;243m165.43337029[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.48182743[0m [38;2;32;148;243m-106.90023352[0m  [38;2;32;148;243m-64.7705743[0m   [38;2;32;148;243m165.56511887[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.48637093[0m [38;2;32;148;243m-106.89777808[0m  [38;2;32;148;243m-64.80198698[0m  [38;2;32;148;243m165.71206936[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.5109001[0m  [38;2;32;148;243m-106.88084364[0m  [38;2;32;148;243m-64.81951543[0m  [38;2;32;148;243m165.87682369[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.65273741[0m [38;2;32;148;243m-106.84415659[0m  [38;2;32;148;243m-64.86410283[0m  [38;2;32;148;243m165.88372877[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m164.81423185[0m [38;2;32;148;243m-106.79258864[0m  [38;2;32;148;243m-64.89568017[0m  [38;2;32;148;243m165.907229[0m  [1m][0m
>      [1m[[0m [38;2;32;148;243m164.99581419[0m [38;2;32;148;243m-106.72681573[0m  [38;2;32;148;243m-64.91397325[0m  [38;2;32;148;243m165.94707031[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m165.19745045[0m [38;2;32;148;243m-106.64633033[0m  [38;2;32;148;243m-64.91893936[0m  [38;2;32;148;243m166.00303581[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.03545409e-01[0m [38;2;32;148;243m-1.55649636e-02[0m [38;2;32;148;243m-2.89912179e-02[0m [38;2;32;148;243m-9.37756002e-02[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.16691378e-01[0m [38;2;32;148;243m-2.89369247e-02[0m [38;2;32;148;243m-7.00816870e-02[0m [38;2;32;148;243m-1.85633657e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.30344565e-01[0m [38;2;32;148;243m-4.17229547e-02[0m [38;2;32;148;243m-1.11042663e-01[0m [38;2;32;148;243m-2.76756255e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-4.44099477e-01[0m [38;2;32;148;243m-5.35784063e-02[0m [38;2;32;148;243m-1.52811765e-01[0m [38;2;32;148;243m-3.68373758e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.29964703e-01[0m [38;2;32;148;243m-4.19038148e-02[0m [38;2;32;148;243m-1.10628446e-01[0m [38;2;32;148;243m-2.77019806e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.15005919e-01[0m [38;2;32;148;243m-2.96903798e-02[0m [38;2;32;148;243m-6.83187461e-02[0m [38;2;32;148;243m-1.85761180e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.95182431e-02[0m [38;2;32;148;243m-1.64095427e-02[0m [38;2;32;148;243m-2.54541349e-02[0m [38;2;32;148;243m-9.35016869e-02[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m1.66336164e-02[0m [38;2;32;148;243m-2.78514421e-03[0m  [38;2;32;148;243m1.78604829e-02[0m  [38;2;32;148;243m3.36451029e-04[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10354541[0m [38;2;32;148;243m-0.01556496[0m [38;2;32;148;243m-0.02899122[0m [38;2;32;148;243m-0.0937756[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21669138[0m [38;2;32;148;243m-0.02893692[0m [38;2;32;148;243m-0.07008169[0m [38;2;32;148;243m-0.18563366[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.33034456[0m [38;2;32;148;243m-0.04172295[0m [38;2;32;148;243m-0.11104266[0m [38;2;32;148;243m-0.27675626[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.44409948[0m [38;2;32;148;243m-0.05357841[0m [38;2;32;148;243m-0.15281176[0m [38;2;32;148;243m-0.36837376[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.11413477[0m [38;2;32;148;243m0.01167459[0m [38;2;32;148;243m0.04218332[0m [38;2;32;148;243m0.09135395[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.22909356[0m [38;2;32;148;243m0.02388803[0m [38;2;32;148;243m0.08449302[0m [38;2;32;148;243m0.18261258[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.34458123[0m [38;2;32;148;243m0.03716886[0m [38;2;32;148;243m0.12735763[0m [38;2;32;148;243m0.27487207[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.46073309[0m [38;2;32;148;243m0.05079326[0m [38;2;32;148;243m0.17067225[0m [38;2;32;148;243m0.36871021[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m  [38;2;32;148;243m0.00000000e+00[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-1.03545409e-01[0m [38;2;32;148;243m-1.55649636e-02[0m [38;2;32;148;243m-2.89912179e-02[0m [38;2;32;148;243m-9.37756002e-02[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.16691378e-01[0m [38;2;32;148;243m-2.89369247e-02[0m [38;2;32;148;243m-7.00816870e-02[0m [38;2;32;148;243m-1.85633657e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.30344565e-01[0m [38;2;32;148;243m-4.17229547e-02[0m [38;2;32;148;243m-1.11042663e-01[0m [38;2;32;148;243m-2.76756255e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-4.44099477e-01[0m [38;2;32;148;243m-5.35784063e-02[0m [38;2;32;148;243m-1.52811765e-01[0m [38;2;32;148;243m-3.68373758e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-3.29964703e-01[0m [38;2;32;148;243m-4.19038148e-02[0m [38;2;32;148;243m-1.10628446e-01[0m [38;2;32;148;243m-2.77019806e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-2.15005919e-01[0m [38;2;32;148;243m-2.96903798e-02[0m [38;2;32;148;243m-6.83187461e-02[0m [38;2;32;148;243m-1.85761180e-01[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-9.95182431e-02[0m [38;2;32;148;243m-1.64095427e-02[0m [38;2;32;148;243m-2.54541349e-02[0m [38;2;32;148;243m-9.35016869e-02[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m1.66336164e-02[0m [38;2;32;148;243m-2.78514421e-03[0m  [38;2;32;148;243m1.78604829e-02[0m  [38;2;32;148;243m3.36451029e-04[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01663362[0m [38;2;32;148;243m-0.00278514[0m  [38;2;32;148;243m0.01786048[0m  [38;2;32;148;243m0.00033645[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.241730249547891e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00392473[0m  [38;2;32;148;243m0.00744468[0m  [38;2;32;148;243m0.00624059[0m  [38;2;32;148;243m0.00392888[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00157616[0m  [38;2;32;148;243m0.00043167[0m [38;2;32;148;243m-0.00271804[0m [38;2;32;148;243m-0.0023798[0m [1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04599303[0m  [38;2;32;148;243m0.01259622[0m [38;2;32;148;243m-0.07931388[0m [38;2;32;148;243m-0.06944374[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00434508[0m [38;2;32;148;243m0.00624341[0m [38;2;32;148;243m0.00080301[0m [38;2;32;148;243m0.00256726[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.48903642e-04[0m [38;2;32;148;243m2.13958444e-04[0m [38;2;32;148;243m2.75186210e-05[0m [38;2;32;148;243m8.79785320e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">31</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:31] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">32</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:31][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m50.52861395[0m [38;2;32;148;243m-178.41428574[0m   [38;2;32;148;243m20.79589968[0m   [38;2;32;148;243m76.51722088[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.68718719[0m [38;2;32;148;243m-181.9700319[0m     [38;2;32;148;243m2.48696532[0m   [38;2;32;148;243m64.31002245[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.56933446[0m [38;2;32;148;243m-181.85597192[0m    [38;2;32;148;243m2.39757479[0m   [38;2;32;148;243m64.28251078[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.47024408[0m [38;2;32;148;243m-181.72791593[0m    [38;2;32;148;243m2.32202127[0m   [38;2;32;148;243m64.26822319[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.39068275[0m [38;2;32;148;243m-181.58547459[0m    [38;2;32;148;243m2.26278958[0m   [38;2;32;148;243m64.26821479[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.41953563[0m [38;2;32;148;243m-181.59829776[0m    [38;2;32;148;243m2.330283[0m     [38;2;32;148;243m64.40132027[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.46732514[0m [38;2;32;148;243m-181.59647438[0m    [38;2;32;148;243m2.41208281[0m   [38;2;32;148;243m64.54989607[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.53428071[0m [38;2;32;148;243m-181.58025227[0m    [38;2;32;148;243m2.50768149[0m   [38;2;32;148;243m64.71337351[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.62024889[0m [38;2;32;148;243m-181.55003025[0m    [38;2;32;148;243m2.61772195[0m   [38;2;32;148;243m64.89076471[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m50.52861395[0m [38;2;32;148;243m-178.41428574[0m   [38;2;32;148;243m20.79589968[0m   [38;2;32;148;243m76.51722088[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.79259493[0m [38;2;32;148;243m-181.94913384[0m    [38;2;32;148;243m2.52072261[0m   [38;2;32;148;243m64.38750977[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.78516527[0m [38;2;32;148;243m-181.81811405[0m    [38;2;32;148;243m2.47226402[0m   [38;2;32;148;243m64.43159795[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.79691782[0m [38;2;32;148;243m-181.67298377[0m    [38;2;32;148;243m2.43749044[0m   [38;2;32;148;243m64.48994943[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.82821999[0m [38;2;32;148;243m-181.51333533[0m    [38;2;32;148;243m2.41778558[0m   [38;2;32;148;243m64.56256728[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.7459287[0m  [38;2;32;148;243m-181.54410271[0m    [38;2;32;148;243m2.44612143[0m   [38;2;32;148;243m64.62293961[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.68204797[0m [38;2;32;148;243m-181.56005962[0m    [38;2;32;148;243m2.48865531[0m   [38;2;32;148;243m64.698162[0m  [1m][0m
>      [1m[[0m  [38;2;32;148;243m30.63676858[0m [38;2;32;148;243m-181.561427[0m      [38;2;32;148;243m2.54493974[0m   [38;2;32;148;243m64.78863651[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m30.61013569[0m [38;2;32;148;243m-181.54824353[0m    [38;2;32;148;243m2.6148425[0m    [38;2;32;148;243m64.89438964[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10540775[0m [38;2;32;148;243m-0.02089805[0m [38;2;32;148;243m-0.03375729[0m [38;2;32;148;243m-0.07748732[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21583081[0m [38;2;32;148;243m-0.03785787[0m [38;2;32;148;243m-0.07468923[0m [38;2;32;148;243m-0.14908717[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32667373[0m [38;2;32;148;243m-0.05493215[0m [38;2;32;148;243m-0.11546918[0m [38;2;32;148;243m-0.22172624[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43753723[0m [38;2;32;148;243m-0.07213926[0m [38;2;32;148;243m-0.154996[0m   [38;2;32;148;243m-0.29435249[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32639307[0m [38;2;32;148;243m-0.05419505[0m [38;2;32;148;243m-0.11583844[0m [38;2;32;148;243m-0.22161934[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21472283[0m [38;2;32;148;243m-0.03641476[0m [38;2;32;148;243m-0.0765725[0m  [38;2;32;148;243m-0.14826593[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10248787[0m [38;2;32;148;243m-0.01882527[0m [38;2;32;148;243m-0.03725825[0m [38;2;32;148;243m-0.075263[0m  [1m][0m
>      [1m[[0m [38;2;32;148;243m0.0101132[0m  [38;2;32;148;243m-0.00178672[0m  [38;2;32;148;243m0.00287945[0m [38;2;32;148;243m-0.00362494[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10540775[0m [38;2;32;148;243m-0.02089805[0m [38;2;32;148;243m-0.03375729[0m [38;2;32;148;243m-0.07748732[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21583081[0m [38;2;32;148;243m-0.03785787[0m [38;2;32;148;243m-0.07468923[0m [38;2;32;148;243m-0.14908717[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32667373[0m [38;2;32;148;243m-0.05493215[0m [38;2;32;148;243m-0.11546918[0m [38;2;32;148;243m-0.22172624[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43753723[0m [38;2;32;148;243m-0.07213926[0m [38;2;32;148;243m-0.154996[0m   [38;2;32;148;243m-0.29435249[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.11114417[0m [38;2;32;148;243m0.01794421[0m [38;2;32;148;243m0.03915757[0m [38;2;32;148;243m0.07273315[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.22281441[0m [38;2;32;148;243m0.03572449[0m [38;2;32;148;243m0.0784235[0m  [38;2;32;148;243m0.14608656[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.33504936[0m [38;2;32;148;243m0.05331399[0m [38;2;32;148;243m0.11773775[0m [38;2;32;148;243m0.21908949[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.44765043[0m [38;2;32;148;243m0.07035254[0m [38;2;32;148;243m0.15787545[0m [38;2;32;148;243m0.29072755[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10540775[0m [38;2;32;148;243m-0.02089805[0m [38;2;32;148;243m-0.03375729[0m [38;2;32;148;243m-0.07748732[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21583081[0m [38;2;32;148;243m-0.03785787[0m [38;2;32;148;243m-0.07468923[0m [38;2;32;148;243m-0.14908717[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32667373[0m [38;2;32;148;243m-0.05493215[0m [38;2;32;148;243m-0.11546918[0m [38;2;32;148;243m-0.22172624[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43753723[0m [38;2;32;148;243m-0.07213926[0m [38;2;32;148;243m-0.154996[0m   [38;2;32;148;243m-0.29435249[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32639307[0m [38;2;32;148;243m-0.05419505[0m [38;2;32;148;243m-0.11583844[0m [38;2;32;148;243m-0.22161934[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21472283[0m [38;2;32;148;243m-0.03641476[0m [38;2;32;148;243m-0.0765725[0m  [38;2;32;148;243m-0.14826593[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10248787[0m [38;2;32;148;243m-0.01882527[0m [38;2;32;148;243m-0.03725825[0m [38;2;32;148;243m-0.075263[0m  [1m][0m
>      [1m[[0m [38;2;32;148;243m0.0101132[0m  [38;2;32;148;243m-0.00178672[0m  [38;2;32;148;243m0.00287945[0m [38;2;32;148;243m-0.00362494[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0101132[0m  [38;2;32;148;243m-0.00178672[0m  [38;2;32;148;243m0.00287945[0m [38;2;32;148;243m-0.00362494[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.155062018783672e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00374981[0m  [38;2;32;148;243m0.00761301[0m  [38;2;32;148;243m0.00619221[0m  [38;2;32;148;243m0.0041061[0m [1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00154897[0m  [38;2;32;148;243m0.00034634[0m [38;2;32;148;243m-0.0027607[0m  [38;2;32;148;243m-0.00228755[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04519968[0m  [38;2;32;148;243m0.01010646[0m [38;2;32;148;243m-0.08055862[0m [38;2;32;148;243m-0.06675195[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00463783[0m [38;2;32;148;243m0.00145371[0m [38;2;32;148;243m0.00271898[0m [38;2;32;148;243m0.00451945[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.58935901e-04[0m [38;2;32;148;243m4.98178924e-05[0m [38;2;32;148;243m9.31778366e-05[0m [38;2;32;148;243m1.54879109e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">33</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:32] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">34</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:32][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m74.92880994[0m [38;2;32;148;243m-182.70511458[0m  [38;2;32;148;243m-39.69537499[0m  [38;2;32;148;243m-27.25250162[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.79239625[0m [38;2;32;148;243m-186.09473615[0m  [38;2;32;148;243m-57.22733036[0m  [38;2;32;148;243m-40.41684613[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.67299931[0m [38;2;32;148;243m-186.18214434[0m  [38;2;32;148;243m-57.18027841[0m  [38;2;32;148;243m-40.48199145[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.5705083[0m  [38;2;32;148;243m-186.2533408[0m   [38;2;32;148;243m-57.11750375[0m  [38;2;32;148;243m-40.5343663[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m55.48475137[0m [38;2;32;148;243m-186.30814703[0m  [38;2;32;148;243m-57.03990315[0m  [38;2;32;148;243m-40.56818088[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.72676428[0m [38;2;32;148;243m-186.29876425[0m  [38;2;32;148;243m-57.01873247[0m  [38;2;32;148;243m-40.47357665[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.98598735[0m [38;2;32;148;243m-186.27614725[0m  [38;2;32;148;243m-56.98330972[0m  [38;2;32;148;243m-40.36414313[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m56.26179828[0m [38;2;32;148;243m-186.24337663[0m  [38;2;32;148;243m-56.93307792[0m  [38;2;32;148;243m-40.24023115[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m56.55504835[0m [38;2;32;148;243m-186.19948866[0m  [38;2;32;148;243m-56.86766223[0m  [38;2;32;148;243m-40.09930656[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m74.92880994[0m [38;2;32;148;243m-182.70511458[0m  [38;2;32;148;243m-39.69537499[0m  [38;2;32;148;243m-27.25250162[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.89426131[0m [38;2;32;148;243m-186.04645579[0m  [38;2;32;148;243m-57.2091828[0m   [38;2;32;148;243m-40.34048656[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.87253804[0m [38;2;32;148;243m-186.07667698[0m  [38;2;32;148;243m-57.14775794[0m  [38;2;32;148;243m-40.33015975[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.86671099[0m [38;2;32;148;243m-186.09203927[0m  [38;2;32;148;243m-57.07307576[0m  [38;2;32;148;243m-40.30627388[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m55.87789084[0m [38;2;32;148;243m-186.09214799[0m  [38;2;32;148;243m-56.98304617[0m  [38;2;32;148;243m-40.2653189[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m56.0221772[0m  [38;2;32;148;243m-186.13770233[0m  [38;2;32;148;243m-56.97548894[0m  [38;2;32;148;243m-40.24410023[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m56.18357892[0m [38;2;32;148;243m-186.16823336[0m  [38;2;32;148;243m-56.95284355[0m  [38;2;32;148;243m-40.20777472[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m56.36188957[0m [38;2;32;148;243m-186.18267865[0m  [38;2;32;148;243m-56.91529636[0m  [38;2;32;148;243m-40.15668495[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m56.55740189[0m [38;2;32;148;243m-186.18156121[0m  [38;2;32;148;243m-56.86264086[0m  [38;2;32;148;243m-40.08977762[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10186505[0m [38;2;32;148;243m-0.04828036[0m [38;2;32;148;243m-0.01814756[0m [38;2;32;148;243m-0.07635958[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19953872[0m [38;2;32;148;243m-0.10546736[0m [38;2;32;148;243m-0.03252047[0m [38;2;32;148;243m-0.1518317[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29620269[0m [38;2;32;148;243m-0.16130153[0m [38;2;32;148;243m-0.04442799[0m [38;2;32;148;243m-0.22809242[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39313947[0m [38;2;32;148;243m-0.21599903[0m [38;2;32;148;243m-0.05685698[0m [38;2;32;148;243m-0.30286198[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29541292[0m [38;2;32;148;243m-0.16106192[0m [38;2;32;148;243m-0.04324352[0m [38;2;32;148;243m-0.22947642[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19759157[0m [38;2;32;148;243m-0.10791389[0m [38;2;32;148;243m-0.03046616[0m [38;2;32;148;243m-0.15636841[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10009129[0m [38;2;32;148;243m-0.06069797[0m [38;2;32;148;243m-0.01778156[0m [38;2;32;148;243m-0.08354619[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00235355[0m [38;2;32;148;243m-0.01792746[0m [38;2;32;148;243m-0.00502137[0m [38;2;32;148;243m-0.00952894[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10186505[0m [38;2;32;148;243m-0.04828036[0m [38;2;32;148;243m-0.01814756[0m [38;2;32;148;243m-0.07635958[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19953872[0m [38;2;32;148;243m-0.10546736[0m [38;2;32;148;243m-0.03252047[0m [38;2;32;148;243m-0.1518317[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29620269[0m [38;2;32;148;243m-0.16130153[0m [38;2;32;148;243m-0.04442799[0m [38;2;32;148;243m-0.22809242[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39313947[0m [38;2;32;148;243m-0.21599903[0m [38;2;32;148;243m-0.05685698[0m [38;2;32;148;243m-0.30286198[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.09772655[0m [38;2;32;148;243m0.05493711[0m [38;2;32;148;243m0.01361346[0m [38;2;32;148;243m0.07338556[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.1955479[0m  [38;2;32;148;243m0.10808514[0m [38;2;32;148;243m0.02639081[0m [38;2;32;148;243m0.14649357[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.29304819[0m [38;2;32;148;243m0.15530106[0m [38;2;32;148;243m0.03907542[0m [38;2;32;148;243m0.21931579[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.39078593[0m [38;2;32;148;243m0.19807158[0m [38;2;32;148;243m0.05183561[0m [38;2;32;148;243m0.29333304[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10186505[0m [38;2;32;148;243m-0.04828036[0m [38;2;32;148;243m-0.01814756[0m [38;2;32;148;243m-0.07635958[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19953872[0m [38;2;32;148;243m-0.10546736[0m [38;2;32;148;243m-0.03252047[0m [38;2;32;148;243m-0.1518317[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29620269[0m [38;2;32;148;243m-0.16130153[0m [38;2;32;148;243m-0.04442799[0m [38;2;32;148;243m-0.22809242[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39313947[0m [38;2;32;148;243m-0.21599903[0m [38;2;32;148;243m-0.05685698[0m [38;2;32;148;243m-0.30286198[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29541292[0m [38;2;32;148;243m-0.16106192[0m [38;2;32;148;243m-0.04324352[0m [38;2;32;148;243m-0.22947642[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19759157[0m [38;2;32;148;243m-0.10791389[0m [38;2;32;148;243m-0.03046616[0m [38;2;32;148;243m-0.15636841[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10009129[0m [38;2;32;148;243m-0.06069797[0m [38;2;32;148;243m-0.01778156[0m [38;2;32;148;243m-0.08354619[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.00235355[0m [38;2;32;148;243m-0.01792746[0m [38;2;32;148;243m-0.00502137[0m [38;2;32;148;243m-0.00952894[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00235355[0m [38;2;32;148;243m-0.01792746[0m [38;2;32;148;243m-0.00502137[0m [38;2;32;148;243m-0.00952894[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.247447757482707e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00354041[0m  [38;2;32;148;243m0.00763568[0m  [38;2;32;148;243m0.00653372[0m  [38;2;32;148;243m0.00412247[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0013728[0m   [38;2;32;148;243m0.0003291[0m  [38;2;32;148;243m-0.00278975[0m [38;2;32;148;243m-0.00219927[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04005891[0m  [38;2;32;148;243m0.00960323[0m [38;2;32;148;243m-0.08140624[0m [38;2;32;148;243m-0.0641757[0m [1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00467942[0m [38;2;32;148;243m0.00249438[0m [38;2;32;148;243m0.0030545[0m  [38;2;32;148;243m0.00207526[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.60361062e-04[0m [38;2;32;148;243m8.54808986e-05[0m [38;2;32;148;243m1.04676151e-04[0m [38;2;32;148;243m7.11178738e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">35</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:33] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">36</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:33][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m112.6789316[0m  [38;2;32;148;243m-137.6519396[0m    [38;2;32;148;243m37.65671306[0m [38;2;32;148;243m-127.87433435[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.48502218[0m [38;2;32;148;243m-141.14225385[0m   [38;2;32;148;243m19.30991479[0m [38;2;32;148;243m-140.16098179[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.31094607[0m [38;2;32;148;243m-141.08719399[0m   [38;2;32;148;243m19.31376414[0m [38;2;32;148;243m-140.16379303[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.15358912[0m [38;2;32;148;243m-141.01793853[0m   [38;2;32;148;243m19.33327433[0m [38;2;32;148;243m-140.15081693[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.01351514[0m [38;2;32;148;243m-140.93436227[0m   [38;2;32;148;243m19.36809026[0m [38;2;32;148;243m-140.12320538[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.05707812[0m [38;2;32;148;243m-140.90245986[0m   [38;2;32;148;243m19.468945[0m   [38;2;32;148;243m-140.04413358[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.1166652[0m  [38;2;32;148;243m-140.85705571[0m   [38;2;32;148;243m19.58659885[0m [38;2;32;148;243m-139.94935126[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.19322[0m    [38;2;32;148;243m-140.79827396[0m   [38;2;32;148;243m19.72094828[0m [38;2;32;148;243m-139.83858442[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.28646643[0m [38;2;32;148;243m-140.72621824[0m   [38;2;32;148;243m19.87126243[0m [38;2;32;148;243m-139.71186011[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m112.6789316[0m  [38;2;32;148;243m-137.6519396[0m    [38;2;32;148;243m37.65671306[0m [38;2;32;148;243m-127.87433435[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.58072677[0m [38;2;32;148;243m-141.10652899[0m   [38;2;32;148;243m19.33345276[0m [38;2;32;148;243m-140.07861647[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.49763661[0m [38;2;32;148;243m-141.00212212[0m   [38;2;32;148;243m19.35291461[0m [38;2;32;148;243m-139.99543486[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.43223269[0m [38;2;32;148;243m-140.88650419[0m   [38;2;32;148;243m19.38712363[0m [38;2;32;148;243m-139.89656251[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.38567723[0m [38;2;32;148;243m-140.75798639[0m   [38;2;32;148;243m19.43627117[0m [38;2;32;148;243m-139.78371247[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.33387416[0m [38;2;32;148;243m-140.77229949[0m   [38;2;32;148;243m19.52270199[0m [38;2;32;148;243m-139.7890863[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m92.29644736[0m [38;2;32;148;243m-140.77297197[0m   [38;2;32;148;243m19.62780006[0m [38;2;32;148;243m-139.78037698[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.27601996[0m [38;2;32;148;243m-140.76043489[0m   [38;2;32;148;243m19.75113299[0m [38;2;32;148;243m-139.75674949[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m92.27332048[0m [38;2;32;148;243m-140.73462654[0m   [38;2;32;148;243m19.89106076[0m [38;2;32;148;243m-139.7174389[0m [1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0957046[0m  [38;2;32;148;243m-0.03572485[0m [38;2;32;148;243m-0.02353797[0m [38;2;32;148;243m-0.08236532[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18669054[0m [38;2;32;148;243m-0.08507187[0m [38;2;32;148;243m-0.03915047[0m [38;2;32;148;243m-0.16835817[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27864358[0m [38;2;32;148;243m-0.13143433[0m [38;2;32;148;243m-0.0538493[0m  [38;2;32;148;243m-0.25425443[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3721621[0m  [38;2;32;148;243m-0.17637587[0m [38;2;32;148;243m-0.06818091[0m [38;2;32;148;243m-0.33949291[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27679604[0m [38;2;32;148;243m-0.13016037[0m [38;2;32;148;243m-0.053757[0m   [38;2;32;148;243m-0.25504728[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17978216[0m [38;2;32;148;243m-0.08408374[0m [38;2;32;148;243m-0.04120121[0m [38;2;32;148;243m-0.16897428[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.08279996[0m [38;2;32;148;243m-0.03783907[0m [38;2;32;148;243m-0.03018471[0m [38;2;32;148;243m-0.08183493[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01314594[0m  [38;2;32;148;243m0.0084083[0m  [38;2;32;148;243m-0.01979833[0m  [38;2;32;148;243m0.00557879[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0957046[0m  [38;2;32;148;243m-0.03572485[0m [38;2;32;148;243m-0.02353797[0m [38;2;32;148;243m-0.08236532[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18669054[0m [38;2;32;148;243m-0.08507187[0m [38;2;32;148;243m-0.03915047[0m [38;2;32;148;243m-0.16835817[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27864358[0m [38;2;32;148;243m-0.13143433[0m [38;2;32;148;243m-0.0538493[0m  [38;2;32;148;243m-0.25425443[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3721621[0m  [38;2;32;148;243m-0.17637587[0m [38;2;32;148;243m-0.06818091[0m [38;2;32;148;243m-0.33949291[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.09536606[0m [38;2;32;148;243m0.0462155[0m  [38;2;32;148;243m0.01442392[0m [38;2;32;148;243m0.08444563[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.19237994[0m [38;2;32;148;243m0.09229213[0m [38;2;32;148;243m0.0269797[0m  [38;2;32;148;243m0.17051863[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.28936214[0m [38;2;32;148;243m0.13853681[0m [38;2;32;148;243m0.0379962[0m  [38;2;32;148;243m0.25765798[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.38530804[0m [38;2;32;148;243m0.18478417[0m [38;2;32;148;243m0.04838258[0m [38;2;32;148;243m0.3450717[0m [1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.0957046[0m  [38;2;32;148;243m-0.03572485[0m [38;2;32;148;243m-0.02353797[0m [38;2;32;148;243m-0.08236532[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18669054[0m [38;2;32;148;243m-0.08507187[0m [38;2;32;148;243m-0.03915047[0m [38;2;32;148;243m-0.16835817[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27864358[0m [38;2;32;148;243m-0.13143433[0m [38;2;32;148;243m-0.0538493[0m  [38;2;32;148;243m-0.25425443[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3721621[0m  [38;2;32;148;243m-0.17637587[0m [38;2;32;148;243m-0.06818091[0m [38;2;32;148;243m-0.33949291[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27679604[0m [38;2;32;148;243m-0.13016037[0m [38;2;32;148;243m-0.053757[0m   [38;2;32;148;243m-0.25504728[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.17978216[0m [38;2;32;148;243m-0.08408374[0m [38;2;32;148;243m-0.04120121[0m [38;2;32;148;243m-0.16897428[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.08279996[0m [38;2;32;148;243m-0.03783907[0m [38;2;32;148;243m-0.03018471[0m [38;2;32;148;243m-0.08183493[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01314594[0m  [38;2;32;148;243m0.0084083[0m  [38;2;32;148;243m-0.01979833[0m  [38;2;32;148;243m0.00557879[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01314594[0m  [38;2;32;148;243m0.0084083[0m  [38;2;32;148;243m-0.01979833[0m  [38;2;32;148;243m0.00557879[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.248725889206352e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00357988[0m  [38;2;32;148;243m0.00777211[0m  [38;2;32;148;243m0.00669951[0m  [38;2;32;148;243m0.00430707[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00171328[0m  [38;2;32;148;243m0.00033568[0m [38;2;32;148;243m-0.00260917[0m [38;2;32;148;243m-0.00238742[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04999435[0m  [38;2;32;148;243m0.00979537[0m [38;2;32;148;243m-0.0761368[0m  [38;2;32;148;243m-0.06966602[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00540308[0m [38;2;32;148;243m0.00142743[0m [38;2;32;148;243m0.00242872[0m [38;2;32;148;243m0.00676989[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.85160666e-04[0m [38;2;32;148;243m4.89173385e-05[0m [38;2;32;148;243m8.32309639e-05[0m [38;2;32;148;243m2.32000474e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">37</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:34] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">38</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:34][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m44.88891258[0m [38;2;32;148;243m-203.76869411[0m [38;2;32;148;243m-297.67307078[0m [38;2;32;148;243m-111.9452155[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m24.72224401[0m [38;2;32;148;243m-207.45985592[0m [38;2;32;148;243m-316.47312288[0m [38;2;32;148;243m-124.67860952[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.5962375[0m  [38;2;32;148;243m-207.45053717[0m [38;2;32;148;243m-316.4805086[0m  [38;2;32;148;243m-124.77768652[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.48895162[0m [38;2;32;148;243m-207.42991488[0m [38;2;32;148;243m-316.47232253[0m [38;2;32;148;243m-124.8620405[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m24.39992623[0m [38;2;32;148;243m-207.39850667[0m [38;2;32;148;243m-316.44825811[0m [38;2;32;148;243m-124.93155662[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.54442062[0m [38;2;32;148;243m-207.36245976[0m [38;2;32;148;243m-316.42564629[0m [38;2;32;148;243m-124.80034997[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.70779707[0m [38;2;32;148;243m-207.31224841[0m [38;2;32;148;243m-316.38686339[0m [38;2;32;148;243m-124.65260372[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.89045669[0m [38;2;32;148;243m-207.24856859[0m [38;2;32;148;243m-316.33340037[0m [38;2;32;148;243m-124.48746799[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m25.09195134[0m [38;2;32;148;243m-207.17084009[0m [38;2;32;148;243m-316.26924213[0m [38;2;32;148;243m-124.30832288[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m44.88891258[0m [38;2;32;148;243m-203.76869411[0m [38;2;32;148;243m-297.67307078[0m [38;2;32;148;243m-111.9452155[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m24.82369672[0m [38;2;32;148;243m-207.43635382[0m [38;2;32;148;243m-316.45012639[0m [38;2;32;148;243m-124.60277519[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.8067966[0m  [38;2;32;148;243m-207.3947834[0m  [38;2;32;148;243m-316.42468676[0m [38;2;32;148;243m-124.62493316[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.80797489[0m [38;2;32;148;243m-207.33856477[0m [38;2;32;148;243m-316.3857758[0m  [38;2;32;148;243m-124.63191608[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.82757997[0m [38;2;32;148;243m-207.26767998[0m [38;2;32;148;243m-316.33307999[0m [38;2;32;148;243m-124.62345937[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.86317585[0m [38;2;32;148;243m-207.2735048[0m  [38;2;32;148;243m-316.33977392[0m [38;2;32;148;243m-124.570238[0m  [1m][0m
>      [1m[[0m  [38;2;32;148;243m24.9169385[0m  [38;2;32;148;243m-207.26600792[0m [38;2;32;148;243m-316.331626[0m   [38;2;32;148;243m-124.50130283[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.98971303[0m [38;2;32;148;243m-207.24698827[0m [38;2;32;148;243m-316.30938657[0m [38;2;32;148;243m-124.4163161[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m25.08210632[0m [38;2;32;148;243m-207.21502822[0m [38;2;32;148;243m-316.27460538[0m [38;2;32;148;243m-124.31645819[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10145271[0m [38;2;32;148;243m-0.0235021[0m  [38;2;32;148;243m-0.02299649[0m [38;2;32;148;243m-0.07583434[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21055909[0m [38;2;32;148;243m-0.05575377[0m [38;2;32;148;243m-0.05582184[0m [38;2;32;148;243m-0.15275336[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31902328[0m [38;2;32;148;243m-0.09135011[0m [38;2;32;148;243m-0.08654673[0m [38;2;32;148;243m-0.23012442[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.42765374[0m [38;2;32;148;243m-0.13082669[0m [38;2;32;148;243m-0.11517812[0m [38;2;32;148;243m-0.30809725[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31875523[0m [38;2;32;148;243m-0.08895496[0m [38;2;32;148;243m-0.08587237[0m [38;2;32;148;243m-0.23011197[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20914143[0m [38;2;32;148;243m-0.04624049[0m [38;2;32;148;243m-0.0552374[0m  [38;2;32;148;243m-0.15130089[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09925634[0m [38;2;32;148;243m-0.00158032[0m [38;2;32;148;243m-0.02401379[0m [38;2;32;148;243m-0.07115189[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00984501[0m  [38;2;32;148;243m0.04418814[0m  [38;2;32;148;243m0.00536325[0m  [38;2;32;148;243m0.00813531[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10145271[0m [38;2;32;148;243m-0.0235021[0m  [38;2;32;148;243m-0.02299649[0m [38;2;32;148;243m-0.07583434[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21055909[0m [38;2;32;148;243m-0.05575377[0m [38;2;32;148;243m-0.05582184[0m [38;2;32;148;243m-0.15275336[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31902328[0m [38;2;32;148;243m-0.09135011[0m [38;2;32;148;243m-0.08654673[0m [38;2;32;148;243m-0.23012442[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.42765374[0m [38;2;32;148;243m-0.13082669[0m [38;2;32;148;243m-0.11517812[0m [38;2;32;148;243m-0.30809725[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10889852[0m [38;2;32;148;243m0.04187173[0m [38;2;32;148;243m0.02930575[0m [38;2;32;148;243m0.07798527[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21851231[0m [38;2;32;148;243m0.0845862[0m  [38;2;32;148;243m0.05994072[0m [38;2;32;148;243m0.15679636[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.3283974[0m  [38;2;32;148;243m0.12924637[0m [38;2;32;148;243m0.09116432[0m [38;2;32;148;243m0.23694535[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.43749876[0m [38;2;32;148;243m0.17501483[0m [38;2;32;148;243m0.12054137[0m [38;2;32;148;243m0.31623255[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10145271[0m [38;2;32;148;243m-0.0235021[0m  [38;2;32;148;243m-0.02299649[0m [38;2;32;148;243m-0.07583434[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21055909[0m [38;2;32;148;243m-0.05575377[0m [38;2;32;148;243m-0.05582184[0m [38;2;32;148;243m-0.15275336[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31902328[0m [38;2;32;148;243m-0.09135011[0m [38;2;32;148;243m-0.08654673[0m [38;2;32;148;243m-0.23012442[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.42765374[0m [38;2;32;148;243m-0.13082669[0m [38;2;32;148;243m-0.11517812[0m [38;2;32;148;243m-0.30809725[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.31875523[0m [38;2;32;148;243m-0.08895496[0m [38;2;32;148;243m-0.08587237[0m [38;2;32;148;243m-0.23011197[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20914143[0m [38;2;32;148;243m-0.04624049[0m [38;2;32;148;243m-0.0552374[0m  [38;2;32;148;243m-0.15130089[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09925634[0m [38;2;32;148;243m-0.00158032[0m [38;2;32;148;243m-0.02401379[0m [38;2;32;148;243m-0.07115189[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00984501[0m  [38;2;32;148;243m0.04418814[0m  [38;2;32;148;243m0.00536325[0m  [38;2;32;148;243m0.00813531[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00984501[0m [38;2;32;148;243m0.04418814[0m [38;2;32;148;243m0.00536325[0m [38;2;32;148;243m0.00813531[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.194705207402027e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00332281[0m  [38;2;32;148;243m0.00789375[0m  [38;2;32;148;243m0.0067447[0m   [38;2;32;148;243m0.00430028[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00186687[0m  [38;2;32;148;243m0.00015468[0m [38;2;32;148;243m-0.00251314[0m [38;2;32;148;243m-0.00228563[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.05447631[0m  [38;2;32;148;243m0.00451359[0m [38;2;32;148;243m-0.0733346[0m  [38;2;32;148;243m-0.06669579[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00073056[0m [38;2;32;148;243m0.00145581[0m [38;2;32;148;243m0.00766154[0m [38;2;32;148;243m0.00384[0m   [1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m2.50358021e-05[0m [38;2;32;148;243m4.98899584e-05[0m [38;2;32;148;243m2.62556895e-04[0m [38;2;32;148;243m1.31594731e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">39</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:35] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">40</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:35][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m106.23528662[0m [38;2;32;148;243m-290.17062525[0m [38;2;32;148;243m-159.6201671[0m  [38;2;32;148;243m-189.98648922[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.52688055[0m [38;2;32;148;243m-293.9108656[0m  [38;2;32;148;243m-177.48515651[0m [38;2;32;148;243m-202.2620761[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m85.44024672[0m [38;2;32;148;243m-294.01300049[0m [38;2;32;148;243m-177.44132191[0m [38;2;32;148;243m-202.12991156[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.37026265[0m [38;2;32;148;243m-294.10072964[0m [38;2;32;148;243m-177.38028803[0m [38;2;32;148;243m-201.98193368[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.31699528[0m [38;2;32;148;243m-294.17204307[0m [38;2;32;148;243m-177.30478544[0m [38;2;32;148;243m-201.81823139[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.41639208[0m [38;2;32;148;243m-294.14967811[0m [38;2;32;148;243m-177.21605084[0m [38;2;32;148;243m-201.77056058[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.53433558[0m [38;2;32;148;243m-294.11341036[0m [38;2;32;148;243m-177.11594323[0m [38;2;32;148;243m-201.70800504[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.67081597[0m [38;2;32;148;243m-294.06376415[0m [38;2;32;148;243m-177.00296784[0m [38;2;32;148;243m-201.6293551[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m85.82586154[0m [38;2;32;148;243m-294.00074864[0m [38;2;32;148;243m-176.87646334[0m [38;2;32;148;243m-201.53455858[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m106.23528662[0m [38;2;32;148;243m-290.17062525[0m [38;2;32;148;243m-159.6201671[0m  [38;2;32;148;243m-189.98648922[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.62777361[0m [38;2;32;148;243m-293.89374808[0m [38;2;32;148;243m-177.44790123[0m [38;2;32;148;243m-202.18025038[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.6431835[0m  [38;2;32;148;243m-293.97496572[0m [38;2;32;148;243m-177.36083902[0m [38;2;32;148;243m-201.96654809[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.67588402[0m [38;2;32;148;243m-294.04011126[0m [38;2;32;148;243m-177.25894581[0m [38;2;32;148;243m-201.73726089[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.72654302[0m [38;2;32;148;243m-294.08316652[0m [38;2;32;148;243m-177.14315312[0m [38;2;32;148;243m-201.49251313[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.72027525[0m [38;2;32;148;243m-294.09045697[0m [38;2;32;148;243m-177.09408032[0m [38;2;32;148;243m-201.52553054[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.7316803[0m  [38;2;32;148;243m-294.08278775[0m [38;2;32;148;243m-177.03225268[0m [38;2;32;148;243m-201.54339144[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.76107002[0m [38;2;32;148;243m-294.06306558[0m [38;2;32;148;243m-176.95709584[0m [38;2;32;148;243m-201.54577551[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m85.80858922[0m [38;2;32;148;243m-294.03111542[0m [38;2;32;148;243m-176.8681786[0m  [38;2;32;148;243m-201.53272987[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10089306[0m [38;2;32;148;243m-0.01711752[0m [38;2;32;148;243m-0.03725528[0m [38;2;32;148;243m-0.08182572[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20293678[0m [38;2;32;148;243m-0.03803477[0m [38;2;32;148;243m-0.08048289[0m [38;2;32;148;243m-0.16336347[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30562137[0m [38;2;32;148;243m-0.06061839[0m [38;2;32;148;243m-0.12134221[0m [38;2;32;148;243m-0.24467278[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40954774[0m [38;2;32;148;243m-0.08887655[0m [38;2;32;148;243m-0.16163232[0m [38;2;32;148;243m-0.32571826[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30388316[0m [38;2;32;148;243m-0.05922114[0m [38;2;32;148;243m-0.12197052[0m [38;2;32;148;243m-0.24503005[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19734472[0m [38;2;32;148;243m-0.03062261[0m [38;2;32;148;243m-0.08369055[0m [38;2;32;148;243m-0.1646136[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09025406[0m [38;2;32;148;243m-0.00069856[0m [38;2;32;148;243m-0.045872[0m   [38;2;32;148;243m-0.08357959[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01727232[0m  [38;2;32;148;243m0.03036678[0m [38;2;32;148;243m-0.00828474[0m [38;2;32;148;243m-0.00182871[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10089306[0m [38;2;32;148;243m-0.01711752[0m [38;2;32;148;243m-0.03725528[0m [38;2;32;148;243m-0.08182572[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20293678[0m [38;2;32;148;243m-0.03803477[0m [38;2;32;148;243m-0.08048289[0m [38;2;32;148;243m-0.16336347[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30562137[0m [38;2;32;148;243m-0.06061839[0m [38;2;32;148;243m-0.12134221[0m [38;2;32;148;243m-0.24467278[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40954774[0m [38;2;32;148;243m-0.08887655[0m [38;2;32;148;243m-0.16163232[0m [38;2;32;148;243m-0.32571826[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10566457[0m [38;2;32;148;243m0.02965541[0m [38;2;32;148;243m0.0396618[0m  [38;2;32;148;243m0.08068821[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.21220301[0m [38;2;32;148;243m0.05825393[0m [38;2;32;148;243m0.07794177[0m [38;2;32;148;243m0.16110465[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.31929368[0m [38;2;32;148;243m0.08817798[0m [38;2;32;148;243m0.11576032[0m [38;2;32;148;243m0.24213867[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.42682006[0m [38;2;32;148;243m0.11924332[0m [38;2;32;148;243m0.15334758[0m [38;2;32;148;243m0.32388955[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10089306[0m [38;2;32;148;243m-0.01711752[0m [38;2;32;148;243m-0.03725528[0m [38;2;32;148;243m-0.08182572[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20293678[0m [38;2;32;148;243m-0.03803477[0m [38;2;32;148;243m-0.08048289[0m [38;2;32;148;243m-0.16336347[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30562137[0m [38;2;32;148;243m-0.06061839[0m [38;2;32;148;243m-0.12134221[0m [38;2;32;148;243m-0.24467278[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.40954774[0m [38;2;32;148;243m-0.08887655[0m [38;2;32;148;243m-0.16163232[0m [38;2;32;148;243m-0.32571826[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30388316[0m [38;2;32;148;243m-0.05922114[0m [38;2;32;148;243m-0.12197052[0m [38;2;32;148;243m-0.24503005[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19734472[0m [38;2;32;148;243m-0.03062261[0m [38;2;32;148;243m-0.08369055[0m [38;2;32;148;243m-0.1646136[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09025406[0m [38;2;32;148;243m-0.00069856[0m [38;2;32;148;243m-0.045872[0m   [38;2;32;148;243m-0.08357959[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.01727232[0m  [38;2;32;148;243m0.03036678[0m [38;2;32;148;243m-0.00828474[0m [38;2;32;148;243m-0.00182871[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.01727232[0m  [38;2;32;148;243m0.03036678[0m [38;2;32;148;243m-0.00828474[0m [38;2;32;148;243m-0.00182871[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.17654446569214e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00304076[0m  [38;2;32;148;243m0.00786173[0m  [38;2;32;148;243m0.00719686[0m  [38;2;32;148;243m0.00432946[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00173932[0m  [38;2;32;148;243m0.00026521[0m [38;2;32;148;243m-0.00291428[0m [38;2;32;148;243m-0.00230536[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0507541[0m   [38;2;32;148;243m0.00773885[0m [38;2;32;148;243m-0.08504023[0m [38;2;32;148;243m-0.06727142[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00199336[0m [38;2;32;148;243m0.00863884[0m [38;2;32;148;243m0.00459806[0m [38;2;32;148;243m0.00080989[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m6.83114666e-05[0m [38;2;32;148;243m2.96048251e-04[0m [38;2;32;148;243m1.57573049e-04[0m [38;2;32;148;243m2.77545685e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">41</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:36] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">42</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:36][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m45.08247249[0m [38;2;32;148;243m-218.94467148[0m  [38;2;32;148;243m-40.42705107[0m  [38;2;32;148;243m-45.87979538[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.88875382[0m [38;2;32;148;243m-221.68344916[0m  [38;2;32;148;243m-57.74491095[0m  [38;2;32;148;243m-58.84500785[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.72697675[0m [38;2;32;148;243m-221.74890706[0m  [38;2;32;148;243m-57.85437104[0m  [38;2;32;148;243m-58.87851347[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.58233039[0m [38;2;32;148;243m-221.80001817[0m  [38;2;32;148;243m-57.95455121[0m  [38;2;32;148;243m-58.89820102[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.45405754[0m [38;2;32;148;243m-221.83648852[0m  [38;2;32;148;243m-58.04099729[0m  [38;2;32;148;243m-58.90403978[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.6740094[0m  [38;2;32;148;243m-221.74836851[0m  [38;2;32;148;243m-57.87524269[0m  [38;2;32;148;243m-58.87469989[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.90952716[0m [38;2;32;148;243m-221.64615476[0m  [38;2;32;148;243m-57.69795089[0m  [38;2;32;148;243m-58.83046746[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m25.16189711[0m [38;2;32;148;243m-221.53004749[0m  [38;2;32;148;243m-57.50675058[0m  [38;2;32;148;243m-58.77083429[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m25.43115449[0m [38;2;32;148;243m-221.39974347[0m  [38;2;32;148;243m-57.30083285[0m  [38;2;32;148;243m-58.69653692[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m45.08247249[0m [38;2;32;148;243m-218.94467148[0m  [38;2;32;148;243m-40.42705107[0m  [38;2;32;148;243m-45.87979538[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.9834256[0m  [38;2;32;148;243m-221.67081513[0m  [38;2;32;148;243m-57.72006693[0m  [38;2;32;148;243m-58.76554207[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.91149183[0m [38;2;32;148;243m-221.71545445[0m  [38;2;32;148;243m-57.79256646[0m  [38;2;32;148;243m-58.72944584[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.85490919[0m [38;2;32;148;243m-221.74436731[0m  [38;2;32;148;243m-57.8533987[0m   [38;2;32;148;243m-58.6786763[0m [1m][0m
>      [1m[[0m  [38;2;32;148;243m24.81371571[0m [38;2;32;148;243m-221.75770916[0m  [38;2;32;148;243m-57.90064845[0m  [38;2;32;148;243m-58.61294603[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m24.94740352[0m [38;2;32;148;243m-221.69404354[0m  [38;2;32;148;243m-57.77196138[0m  [38;2;32;148;243m-58.65529608[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m25.09656817[0m [38;2;32;148;243m-221.61617355[0m  [38;2;32;148;243m-57.62947867[0m  [38;2;32;148;243m-58.68317078[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m25.26184398[0m [38;2;32;148;243m-221.52461979[0m  [38;2;32;148;243m-57.47290454[0m  [38;2;32;148;243m-58.69769633[0m[1m][0m
>      [1m[[0m  [38;2;32;148;243m25.44320293[0m [38;2;32;148;243m-221.41931996[0m  [38;2;32;148;243m-57.30269866[0m  [38;2;32;148;243m-58.69763577[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09467178[0m [38;2;32;148;243m-0.01263403[0m [38;2;32;148;243m-0.02484402[0m [38;2;32;148;243m-0.07946579[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18451508[0m [38;2;32;148;243m-0.03345261[0m [38;2;32;148;243m-0.06180458[0m [38;2;32;148;243m-0.14906764[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2725788[0m  [38;2;32;148;243m-0.05565086[0m [38;2;32;148;243m-0.1011525[0m  [38;2;32;148;243m-0.21952472[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.35965817[0m [38;2;32;148;243m-0.07877937[0m [38;2;32;148;243m-0.14034884[0m [38;2;32;148;243m-0.29109375[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27339412[0m [38;2;32;148;243m-0.05432498[0m [38;2;32;148;243m-0.10328132[0m [38;2;32;148;243m-0.21940382[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.187041[0m   [38;2;32;148;243m-0.02998121[0m [38;2;32;148;243m-0.06847222[0m [38;2;32;148;243m-0.14729668[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09994687[0m [38;2;32;148;243m-0.0054277[0m  [38;2;32;148;243m-0.03384605[0m [38;2;32;148;243m-0.07313796[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.01204845[0m  [38;2;32;148;243m0.01957649[0m  [38;2;32;148;243m0.00186582[0m  [38;2;32;148;243m0.00109885[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09467178[0m [38;2;32;148;243m-0.01263403[0m [38;2;32;148;243m-0.02484402[0m [38;2;32;148;243m-0.07946579[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18451508[0m [38;2;32;148;243m-0.03345261[0m [38;2;32;148;243m-0.06180458[0m [38;2;32;148;243m-0.14906764[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2725788[0m  [38;2;32;148;243m-0.05565086[0m [38;2;32;148;243m-0.1011525[0m  [38;2;32;148;243m-0.21952472[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.35965817[0m [38;2;32;148;243m-0.07877937[0m [38;2;32;148;243m-0.14034884[0m [38;2;32;148;243m-0.29109375[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.08626405[0m [38;2;32;148;243m0.02445439[0m [38;2;32;148;243m0.03706752[0m [38;2;32;148;243m0.07168994[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.17261717[0m [38;2;32;148;243m0.04879816[0m [38;2;32;148;243m0.07187662[0m [38;2;32;148;243m0.14379708[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.2597113[0m  [38;2;32;148;243m0.07335167[0m [38;2;32;148;243m0.10650279[0m [38;2;32;148;243m0.21795579[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.34760972[0m [38;2;32;148;243m0.09835586[0m [38;2;32;148;243m0.14221466[0m [38;2;32;148;243m0.2921926[0m [1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09467178[0m [38;2;32;148;243m-0.01263403[0m [38;2;32;148;243m-0.02484402[0m [38;2;32;148;243m-0.07946579[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.18451508[0m [38;2;32;148;243m-0.03345261[0m [38;2;32;148;243m-0.06180458[0m [38;2;32;148;243m-0.14906764[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2725788[0m  [38;2;32;148;243m-0.05565086[0m [38;2;32;148;243m-0.1011525[0m  [38;2;32;148;243m-0.21952472[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.35965817[0m [38;2;32;148;243m-0.07877937[0m [38;2;32;148;243m-0.14034884[0m [38;2;32;148;243m-0.29109375[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.27339412[0m [38;2;32;148;243m-0.05432498[0m [38;2;32;148;243m-0.10328132[0m [38;2;32;148;243m-0.21940382[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.187041[0m   [38;2;32;148;243m-0.02998121[0m [38;2;32;148;243m-0.06847222[0m [38;2;32;148;243m-0.14729668[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09994687[0m [38;2;32;148;243m-0.0054277[0m  [38;2;32;148;243m-0.03384605[0m [38;2;32;148;243m-0.07313796[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.01204845[0m  [38;2;32;148;243m0.01957649[0m  [38;2;32;148;243m0.00186582[0m  [38;2;32;148;243m0.00109885[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.01204845[0m  [38;2;32;148;243m0.01957649[0m  [38;2;32;148;243m0.00186582[0m  [38;2;32;148;243m0.00109885[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.235780692896638e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00284199[0m  [38;2;32;148;243m0.00793582[0m  [38;2;32;148;243m0.00717097[0m  [38;2;32;148;243m0.00465891[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.0016943[0m   [38;2;32;148;243m0.00023331[0m [38;2;32;148;243m-0.00275654[0m [38;2;32;148;243m-0.00244255[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04944038[0m  [38;2;32;148;243m0.00680803[0m [38;2;32;148;243m-0.08043712[0m [38;2;32;148;243m-0.07127484[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00743408[0m [38;2;32;148;243m0.0035648[0m  [38;2;32;148;243m0.00570171[0m [38;2;32;148;243m0.00128239[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m2.54762047e-04[0m [38;2;32;148;243m1.22163862e-04[0m [38;2;32;148;243m1.95394465e-04[0m [38;2;32;148;243m4.39469331e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">43</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:37] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">44</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:37][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m120.82488687[0m  [38;2;32;148;243m-80.38012025[0m [38;2;32;148;243m-117.15967936[0m  [38;2;32;148;243m-56.73565516[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.44898882[0m  [38;2;32;148;243m-83.71729572[0m [38;2;32;148;243m-134.59798425[0m  [38;2;32;148;243m-69.69895569[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.43184524[0m  [38;2;32;148;243m-83.84781282[0m [38;2;32;148;243m-134.73144051[0m  [38;2;32;148;243m-69.65093221[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.43349918[0m  [38;2;32;148;243m-83.96609418[0m [38;2;32;148;243m-134.8506462[0m   [38;2;32;148;243m-69.58857939[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.45425475[0m  [38;2;32;148;243m-84.07282237[0m [38;2;32;148;243m-134.95632605[0m  [38;2;32;148;243m-69.51176483[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.55769326[0m  [38;2;32;148;243m-84.02587068[0m [38;2;32;148;243m-135.11515693[0m  [38;2;32;148;243m-69.52300587[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.68058191[0m  [38;2;32;148;243m-83.96676659[0m [38;2;32;148;243m-135.26063242[0m  [38;2;32;148;243m-69.51917375[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.82265078[0m  [38;2;32;148;243m-83.89354142[0m [38;2;32;148;243m-135.39055237[0m  [38;2;32;148;243m-69.50092255[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.98405095[0m  [38;2;32;148;243m-83.80279629[0m [38;2;32;148;243m-135.50265454[0m  [38;2;32;148;243m-69.46834829[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m120.82488687[0m  [38;2;32;148;243m-80.38012025[0m [38;2;32;148;243m-117.15967936[0m  [38;2;32;148;243m-56.73565516[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.55466438[0m  [38;2;32;148;243m-83.68789026[0m [38;2;32;148;243m-134.57612433[0m  [38;2;32;148;243m-69.61969084[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.64720342[0m  [38;2;32;148;243m-83.78541926[0m [38;2;32;148;243m-134.67928772[0m  [38;2;32;148;243m-69.50125134[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.75908795[0m  [38;2;32;148;243m-83.8697308[0m  [38;2;32;148;243m-134.76893385[0m  [38;2;32;148;243m-69.36737332[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.88979462[0m  [38;2;32;148;243m-83.94078676[0m [38;2;32;148;243m-134.84457667[0m  [38;2;32;148;243m-69.21946834[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.88324237[0m  [38;2;32;148;243m-83.92860548[0m [38;2;32;148;243m-135.03380797[0m  [38;2;32;148;243m-69.30044316[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.89561884[0m  [38;2;32;148;243m-83.90316738[0m [38;2;32;148;243m-135.20938141[0m  [38;2;32;148;243m-69.36620395[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.92665192[0m  [38;2;32;148;243m-83.86385788[0m [38;2;32;148;243m-135.37009709[0m  [38;2;32;148;243m-69.41882816[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m100.97673539[0m  [38;2;32;148;243m-83.80878115[0m [38;2;32;148;243m-135.5148116[0m   [38;2;32;148;243m-69.45886986[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10567556[0m [38;2;32;148;243m-0.02940546[0m [38;2;32;148;243m-0.02185992[0m [38;2;32;148;243m-0.07926485[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21535818[0m [38;2;32;148;243m-0.06239356[0m [38;2;32;148;243m-0.05215279[0m [38;2;32;148;243m-0.14968088[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32558877[0m [38;2;32;148;243m-0.09636339[0m [38;2;32;148;243m-0.08171235[0m [38;2;32;148;243m-0.22120607[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43553987[0m [38;2;32;148;243m-0.13203561[0m [38;2;32;148;243m-0.11174938[0m [38;2;32;148;243m-0.29229649[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3255491[0m  [38;2;32;148;243m-0.0972652[0m  [38;2;32;148;243m-0.08134897[0m [38;2;32;148;243m-0.22256271[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21503693[0m [38;2;32;148;243m-0.0635992[0m  [38;2;32;148;243m-0.05125101[0m [38;2;32;148;243m-0.1529698[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10400114[0m [38;2;32;148;243m-0.02968355[0m [38;2;32;148;243m-0.02045528[0m [38;2;32;148;243m-0.0820944[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0.00731556[0m  [38;2;32;148;243m0.00598486[0m  [38;2;32;148;243m0.01215706[0m [38;2;32;148;243m-0.00947843[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10567556[0m [38;2;32;148;243m-0.02940546[0m [38;2;32;148;243m-0.02185992[0m [38;2;32;148;243m-0.07926485[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21535818[0m [38;2;32;148;243m-0.06239356[0m [38;2;32;148;243m-0.05215279[0m [38;2;32;148;243m-0.14968088[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32558877[0m [38;2;32;148;243m-0.09636339[0m [38;2;32;148;243m-0.08171235[0m [38;2;32;148;243m-0.22120607[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43553987[0m [38;2;32;148;243m-0.13203561[0m [38;2;32;148;243m-0.11174938[0m [38;2;32;148;243m-0.29229649[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10999077[0m [38;2;32;148;243m0.03477041[0m [38;2;32;148;243m0.03040042[0m [38;2;32;148;243m0.06973378[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.22050294[0m [38;2;32;148;243m0.06843641[0m [38;2;32;148;243m0.06049837[0m [38;2;32;148;243m0.1393267[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.33153873[0m [38;2;32;148;243m0.10235206[0m [38;2;32;148;243m0.0912941[0m  [38;2;32;148;243m0.21020209[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.44285543[0m [38;2;32;148;243m0.13802046[0m [38;2;32;148;243m0.12390645[0m [38;2;32;148;243m0.28281806[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10567556[0m [38;2;32;148;243m-0.02940546[0m [38;2;32;148;243m-0.02185992[0m [38;2;32;148;243m-0.07926485[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21535818[0m [38;2;32;148;243m-0.06239356[0m [38;2;32;148;243m-0.05215279[0m [38;2;32;148;243m-0.14968088[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.32558877[0m [38;2;32;148;243m-0.09636339[0m [38;2;32;148;243m-0.08171235[0m [38;2;32;148;243m-0.22120607[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.43553987[0m [38;2;32;148;243m-0.13203561[0m [38;2;32;148;243m-0.11174938[0m [38;2;32;148;243m-0.29229649[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.3255491[0m  [38;2;32;148;243m-0.0972652[0m  [38;2;32;148;243m-0.08134897[0m [38;2;32;148;243m-0.22256271[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.21503693[0m [38;2;32;148;243m-0.0635992[0m  [38;2;32;148;243m-0.05125101[0m [38;2;32;148;243m-0.1529698[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10400114[0m [38;2;32;148;243m-0.02968355[0m [38;2;32;148;243m-0.02045528[0m [38;2;32;148;243m-0.0820944[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0.00731556[0m  [38;2;32;148;243m0.00598486[0m  [38;2;32;148;243m0.01215706[0m [38;2;32;148;243m-0.00947843[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00731556[0m  [38;2;32;148;243m0.00598486[0m  [38;2;32;148;243m0.01215706[0m [38;2;32;148;243m-0.00947843[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.295091598065646e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00255669[0m  [38;2;32;148;243m0.00833024[0m  [38;2;32;148;243m0.00734708[0m  [38;2;32;148;243m0.00485666[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00154346[0m  [38;2;32;148;243m0.00020992[0m [38;2;32;148;243m-0.00263622[0m [38;2;32;148;243m-0.00245675[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04503892[0m  [38;2;32;148;243m0.00612545[0m [38;2;32;148;243m-0.07692619[0m [38;2;32;148;243m-0.07168927[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.01355264[0m [38;2;32;148;243m0.0006327[0m  [38;2;32;148;243m0.0003954[0m  [38;2;32;148;243m0.00245302[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m4.64441511e-04[0m [38;2;32;148;243m2.16821831e-05[0m [38;2;32;148;243m1.35499786e-05[0m [38;2;32;148;243m8.40638310e-05[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">45</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:38] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">46</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:38][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m10.15674593[0m [38;2;32;148;243m-229.7782156[0m  [38;2;32;148;243m-199.41260428[0m   [38;2;32;148;243m-7.53168592[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.22709283[0m [38;2;32;148;243m-232.61536672[0m [38;2;32;148;243m-218.57611715[0m  [38;2;32;148;243m-21.60021188[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.31483575[0m [38;2;32;148;243m-232.68137878[0m [38;2;32;148;243m-218.62675399[0m  [38;2;32;148;243m-21.6446776[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m-10.3851185[0m  [38;2;32;148;243m-232.73202897[0m [38;2;32;148;243m-218.66655071[0m  [38;2;32;148;243m-21.67507296[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.43760149[0m [38;2;32;148;243m-232.76664194[0m [38;2;32;148;243m-218.6949896[0m   [38;2;32;148;243m-21.69079443[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.39041625[0m [38;2;32;148;243m-232.76763588[0m [38;2;32;148;243m-218.7120165[0m   [38;2;32;148;243m-21.69326403[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.32384235[0m [38;2;32;148;243m-232.75399958[0m [38;2;32;148;243m-218.71590497[0m  [38;2;32;148;243m-21.6800835[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m-10.23898794[0m [38;2;32;148;243m-232.72707336[0m [38;2;32;148;243m-218.7065521[0m   [38;2;32;148;243m-21.65108882[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.13646712[0m [38;2;32;148;243m-232.68624622[0m [38;2;32;148;243m-218.68362209[0m  [38;2;32;148;243m-21.60739353[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m  [38;2;32;148;243m10.15674593[0m [38;2;32;148;243m-229.7782156[0m  [38;2;32;148;243m-199.41260428[0m   [38;2;32;148;243m-7.53168592[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.13188797[0m [38;2;32;148;243m-232.58404038[0m [38;2;32;148;243m-218.53631176[0m  [38;2;32;148;243m-21.52838041[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.11906447[0m [38;2;32;148;243m-232.61731533[0m [38;2;32;148;243m-218.53610938[0m  [38;2;32;148;243m-21.50981832[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.08922291[0m [38;2;32;148;243m-232.63614521[0m [38;2;32;148;243m-218.52289796[0m  [38;2;32;148;243m-21.47709538[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.04217127[0m [38;2;32;148;243m-232.64017045[0m [38;2;32;148;243m-218.49665518[0m  [38;2;32;148;243m-21.42751084[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.09543195[0m [38;2;32;148;243m-232.67357259[0m [38;2;32;148;243m-218.56801253[0m  [38;2;32;148;243m-21.49694953[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.13024304[0m [38;2;32;148;243m-232.69460456[0m [38;2;32;148;243m-218.62520443[0m  [38;2;32;148;243m-21.54936568[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.14706559[0m [38;2;32;148;243m-232.70117806[0m [38;2;32;148;243m-218.6696094[0m   [38;2;32;148;243m-21.58309844[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-10.14627466[0m [38;2;32;148;243m-232.69477256[0m [38;2;32;148;243m-218.70164225[0m  [38;2;32;148;243m-21.60146162[0m[1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09520486[0m [38;2;32;148;243m-0.03132634[0m [38;2;32;148;243m-0.0398054[0m  [38;2;32;148;243m-0.07183147[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19577128[0m [38;2;32;148;243m-0.06406345[0m [38;2;32;148;243m-0.09064461[0m [38;2;32;148;243m-0.13485928[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29589559[0m [38;2;32;148;243m-0.09588376[0m [38;2;32;148;243m-0.14365275[0m [38;2;32;148;243m-0.19797758[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39543022[0m [38;2;32;148;243m-0.12647149[0m [38;2;32;148;243m-0.19833442[0m [38;2;32;148;243m-0.2632836[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2949843[0m  [38;2;32;148;243m-0.0940633[0m  [38;2;32;148;243m-0.14400397[0m [38;2;32;148;243m-0.1963145[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19359931[0m [38;2;32;148;243m-0.05939502[0m [38;2;32;148;243m-0.09070054[0m [38;2;32;148;243m-0.13071782[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09192235[0m [38;2;32;148;243m-0.02589531[0m [38;2;32;148;243m-0.0369427[0m  [38;2;32;148;243m-0.06799038[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00980754[0m  [38;2;32;148;243m0.00852633[0m  [38;2;32;148;243m0.01802016[0m [38;2;32;148;243m-0.00593191[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09520486[0m [38;2;32;148;243m-0.03132634[0m [38;2;32;148;243m-0.0398054[0m  [38;2;32;148;243m-0.07183147[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19577128[0m [38;2;32;148;243m-0.06406345[0m [38;2;32;148;243m-0.09064461[0m [38;2;32;148;243m-0.13485928[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29589559[0m [38;2;32;148;243m-0.09588376[0m [38;2;32;148;243m-0.14365275[0m [38;2;32;148;243m-0.19797758[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39543022[0m [38;2;32;148;243m-0.12647149[0m [38;2;32;148;243m-0.19833442[0m [38;2;32;148;243m-0.2632836[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10044592[0m [38;2;32;148;243m0.0324082[0m  [38;2;32;148;243m0.05433045[0m [38;2;32;148;243m0.0669691[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m0.20183091[0m [38;2;32;148;243m0.06707647[0m [38;2;32;148;243m0.10763388[0m [38;2;32;148;243m0.13256578[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.30350787[0m [38;2;32;148;243m0.10057619[0m [38;2;32;148;243m0.16139172[0m [38;2;32;148;243m0.19529321[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.40523776[0m [38;2;32;148;243m0.13499783[0m [38;2;32;148;243m0.21635457[0m [38;2;32;148;243m0.25735168[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09520486[0m [38;2;32;148;243m-0.03132634[0m [38;2;32;148;243m-0.0398054[0m  [38;2;32;148;243m-0.07183147[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19577128[0m [38;2;32;148;243m-0.06406345[0m [38;2;32;148;243m-0.09064461[0m [38;2;32;148;243m-0.13485928[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.29589559[0m [38;2;32;148;243m-0.09588376[0m [38;2;32;148;243m-0.14365275[0m [38;2;32;148;243m-0.19797758[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.39543022[0m [38;2;32;148;243m-0.12647149[0m [38;2;32;148;243m-0.19833442[0m [38;2;32;148;243m-0.2632836[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2949843[0m  [38;2;32;148;243m-0.0940633[0m  [38;2;32;148;243m-0.14400397[0m [38;2;32;148;243m-0.1963145[0m [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.19359931[0m [38;2;32;148;243m-0.05939502[0m [38;2;32;148;243m-0.09070054[0m [38;2;32;148;243m-0.13071782[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.09192235[0m [38;2;32;148;243m-0.02589531[0m [38;2;32;148;243m-0.0369427[0m  [38;2;32;148;243m-0.06799038[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00980754[0m  [38;2;32;148;243m0.00852633[0m  [38;2;32;148;243m0.01802016[0m [38;2;32;148;243m-0.00593191[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00980754[0m  [38;2;32;148;243m0.00852633[0m  [38;2;32;148;243m0.01802016[0m [38;2;32;148;243m-0.00593191[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.249904370901665e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00249081[0m  [38;2;32;148;243m0.00863583[0m  [38;2;32;148;243m0.00730841[0m  [38;2;32;148;243m0.00507512[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00190839[0m  [38;2;32;148;243m0.00011151[0m [38;2;32;148;243m-0.00263163[0m [38;2;32;148;243m-0.00243161[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.05568786[0m  [38;2;32;148;243m0.00325387[0m [38;2;32;148;243m-0.07679241[0m [38;2;32;148;243m-0.07095555[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00386778[0m [38;2;32;148;243m0.00657023[0m [38;2;32;148;243m0.00308145[0m [38;2;32;148;243m0.00856097[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00013255[0m [38;2;32;148;243m0.00022516[0m [38;2;32;148;243m0.0001056[0m  [38;2;32;148;243m0.00029338[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">47</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:39] </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">48</span>                                                                                          
> </pre>
>
>     [38;2;105;105;105m[07/10/23 07:57:39][0m[34m[INFO][0m[2;38;2;144;144;144m[[0m[2;38;2;144;144;144mcommon.py[0m[2;38;2;144;144;144m:[0m[2;38;2;144;144;144m97[0m[2;38;2;144;144;144m][0m - energy: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m   [38;2;32;148;243m7.36263574[0m [38;2;32;148;243m-110.8740819[0m  [38;2;32;148;243m-134.56757381[0m [38;2;32;148;243m-138.70569471[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-14.02149814[0m [38;2;32;148;243m-113.02669004[0m [38;2;32;148;243m-153.51544871[0m [38;2;32;148;243m-152.45722335[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-14.08668762[0m [38;2;32;148;243m-113.05181156[0m [38;2;32;148;243m-153.56075138[0m [38;2;32;148;243m-152.47923199[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-14.13392908[0m [38;2;32;148;243m-113.06433546[0m [38;2;32;148;243m-153.58899227[0m [38;2;32;148;243m-152.48700377[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-14.16258838[0m [38;2;32;148;243m-113.06065644[0m [38;2;32;148;243m-153.60185673[0m [38;2;32;148;243m-152.47998551[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-14.0554813[0m  [38;2;32;148;243m-113.09346307[0m [38;2;32;148;243m-153.50577637[0m [38;2;32;148;243m-152.39727388[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.93091566[0m [38;2;32;148;243m-113.1092577[0m  [38;2;32;148;243m-153.39433411[0m [38;2;32;148;243m-152.30002726[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.7889218[0m  [38;2;32;148;243m-113.11145403[0m [38;2;32;148;243m-153.26921104[0m [38;2;32;148;243m-152.18820749[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.63012791[0m [38;2;32;148;243m-113.10198911[0m [38;2;32;148;243m-153.13019419[0m [38;2;32;148;243m-152.06212176[0m[1m][0m[1m][0m
>     logprob: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m   [38;2;32;148;243m7.36263574[0m [38;2;32;148;243m-110.8740819[0m  [38;2;32;148;243m-134.56757381[0m [38;2;32;148;243m-138.70569471[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.91661817[0m [38;2;32;148;243m-112.99635708[0m [38;2;32;148;243m-153.49891346[0m [38;2;32;148;243m-152.37632023[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.88031232[0m [38;2;32;148;243m-112.98328332[0m [38;2;32;148;243m-153.52751651[0m [38;2;32;148;243m-152.32566146[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.8252355[0m  [38;2;32;148;243m-112.95702826[0m [38;2;32;148;243m-153.54035112[0m [38;2;32;148;243m-152.26239636[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.75068682[0m [38;2;32;148;243m-112.91669689[0m [38;2;32;148;243m-153.53809365[0m [38;2;32;148;243m-152.18323152[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.74708251[0m [38;2;32;148;243m-112.98494292[0m [38;2;32;148;243m-153.45570978[0m [38;2;32;148;243m-152.1746215[0m [1m][0m
>      [1m[[0m [38;2;32;148;243m-13.72595966[0m [38;2;32;148;243m-113.03786467[0m [38;2;32;148;243m-153.35744618[0m [38;2;32;148;243m-152.15229908[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.68718583[0m [38;2;32;148;243m-113.07703105[0m [38;2;32;148;243m-153.24791401[0m [38;2;32;148;243m-152.11639257[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m-13.63130354[0m [38;2;32;148;243m-113.10312522[0m [38;2;32;148;243m-153.12666469[0m [38;2;32;148;243m-152.0668372[0m [1m][0m[1m][0m
>     logdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10487997[0m [38;2;32;148;243m-0.03033296[0m [38;2;32;148;243m-0.01653525[0m [38;2;32;148;243m-0.08090312[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2063753[0m  [38;2;32;148;243m-0.06852823[0m [38;2;32;148;243m-0.03323487[0m [38;2;32;148;243m-0.15357053[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30869358[0m [38;2;32;148;243m-0.1073072[0m  [38;2;32;148;243m-0.04864114[0m [38;2;32;148;243m-0.22460741[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.41190155[0m [38;2;32;148;243m-0.14395956[0m [38;2;32;148;243m-0.06376308[0m [38;2;32;148;243m-0.29675398[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30839879[0m [38;2;32;148;243m-0.10852015[0m [38;2;32;148;243m-0.05006659[0m [38;2;32;148;243m-0.22265237[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20495601[0m [38;2;32;148;243m-0.07139304[0m [38;2;32;148;243m-0.03688793[0m [38;2;32;148;243m-0.14772817[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10173597[0m [38;2;32;148;243m-0.03442298[0m [38;2;32;148;243m-0.02129703[0m [38;2;32;148;243m-0.07181491[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00117563[0m  [38;2;32;148;243m0.00113611[0m [38;2;32;148;243m-0.0035295[0m   [38;2;32;148;243m0.00471544[0m[1m][0m[1m][0m
>     sldf: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10487997[0m [38;2;32;148;243m-0.03033296[0m [38;2;32;148;243m-0.01653525[0m [38;2;32;148;243m-0.08090312[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2063753[0m  [38;2;32;148;243m-0.06852823[0m [38;2;32;148;243m-0.03323487[0m [38;2;32;148;243m-0.15357053[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30869358[0m [38;2;32;148;243m-0.1073072[0m  [38;2;32;148;243m-0.04864114[0m [38;2;32;148;243m-0.22460741[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.41190155[0m [38;2;32;148;243m-0.14395956[0m [38;2;32;148;243m-0.06376308[0m [38;2;32;148;243m-0.29675398[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m[1m][0m
>     sldb: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.         [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m0.10350276[0m [38;2;32;148;243m0.0354394[0m  [38;2;32;148;243m0.01369649[0m [38;2;32;148;243m0.07410161[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.20694555[0m [38;2;32;148;243m0.07256652[0m [38;2;32;148;243m0.02687515[0m [38;2;32;148;243m0.14902581[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.31016558[0m [38;2;32;148;243m0.10953658[0m [38;2;32;148;243m0.04246605[0m [38;2;32;148;243m0.22493907[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m0.41307718[0m [38;2;32;148;243m0.14509566[0m [38;2;32;148;243m0.06023357[0m [38;2;32;148;243m0.30146942[0m[1m][0m[1m][0m
>     sld: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m, [38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[1m[[0m [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.          [38;2;32;148;243m0[0m.        [1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10487997[0m [38;2;32;148;243m-0.03033296[0m [38;2;32;148;243m-0.01653525[0m [38;2;32;148;243m-0.08090312[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.2063753[0m  [38;2;32;148;243m-0.06852823[0m [38;2;32;148;243m-0.03323487[0m [38;2;32;148;243m-0.15357053[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30869358[0m [38;2;32;148;243m-0.1073072[0m  [38;2;32;148;243m-0.04864114[0m [38;2;32;148;243m-0.22460741[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.41190155[0m [38;2;32;148;243m-0.14395956[0m [38;2;32;148;243m-0.06376308[0m [38;2;32;148;243m-0.29675398[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.30839879[0m [38;2;32;148;243m-0.10852015[0m [38;2;32;148;243m-0.05006659[0m [38;2;32;148;243m-0.22265237[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.20495601[0m [38;2;32;148;243m-0.07139304[0m [38;2;32;148;243m-0.03688793[0m [38;2;32;148;243m-0.14772817[0m[1m][0m
>      [1m[[0m[38;2;32;148;243m-0.10173597[0m [38;2;32;148;243m-0.03442298[0m [38;2;32;148;243m-0.02129703[0m [38;2;32;148;243m-0.07181491[0m[1m][0m
>      [1m[[0m [38;2;32;148;243m0.00117563[0m  [38;2;32;148;243m0.00113611[0m [38;2;32;148;243m-0.0035295[0m   [38;2;32;148;243m0.00471544[0m[1m][0m[1m][0m
>     xeps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     veps: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m9[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m [38;2;32;148;243m0.001[0m[1m][0m
>     acc: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     sumlogdet: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00117563[0m  [38;2;32;148;243m0.00113611[0m [38;2;32;148;243m-0.0035295[0m   [38;2;32;148;243m0.00471544[0m[1m][0m
>     beta: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[1m][0m[1m)[0m torch.float64 
>     [38;2;32;148;243m6.0[0m
>     acc_mask: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float32 
>     [1m[[0m[38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m. [38;2;32;148;243m1[0m.[1m][0m
>     loss: [3;38;2;255;0;255mNone[0m [3;38;2;255;0;255mNone[0m 
>     [38;2;32;148;243m-9.236717849885805e-05[0m
>     plaqs: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m-0.00234236[0m  [38;2;32;148;243m0.00869283[0m  [38;2;32;148;243m0.00738453[0m  [38;2;32;148;243m0.00525129[0m[1m][0m
>     sinQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.00163193[0m  [38;2;32;148;243m0.00023077[0m [38;2;32;148;243m-0.00261078[0m [38;2;32;148;243m-0.00241911[0m[1m][0m
>     intQ: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m [38;2;32;148;243m0.04762047[0m  [38;2;32;148;243m0.0067341[0m  [38;2;32;148;243m-0.07618373[0m [38;2;32;148;243m-0.07059076[0m[1m][0m
>     dQint: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m0.00388472[0m [38;2;32;148;243m0.00230856[0m [38;2;32;148;243m0.00034175[0m [38;2;32;148;243m0.00356939[0m[1m][0m
>     dQsin: [1;38;2;255;0;255mtorch.Size[0m[1m([0m[1m[[0m[38;2;32;148;243m4[0m[1m][0m[1m)[0m torch.float64 
>     [1m[[0m[38;2;32;148;243m1.33127299e-04[0m [38;2;32;148;243m7.91132352e-05[0m [38;2;32;148;243m1.17115752e-05[0m [38;2;32;148;243m1.22320953e-04[0m[1m][0m
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>TRAIN STEP: <span style="color: #2094f3; text-decoration-color: #2094f3">49</span>                                                                                          
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:40] </span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">checkSU</span><span style="font-weight: bold">(</span>x_train<span style="font-weight: bold">)</span>: <span style="font-weight: bold">(</span>tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">3.858e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">4.047e-16</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">3</span><span style="color: #2094f3; text-decoration-color: #2094f3">.933e-16</span> <span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">8</span><span style="color: #2094f3; text-decoration-color: #2094f3">.569e-18</span> <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">3.951e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">3.858e-16</span>,
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #2094f3; text-decoration-color: #2094f3">3.877e-16</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">4.047e-16</span><span style="font-weight: bold">]</span>, tensor<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">4</span><span style="font-weight: bold">]</span> f64 x∈<span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">1.270e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.337e-15</span><span style="font-weight: bold">]</span> <span style="color: #7d8697; text-decoration-color: #7d8697">μ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">1</span><span style="color: #2094f3; text-decoration-color: #2094f3">.305e-15</span> <span style="color: #7d8697; text-decoration-color: #7d8697">σ</span>=<span style="color: #2094f3; text-decoration-color: #2094f3">2</span><span style="color: #2094f3; text-decoration-color: #2094f3">.739e-17</span> <span style="font-weight: bold">[</span><span style="color: #2094f3; text-decoration-color: #2094f3">1.303e-15</span>,       
> <span style="color: #696969; text-decoration-color: #696969">           </span><span style="color: #2094f3; text-decoration-color: #2094f3">1.337e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.270e-15</span>, <span style="color: #2094f3; text-decoration-color: #2094f3">1.310e-15</span><span style="font-weight: bold">])</span>                                                                       
> </pre>
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving energy to plots-4dSU3/train                                                                      
> </pre>
>
> ![svg](output_15_76.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving logprob to plots-4dSU3/train                                                                     
> </pre>
>
> ![svg](output_15_78.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving logdet to plots-4dSU3/train                                                                      
> </pre>
>
> ![svg](output_15_80.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sldf to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_82.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:41] </span>Saving sldb to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_84.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sld to plots-4dSU3/train                                                                         
> </pre>
>
> ![svg](output_15_86.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving xeps to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_88.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving veps to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_90.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:42] </span>Saving acc to plots-4dSU3/train                                                                         
> </pre>
>
> ![svg](output_15_92.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sumlogdet to plots-4dSU3/train                                                                   
> </pre>
>
> ![svg](output_15_94.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving beta to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_96.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving acc_mask to plots-4dSU3/train                                                                    
> </pre>
>
> ![svg](output_15_98.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:43] </span>Saving loss to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_100.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving plaqs to plots-4dSU3/train                                                                       
> </pre>
>
> ![svg](output_15_102.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving sinQ to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_104.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">[07:57:44] </span>Saving intQ to plots-4dSU3/train                                                                        
> </pre>
>
> ![svg](output_15_106.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving dQint to plots-4dSU3/train                                                                       
> </pre>
>
> ![svg](output_15_108.svg)
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #696969; text-decoration-color: #696969">           </span>Saving dQsin to plots-4dSU3/train                                                                       
> </pre>
>
> ![svg](output_15_110.svg)
>
> </div>

``` python
```
