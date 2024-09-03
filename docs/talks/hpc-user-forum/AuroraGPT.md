# AuroraGPT
[Sam Foreman](https://samforeman.me)
[<span class="orcid-green"></span>](https://orcid.org/0000-0002-9981-0876)
2024-09-04

## 📄 Overview

<div class="flex-container">

<div class="col1">

<div class="blue-card" style="margin-bottom: 0.5em;">

**AuroraGPT**: *General purpose scientific foundation model*

</div>

- Model sizes:
  - {`7B`, `70B`, `1T`}
- 
  [`Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)
  - ✅ {Intel, NVIDIA, AMD}

</div>

<div class="col2" style="width:75%;">

[![](https://github.com/Hannibal046/Awesome-LLM/raw/main/resources/image8.gif)](https://github.com/Hannibal046/Awesome-LLM/raw/main/resources/)

</div>

</div>

- Trained on:

  <div class="flex-container"
  style="flex-direction:row; justify-content: space-around;">

  <div class="flex-container" style="flex-direction:column;">

  🇺🇸English  
  🇯🇵日本語[^1]  
  🇫🇷French  
  🇩🇪Deutsch  
  🇪🇸Español[^2]  
  🇮🇹Italian

  </div>

  <div class="flex-container" style="flex-direction:column;">

  🧪 scientific text  
  🖼️ images  
  📊 tables  
  ➕ equations  
  📖 proofs

  </div>

  <div class="flex-container" style="flex-direction:column;">

  📆 structured data  
  ⛓️ sequences  
  ⏰ time-series  
  🕸️ graphs  
  🌀 fields

  </div>

  </div>

### 🎯 Project Goals

<div id="fig-project-goals">

![](./assets/AuroraGPT.svg)


Figure 1: Overview of AuroraGPT Project

</div>

### 👥 Teams

<div class="flex-container">

<div class="col1">

- **Planning**
- **Data Prep**
  - Accumulate 20+ T tokens of high-quality scientific text and
    structured data
- <span style="background: oklch(from #ff1a8f calc(l * 1.15) c h / 0.1); border: 1px solid #ff1a8f; border-radius: 0.25px;">**Models
  / Training**</span>[^3]
  - Train (entirely from scratch) a series of models on publicly
    available data
- **Evaluation**
  - Skills, trustworthiness, safety, robustness, privacy, machine ethics

</div>

<div class="col2">

- **Post-Training**
  - Fine-tuning, alignment
- **Inference**
  - Model serving, API development / public-facing web services
- **Distribution**
  - Licensing, generating and distributing artifacts for public
    consumption
- **Communication**

</div>

</div>

### 🤖 Aurora[^4]

<div class="flex-container">

<div class="col1" style="width:30%;">

- 166 Racks
- 10,624 Nodes
- 21,248 CPUs
- 63,744 GPUs
- 84,992 NICs
- 8 PB HBM
- 10 PB DDR5c

</div>

<div class="col2" style="width:70%;">

![](./assets/aurora.png)

</div>

</div>

## 🦙 LLMs

### 🦜 Model Training

<div class="flex-container"
style="text-align: left; width: 100%; justify-content: center; line-height: 1em;">

<div class="col1" width="49%"
style="background: oklch(from #03BD00 calc(l * 1.15) c h / 0.1); border: 1px solid #03BD00; border-radius: 0.25em; padding: 3pt 8pt; margin-right: 1%">

✅ <span style="color: #03BD00;">**Goals**</span>

- Want training runs *at scale* to be:
  - Efficient
  - Stable
  - Reproducible
- This requires:
  - Robust data pipelines / file IO
  - Effectively overlapping compute with communication
  - Stability across {network, filesystem, machine}
- For larger models:
  - Multi-dimensional parallelism strategies

</div>

<div class="col2" width="49%"
style="background: oklch(from #E90102 calc(l * 1.15) c h / 0.1); border: 1px solid #E90102; border-radius: 0.25em; padding: 3pt 8pt; margin-left: 1%;">

❌ <span style="color: #E90102;">**Difficulties**</span>

- *Looong time* to train

- Stability issues

  - Failures are **expensive** <span class="dim-text">(and
    unavoidable)</span>
  - stragglers common at scale

- Individual jobs are:

  - **fragile**
  - only as good as the worst rank
  - one hang or bad worker can crash job
  - network / filesystem / other-user(s) dependent

</div>

</div>

### 📊 Data Pipeline

<div id="fig-data-pipeline">

<div class="flex-container">

![Time spent building `BlendableDataset`](./assets/blendable.svg)

![Time spent building `GPTDataset`](./assets/gpt.svg)

</div>

Figure 2: Complete re-write <span class="dim-text">(parallel,
async)</span> of original data input pipeline gives *significant*
improvements

</div>

### 🚀 Training at Scale

<div class="flex-container">

<div class="col1">

- 3D Parallelism
- Highly optimized GPU kernels
- Network performance
- Cost / benefits of different collective communication algorithms
  - depend on optimized / efficient implementations

</div>

<div class="col2">

- Large batch training
- Second order optimizers
- State space models
- Sub-quadratic attention (?)

</div>

</div>

### ♻️ Life Cycle of the LLM

<div class="panel-tabset" style="text-align:center">

#### 📝 Pre-training

<div id="fig-pretraining">

![](https://jalammar.github.io/images/gpt3/03-gpt3-training-step-back-prop.gif)


Figure 3: **Pre-training**: Virtually all of the compute used during
pretraining phase

</div>

#### 🎀 Fine-Tuning

<div id="fig-fine-tuning">

![](https://jalammar.github.io/images/gpt3/10-gpt3-fine-tuning.gif)


Figure 4: **Fine-tuning**: Fine-tuning actually updates the model’s
weights to make the model better at a certain task.

</div>

</div>

## 🔗 Links

<div class="flex-container">

<div class="col1">

- 🏡 [samforeman.me](https://samforeman.me):
  - 🦜 [Talks](https://samforeman.me/talks/):
    - [HPC User Forum](https://samforeman.me/talks/hpc-user-forum/)
      \[[slides](https://samforeman.me/talks/hpc-user-forum/slides.html)\]
- See my other slides on:
  - [LLMs from Scratch](https://saforem2.github.io/llm-workshop-talk)
  - [Creating Small(~ish) LLMs](https://saforem2.github.io/LLM-tutorial)
  - [Parallel Training
    Techniques](https://saforem2.github.io/parallel-training-slides) for
    additional details
  - [LLMs on
    Polaris](https://samforeman.me/talks/llms-on-polaris/#/title-slide)
  - [Training LLMs at Scale](https://samforeman.me/talks/llms-at-scale/)

</div>

<!-- ## 📓 References {background-color="white"} -->

<div class="col2">

- [🏎️
  `Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)  
  <span class="dim-text">For the largest of large language
  models.</span>
- [🍋 `saforem2/ezpz`](https://github.com/saforem2/ezpz)  
  <span class="dim-text">Distributed training, ezpz.</span>
- 👀 See also:
  - [New international consortium for generative AI models for
    science](https://www.anl.gov/article/new-international-consortium-formed-to-create-trustworthy-and-reliable-generative-ai-models-for)
  - [PyTorch Distributed
    Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
  - [Distributed Data Parallel — PyTorch master
    documentation](https://pytorch.org/docs/master/notes/ddp.html)
  - [🤗 Efficient Training on Multiple
    GPUs](https://huggingface.co/docs/transformers/en/perf_train_gpu_many)
  - [Getting Started -
    DeepSpeed](https://www.deepspeed.ai/getting-started/)

</div>

</div>

### ❤️ Thank you!

- Organizers

- Feel free to reach out!

  <split even>

  [<i class="fas fa-home"></i>](https://samforeman.me)
  [<i class="far fa-paper-plane"></i>](mailto:///foremans@anl.gov)
  [<i class="fab fa-twitter"></i>](https://www.twitter.com/saforem2)

  </split>

> [!NOTE]
>
> ### 🙏 Acknowledgements
>
> This research used resources of the Argonne Leadership Computing
> Facility, which is a DOE Office of Science User Facility supported
> under Contract DE-AC02-06CH11357.

### 📗 Bibliography

- Refs:
  - Wei et al. (2022)
  - Animations from [The Illustrated
    Transformer](http://jalammar.github.io/illustrated-transformer/)

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-song2023ds4sci" class="csl-entry">

Song, Shuaiwen Leon, Bonnie Kruft, Minjia Zhang, Conglong Li, Shiyang
Chen, Chengming Zhang, Masahiro Tanaka, et al. 2023. “DeepSpeed4Science
Initiative: Enabling Large-Scale Scientific Discovery Through
Sophisticated AI System Technologies.”
<https://arxiv.org/abs/2310.04610>.

</div>

<div id="ref-wei2022emergentabilitieslargelanguage" class="csl-entry">

Wei, Jason, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph,
Sebastian Borgeaud, Dani Yogatama, et al. 2022. “Emergent Abilities of
Large Language Models.” <https://arxiv.org/abs/2206.07682>.

</div>

<div id="ref-yang2023harnessing" class="csl-entry">

Yang, Jingfeng, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng,
Haoming Jiang, Bing Yin, and Xia Hu. 2023. “Harnessing the Power of LLMs
in Practice: A Survey on ChatGPT and Beyond.”
<https://arxiv.org/abs/2304.13712>.

</div>

</div>

## 🎁 Extras

### 🔍 Details

- Llama Style Architecture:
  - [
    `argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)
- Performant training implementations of AuroraGPT architecture on
  - {Aurora, Polaris, Cerebras, SambaNova}
- Workflow to capture:
  - loss curves
  - snapshots, checkpoints
  - scaling / performance data
- Training runs for AuroraGPT-7B <span class="dim-text">(ongoing)</span>
  - Baseline (Dolma) @ {Aurora, Polaris} <span class="dim-text">(twins
    for debugging)</span>
  - Baseline + Science @ Aurora
- Trained raw models 📮 delivered to post-pretraining team
  - AuroraGPT-7B-A, AuroraGPT-7B-P, AuroraGPT-7B-S  
    (A=`Aurora`, P=`Polaris`, S=`Science`)

### 🤔 Why?

- **For Science!**
- Data-{sets, pipelines} for {preparing, aggregating, parsing,
  analyzing} scientific data
- Infrastructure to {train, eval, deploy} LLMs for science
  - Comparative analysis across: {models, tasks, languages, contexts, …}
- Augment text data from the web with:
  - full text papers
  - structured scientific data[^5]
- Safety-driven, publicly-visible, open-source approach:
  - Distribution of research grade artifacts (models, checkpoints, etc.)
  - International collaborations on AGI for science

### 🍎 Training LLMs

<div class="flex-container" style="align-items: flex-end;">

<div class="col1" style="width:33%;">

<div id="fig-it-hungers">

![](https://github.com/saforem2/llm-lunch-talk/blob/main/docs/assets/it_hungers.jpeg?raw=true)


Figure 5: It’s hungry!

</div>

</div>

<div class="col2" style="width:60%;">

<div id="fig-evolution">

![](https://github.com/Mooler0410/LLMsPracticalGuide/raw/main/imgs/survey-gif-test.gif)


Figure 6: Visualization from Yang et al. (2023)

</div>

</div>

</div>

### 🚂 Loooooooooong Sequence Lengths

<div class="flex-container"
style="text-align: center; align-items: center;">

<img src="../../assets/anl.svg" style="width:48%;" />

<span class="dim-text"
style="font-size: 2.0em; padding-left: 15pt;"></span>

<img src="../../assets/deepspeed-logo-transparent.svg"
style="width: 60%" />

</div>

<!-- - Working with [ Microsoft -->
<!-- DeepSpeed](https://github.com/microsoft/DeepSpeed) team to enable longer -->
<!-- sequence lengths (context windows) for LLMs -->
<div id="fig-long-seq">

<div class="flex-container">

![25B](https://raw.githubusercontent.com/saforem2/scaling4science/main/assets/25B.svg)

![33B](https://raw.githubusercontent.com/saforem2/scaling4science/main/assets/33B.svg)

</div>

Figure 7: Maximum (achievable) `SEQ_LEN` for both `25B` and `33B` models
(See: Song et al. (2023))

</div>

<div class="aside">

[ `scaling4science`](https://github.com/saforem2/scaling4science)  
[
`Megatron-DS-Benchmarking`](https://github.com/saforem2/Megatron-DS-Benchmarking)

</div>

### 💾 Evaluating Checkpoints

``` python
from typing import Optional
import os
from pathlib import Path

from transformers import LlamaForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7B-hf")

def load_model(ckpt_dir) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(ckpt_dir)

def eval_model(model, max_length: int, prompt: str) -> str:
    return (
        tokenizer.batch_decode(
            model.generate(
                **tokenizer(prompt, return_tensors="pt"),
                 max_length=max_length,
            ),
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )[0]
    )

def load_and_eval_model_from_checkpoint(
        step: int,
        max_length: int = 64,
        prompt: Optional[str] = None,
        ckpt_root: Optional[os.PathLike | Path | str] = None,
) -> str:
    print(f"Loading model from checkpoint at global step: {step}")
    prompt = "What is it like in there?" if prompt is None else prompt
    ckpt_root = Path("checkpoints") if ckpt_root is None else Path(ckpt_root)
    ckpt_dir = ckpt_root.joinpath(f"global_step{step}")
    return (
        eval_model(
            model=load_model(ckpt_dir.as_posix())
            max_length=max_length,
            prompt=prompt,
        )
    )
```

### 🚀 Model Evaluations

<div class="panel-tabset">

#### 7000

Tokens: 88B

``` python
>>> print(load_checkpoint(7000))
Loading model from checkpoint at global step: 7000
"What is it like in there?"
"""
I'm not sure if it's a good idea to use a different name for the same thing,
but I'm sure it's a good idea to use a different name for the same thing.
I'm not sure if it's a good idea to use a different name for the same thing,
but I'm sure it's a good idea to use a different name for the same thing.
I'm not sure if it's a good idea to use a different name for the same thing,
but I'm sure it
"""
```

#### 12000

Tokens: 150B

``` python
>>> print(load_checkpoint(12000))
Loading model from checkpoint at global step: 12000
"What is it like in there?"
"""
What is it like in there?
The people are very friendly and helpful.
What is it like in there?
The people are very friendly and helpful.
What is it like in there?
The people are very friendly and helpful.
What is it like in there?
The people are very friendly and helpful.
What is it like in there?
The people are very friendly and helpful.
What is it like in there?
"""
```

#### 17000

Tokens: 215B

``` python
>>> print(load_checkpoint(17000))
Loading model from checkpoint at global step: 17000
"What is it like in there?"
"""
I’m not sure what to expect. I’m not sure what to expect from the people I’m
with. I’m not sure what to expect from the people I’m with. I’m not sure what
to expect from the people I’m with. I’m not sure what to expect from the people
I’m with.
I’m not sure what to expect from the people I’m with.
I’m not sure what to expect from the people I’m with.
I’m not sure what to expect from the people
"""
```

#### 22000

Tokens: 277B

``` python
>>> print(load_checkpoint(22000))
Loading model from checkpoint at global step: 22000
"What is it like in there?"
"""
I’m a 20 year old guy from the UK. I’m a student at the University of
Manchester, studying Computer Science. I’m a big fan of the band, The Beatles,
and I’m a huge fan of the movie, The Wizard of Oz. I’m a huge fan of the band,
The Beatles, and I’m a huge fan of the movie, The Wizard of Oz.
I’m a big fan of the band, The Beatles, and I’m a huge fan of the movie
"""
```

#### 32000

Tokens: 400B

``` python
>>> print(load_checkpoint(32000))
Loading model from checkpoint at global step: 32000
"What is it like in there?"
"""
I've been to the US and I've been to Canada.
In the US, it's a lot like the US.
In Canada, it's a lot like the US.
In the US, it's a lot like the US.
In Canada, it's a lot like the US.
In the US, it's a lot like the US.
In Canada, it's a lot like the US.
In the US, it's
"""
```

#### 40000

Tokens: 503B

``` python
>>> print(load_checkpoint(40000))
Loading model from checkpoint at global step: 40000
"What is it like in there?"
"""
The first thing you notice when you enter the room is the size. It’s huge. It’s
like a football field. It’s a lot of space.
The second thing you notice is the light. It’s bright. It’s bright.
The third thing you notice is the sound. It’s loud. It’s loud.
The fourth thing you notice is the smell. It’s a lot of smells. It’s a lot of smells.
The fifth thing you notice is the temperature. It’s hot.
"""
```

</div>

[^1]:

    [Argonne and RIKEN sign a MOU in support of AI for
    science](https://www.anl.gov/article/argonne-and-riken-sign-a-memorandum-of-understanding-in-support-of-ai-for-science)

[^2]:

    Collaborations with Barcelona Supercomputing Center

[^3]: Co-led by: Venkat Vishwanath, Sam Foreman

[^4]:

    [The Computer That Will Change Everything – Chicago
    Magazine](https://www.chicagomag.com/chicago-magazine/february-2023/the-computer-that-will-change-everything/)

[^5]: Can be much more difficult than text (or even image) data
