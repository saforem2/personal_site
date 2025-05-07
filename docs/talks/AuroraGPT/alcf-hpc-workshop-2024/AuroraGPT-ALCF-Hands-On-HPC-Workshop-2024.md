# AuroraGPT
Sam Foreman
2024-10-30

<link rel="preconnect" href="https://fonts.googleapis.com">

- [🎯 AuroraGPT Goals](#dart-auroragpt-goals)
- [🧪 AuroraGPT: Open Science Foundation
  Models](#test_tube-auroragpt-open-science-foundation-models)
- [📊 AuroraGPT Outcomes](#bar_chart-auroragpt-outcomes)
- [🌌 Aurora](#milky_way-aurora)
- [🤖 ALCF AI Testbed](#robot-alcf-ai-testbed)
- [👥 Team Leads](#busts_in_silhouette-team-leads)
- [🤝 Teams](#handshake-teams)
- [🦜 Model Training](#parrot-model-training)
- [🚀 Accelerating Dataset Processing at Scale for
  Training](#rocket-accelerating-dataset-processing-at-scale-for-training)
- [🚀 Accelerating Dataset Processing:
  Results](#rocket-accelerating-dataset-processing-results)
- [📓 References](#notebook-references)
- [❤️ Thank you!](#heart-thank-you)
- [📑 Bibliography](#bookmark_tabs-bibliography)
- [🎁 Extras](#gift-extras)
  - [🚂 Loooooooooong Sequence
    Lengths](#steam_locomotive-loooooooooong-sequence-lengths)
  - [♻️ Life Cycle of the LLM](#recycle-life-cycle-of-the-llm)
  - [🍎 Training LLMs](#apple-training-llms)

## 🎯 AuroraGPT Goals

<div class="flex-container"
style="flex-direction: column; justify-content: space-around;">

<div class="flex-container"
style="flex-direction: row; justify-content: space-around; align-items:center;">

<div class="column" style="width: 55%">

<div class="blue-card">

**AuroraGPT**: *General purpose scientific LLM*  
Broadly trained on a general corpora plus scientific {papers, texts,
data}

</div>

- **Explore pathways** towards a “Scientific Assistant” model
- **Build with international partners** (RIKEN, BSC, others)
- **Multilingual** English, 日本語, French, German, Spanish
- **Multimodal**: images, tables, equations, proofs, time series,
  graphs, fields, sequences, etc

</div>

<div class="column" style="text-align: center;">

<div id="fig-awesome-llm">

![](./assets/llms.gif)

Figure 1: Image from
[`Hannibal046/Awesome-LLM`](https://github.com/Hannibal046/Awesome-LLM)

</div>

</div>

</div>

<div id="fig-timeline">

<img src="./assets/timelines.png"
style="margin-left:auto;margin-right:auto;;width:75.0%" />

Figure 2: Credit to the entire AuroraGPT team for slides.

</div>

</div>

<div class="notes">

- Here to talk about AuroraGPT, Argonne’s internal effort to build a
  general purpose scientific LLM, broadly trained on a general corpora
  of text + scientific {papers, text, data}

- As part of this effort, we plan to…

  - Explore pathways, build with international partners, multi-{lingual,
    modal}

- Rough timeline of the project and deliverables:

  - 202{3,4}: text-only models, plan to release a series of {7B, 70B,
    1T} models
  - 202{4,5}: Basic multi-modal models
  - 202{5,6}: Advanced scientific multimodal models

</div>

## 🧪 AuroraGPT: Open Science Foundation Models

<div id="fig-aurora-gpt">

![](./assets/AuroraGPT.svg)

Figure 3:
[`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)

</div>

<div class="notes">

- AuroraGPT will be a publicly distributed, open source foundation model
  for open science
- Is being trained on:
  - Scientific / engineering structured data
  - General text, media, news, etc.
  - Large amounts of low to medium quality data
  - Much less high quality data (that is publicly available for use)
- This data is then cleaned, processed, de-duplicated and used for the
  initial pre-training phase of the model
- The vast majority of the overall compute is spent during this initial
  pre-training phase
  - This is the group I help to lead and will be talking a bit about
    today
- The initial pre-training phase is currently underway
  - Eventually, given a bit of time, effort and magic, the model will be
    ready for fine-tuning and additional training for a variety of
    downstream tasks
- The pretrained model will then be handed off for additional
  fine-tuning on a variety of downstream tasks
  - Scientific discovery
  - Accelerate scientific tasks
  - Digital twins
  - Inverse design
  - Code optimization
  - Accelerated simulations
  - Autonomous experiments
  - Co-design
- Becoming increasingly clear that LLMs have the potential to
  drastically accelerate computational science
  - We’ve seen this already for {GenSLMs, Weather / Climate / Earth
    Systems Modeling, Particle Physics, etc.}

</div>

## 📊 AuroraGPT Outcomes

- **Datasets and data pipelines** for preparing science training data
- **Software infrastructure and workflows** to train, evaluate and
  deploy LLMs at scale for scientific resarch purposes
- **Evaluation of state-of-the-art LLM Models** to determine where they
  fall short in deep scientific tasks and where deep data may have an
  impact
- **Assessment of the approach** of augmenting web training data with
  two forms of data specific to science
  - Full text scientific papers
  - Structured scientific datasets (suitably mapped to narrative form)
- **Research grade artifacts** (**models**) for scientific community for
  adaptation for downstream uses
- **Promotion of responsible AI** best practices where we can figure
  them out
- **International Collaborations** around the long term goal of *AGI for
  science*

<div class="notes">

- Deliverables:

  - datasets, pipelines
  - software infrastructure, workflows to interface with science
    applications
  - checkpoints, models, logs, workbook, insights, etc.

- Hope to understand:

  - How different state-of-the-art models perform at different
    scientific tasks
  - where deep data may have an impact
  - feasibility of generically augmenting text with scientific
    structured data

- Huge undertaking that will require large international collaborations
  around long term goal of AGI for science

- Extra points:

  - Well known that LLMs are good for non-consequential tasks
  - Known to “hallucinate” and create false information
  - Can this be mitigated reliably ??

</div>

## 🌌 Aurora

<div class="flex-container" style="align-items: center;">

<div class="column" style="width:5%;">

<div id="tbl-aurora">

Table 1: Aurora Specs

|       |        |
|------:|:-------|
| Racks | 166    |
| Nodes | 10,624 |
|  CPUs | 21,248 |
|  GPUs | 63,744 |
|  NICs | 84,992 |
|   HBM | 8 PB   |
| DDR5c | 10 PB  |

Aurora Specs

</div>

</div>

<div class="column">

<div id="fig-aurora">

![](./assets/aurora.png)

Figure 4: [Aurora Fact
Sheet](https://www.alcf.anl.gov/sites/default/files/2024-07/Aurora_FactSheet_2024.pdf)

</div>

</div>

</div>

## 🤖 ALCF AI Testbed

- ALCF AI Testbed Systems are in production and [available for
  allocations](https://accounts.alcf.anl.gov/#/allocationRequests) to
  the research community
- Significant improvement in time-to-solution and energy-efficiency for
  diverse AI for science applications.
- [NAIRR Pilot](https://nairrpilot.org/)

<div class="red-card" style="color: #FF5252; font-size:90%;">

Up to **25**$\times$ improvement for genomic foundation models with
**6.5**$\times$ energy efficiency

</div>

<div class="flex-container" style="align-items: flex-end;">

<div id="fig-sambanova">

![](./assets/sambanova.jpeg)

Figure 5: **SambaNova SN-30**: 2nd Gen, 8 nodes with 64 AI Accelerators

</div>

<div id="fig-graphcore">

![](./assets/graphcore.png)

Figure 6: **Graphcore Bow**: generation accelerators: Pod-64
configuration with 64 accelerators

</div>

<div id="fig-cerebras">

![](./assets/cerebras.jpeg)

Figure 7: **Cerebras**: 2x CS-2 WSE with Memory-X and Swarm-X
technologies

</div>

<div id="fig-groq">

![](./assets/groq.jpeg)

Figure 8: **GroqRack**: 9 nodes, 8 GroqChip v1.5 Tensor streaming
processors accelerators per node

</div>

</div>

## 👥 Team Leads

<div style="font-size: 100%;">

<div class="flex-container"
style="text-align: center; align-items: center;">

**Planning**

<img src="./assets/team/rick-stevens.png" style="height:1.04167in"
alt="Rick Stevens" />

<img src="./assets/team/ian-foster.png" style="height:1.04167in"
alt="Ian Foster" />

<img src="./assets/team/rinku-gupta.png" style="height:1.04167in"
alt="Rinku Gupta" />

<img src="./assets/team/mike-papka.png" style="height:1.04167in"
alt="Mike Papka" />

<img src="./assets/team/arvind-ramanathan.png" style="height:1.04167in"
alt="Arvind Ramanathan" />

<img src="./assets/team/fangfang-xia.png" style="height:1.04167in"
alt="Fangfang Xia" />

</div>

<div class="flex-container" style="text-align: center;">

<div class="col">

**Data**

<img src="./assets/team/ian-foster.png" style="height:1.04167in"
alt="Ian Foster" />

<img src="./assets/team/robert-underwood.png" style="height:1.04167in"
alt="Robert Underwood" />

</div>

<div class="col">

**Training**

<img src="./assets/team/venkat-vishwanath.png" style="height:1.04167in"
alt="Venkat Vishwanath" />

<img src="./assets/team/sam-foreman.png" style="height:1.04167in"
alt="Sam Foreman" />

</div>

<div class="col">

**Evaluation**

<img src="./assets/team/franck-cappello.png" style="height:1.04167in"
alt="Franck Cappello" />

<img src="./assets/team/sandeep-madireddy.png" style="height:1.04167in"
alt="Sandeep Madireddy" />

<img src="./assets/team/bo-li.png" style="height:1.04167in"
alt="Bo Li" />

</div>

<div class="col">

**Post**

<img src="./assets/team/eliu-huerta.png" style="height:1.04167in"
alt="Eliu Huerta" />

<img src="./assets/team/azton-wells.png" style="height:1.04167in"
alt="Azton Wells" />

</div>

<div class="col">

**Inference**

<img src="./assets/team/rajeev-thakur.png" style="height:1.04167in"
alt="Rajeev Thakur" />

</div>

<div class="col">

**Comms**

<img src="./assets/team/charlie-catlett.png" style="height:1.04167in"
alt="Charlie Catlett" />

<img src="./assets/team/david-martin.png" style="height:1.04167in"
alt="David Martin" />

</div>

<div class="col">

**Distribution**

<img src="./assets/team/brad-ullrich.png" style="height:1.04167in"
alt="Brad Ullrich" />

</div>

</div>

</div>

## 🤝 Teams

<div class="flex-container">

<div class="column">

- **Planning**
- **Data Prep**
  - Accumulate 20+ T tokens of high-quality scientific text and
    structured data
- <span style="background: oklch(from #ff1a8f calc(l * 1.15) c h / 0.1); border: 1px solid #ff1a8f; border-radius: 0.25px;">**Models
  / Training**</span>[^1]
  - Train (entirely from scratch) a series of models on publicly
    available data
- **Evaluation**
  - Skills, trustworthiness, safety, robustness, privacy, machine ethics

</div>

<div class="column">

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

## 🦜 Model Training

<div class="flex-container"
style="text-align: left; width: 100%; justify-content: space-around; line-height: 1em; gap: 5pt;">

<div class="column"
style="background: oklch(from #03BD00 calc(l * 1.15) c h / 0.1); border: 1px solid #03BD00; border-radius: 0.25em; padding: 3pt 8pt;">

✅ <span style="color: #03BD00;">**Goals**</span>

- Want training runs at scale to be:
  - efficient
  - stable
  - reproducible
- This requires:
  - robust data pipelines / file IO
  - effectively overlapping compute with communication
  - stability across {network, filesystem, machine}
- 3D / Multi-dimensional Parallelism strategies
- Large batch training
- Second order optimizers
- Sub-quadratic attention
- State space models
- *Highly optimized GPU kernels*

</div>

<div class="column"
style="background: oklch(from #E90102 calc(l * 1.15) c h / 0.1); border: 1px solid #E90102; border-radius: 0.25em; padding: 3pt 8pt;">

❌ <span style="color: #E90102;">**Challenges**</span>

- *Looong time* to train, can be:
  - weeks (even months) of continuous training
  - order of magnitude longer than typical NN training jobs
- Stability issues:
  - failures are expensive (but inevitable)
  - stragglers common at scale
- Individual jobs are:
  - **fragile**
  - only as good as the worst rank
  - one hang or bad worker can crash job
  - network / filesystem / other-user(s) dependent
- Cost / benefits of different collective communication algorithms
  - depend on optimized / efficient implementations
- Network performance
- *Highly optimized GPU kernels*

</div>

</div>

<div class="aside">

[argonne-lcf /
`Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)

</div>

## 🚀 Accelerating Dataset Processing at Scale for Training

- To train a fixed model on trillions of tokens requires:
  - Aggregating data from multiple different *corpora* (e.g. Reddit,
    StackExchange, GitHub, etc.)
  - Sampling *each training batch* according to a fixed distribution
    across corpora
  - Building indices that map batches of tokens into these files
    (indexing)
- The original implementation was slow, and designed to run on a single
  device
  - Major bottleneck when debugging data pipeline at scale
- [x] Completely re-wrote an asynchronous, distributed implementation
  that *significantly* improves performance

## 🚀 Accelerating Dataset Processing: Results

- [x] Completely re-wrote an asynchronous, distributed implementation
  that *significantly* improves performance (~~60 min~~ $\rightarrow$ 4
  min)

<div class="flex-container">

<div id="fig-data-blendable">

<img src="./assets/blendable.svg" class="r-stretch" />

Figure 9: Time spent building `BlendableDataset`

</div>

<div id="fig-data-gpt">

<img src="./assets/gpt.svg" class="r-stretch" />

Figure 10: Time spent building `GPTDataset`

</div>

</div>

## 📓 References

<div class="flex-container" style="gap: 2pt;">

<div class="column">

- See my other slides at
  [samforeman.me/talks](https://samforeman.me/talks)
  - [LLMs from Scratch](https://saforem2.github.io/llm-workshop-talk)
  - [Creating Small(~ish) LLMs](https://saforem2.github.io/LLM-tutorial)
  - [Parallel Training
    Techniques](https://saforem2.github.io/parallel-training-slides)
  - [LLMs on
    Polaris](https://samforeman.me/talks/llms-on-polaris/#/title-slide)
  - [Training LLMs at Scale](https://samforeman.me/talks/llms-at-scale/)
-  [argonne-lcf /
  `Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)  
  <span class="dim-text">For the largest of large language
  models.</span>
-  [saforem2 / `ezpz`](https://github.com/saforem2/ezpz)  
  <span class="dim-text">Distributed training, ezpz. 🍋</span>

</div>

<div class="column">

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

## ❤️ Thank you!

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

## 📑 Bibliography

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

### 🚂 Loooooooooong Sequence Lengths

<div class="flex-container"
style="align-items: center; justify-content: center;">

<img src="../../../assets/anl.svg" style="height:50pt;" />

<span class="dim-text" style="font-size: 2.0em;"></span>

<img src="../../../assets/deepspeed-logo-transparent.svg"
style="height:50pt;" />

</div>

- Working with [
  Microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) team to
  enable longer sequence lengths (context windows) for LLMs
  - See my [blog
    post](https://samforeman.me/posts/auroragpt/long-sequences/) for
    additional details

<div id="fig-long-seq">

<div class="flex-container">

![25B](https://raw.githubusercontent.com/saforem2/scaling4science/main/assets/25B.svg)

![33B](https://raw.githubusercontent.com/saforem2/scaling4science/main/assets/33B.svg)

</div>

Figure 11: Maximum (achievable) `SEQ_LEN` for both `25B` and `33B`
models (See: Song et al. (2023))

</div>

<div class="aside">

[ `scaling4science`](https://github.com/saforem2/scaling4science)  
[
`Megatron-DS-Benchmarking`](https://github.com/saforem2/Megatron-DS-Benchmarking)

</div>

### ♻️ Life Cycle of the LLM

<div class="panel-tabset" style="text-align:center">

### 📝 Pre-training

<div id="fig-pretraining">

![](https://jalammar.github.io/images/gpt3/03-gpt3-training-step-back-prop.gif)

Figure 12: **Pre-training**: Virtually all of the compute used during
pretraining phase

</div>

### 🎀 Fine-Tuning

<div id="fig-fine-tuning">

![](https://jalammar.github.io/images/gpt3/10-gpt3-fine-tuning.gif)

Figure 13: **Fine-tuning**: Fine-tuning actually updates the model’s
weights to make the model better at a certain task.

</div>

</div>

### 🍎 Training LLMs

<div class="flex-container" style="align-items: flex-end;">

<div class="column" style="width:33%;">

<div id="fig-it-hungers">

![](https://github.com/saforem2/llm-lunch-talk/blob/main/docs/assets/it_hungers.jpeg?raw=true)

Figure 14: It’s hungry!

</div>

</div>

<div class="column" style="width:60%;">

<div id="fig-evolution">

![](https://github.com/Mooler0410/LLMsPracticalGuide/raw/main/imgs/survey-gif-test.gif)

Figure 15: Visualization from Yang et al. (2023)

</div>

</div>

</div>

[^1]: Co-led by: Venkat Vishwanath, Sam Foreman
