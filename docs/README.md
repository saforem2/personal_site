

<div style="font-size:1.0em; text-align: center;">

<span class="profile-avatar"
style="width: 100%; border: 0px solid var(--bg-border)!important;">![][1]</span>

<span style="font-size: 1.5rem; color: var(--dim-text)!important; padding-bottom: 0pt; font-family: 'IBM Plex Sans Condensed'; font-weight: 400;"><span class="dim-text">👋
Hi, I’m Sam!</span> [<span class="orcid-green"
style="background: none!important;"></span>]</span>

<div style="display: flex; flex-direction: row; align-items: center; text-align: center!important; justify-content: center; background-color: var(--link-bg-color);">

[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][2]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][3]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][4]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][5]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][6]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][7]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][8]
[<span class="icon dim-text"
style="font-size: 1.5rem; padding-right:0pt;"></span>][9]

</div>

</div>

<br>

<div class="panel-tabset" style="justify-content: center;">

### 🧑🏻‍💻 About

<div class="flex-container"
style="width: 100%; justify-content: space-between; align-items: flex-start;">

<div class="flex-container" style="width:50%;">

- 💻 [Computational scientist][ALCF]  
  @ Argonne National Laboratory [(ALCF)]

- 🧪 Interested in:

  - {AI, HPC} for science

  - 🚀 scaling large models across thousands of GPUs

</div>

<div class="flex-container"
style="flex-direction: column; justify-content: flex-start; width: 45%">

> [!TIP]
>
> ### 🎤   <span class="dim-text" style="font-size:1.0em!important;">Recent Talks</span>
>
> <span class="dim-text" style="font-size:1em;">📊 [here] ( + how I
> [make them]! )</span>

> [!TIP]
>
> ### <span style="color:#1CD760;"><img src="./assets/spotify-green.svg" class="inline-icon"
> style="height:1.33rem; vertical-align: text-bottom;" /> Now Playing</span>
>
> [![][10]][11]
>
> > [!TIP]
> >
> > ### <span style="color:#D41109;">[![][12]][13]</span>
> >
> > <script>
> > /**
> >   Developed by Prashant Shrestha
> >   + https://prashant.me
> > */
> > var lastfmData = {
> >   baseURL:
> >     "https://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks&user=",
> >   // Your Last.fm Username
> >   user: "saforem2",
> >   // Your API key
> >   api_key: "1dbc15037c1fe71ce06acbb3f73adc75",
> >   additional: "&format=json&limit=1"
> > };
> > &#10;var getSetLastFM = function() {
> >   $.ajax({
> >     type: "GET",
> >     url:
> >       lastfmData.baseURL +
> >       lastfmData.user +
> >       "&api_key=" +
> >       lastfmData.api_key +
> >       lastfmData.additional,
> >     dataType: "json",
> >     success: function(resp) {
> >       var recentTrack = resp.recenttracks.track[0];
> >       var formatted =
> >         // "<img src='https://api.iconify.design/streamline-emojis:musical-notes.svg?color=%23888888'>" + recentTrack.name;
> >         "🎶 " + recentTrack.name;
> >       $("a#tracktitle")
> >         .html(formatted)
> >         .attr("href", recentTrack.url)
> >         .attr("title", recentTrack.name + " by " + recentTrack.artist["#text"])
> >         .attr("target", "_blank");
> > &#10;      var artistFormatted =
> >         // "<img src='https://api.iconify.design/material-symbols:person.svg?color=%23888888'>" + recentTrack.artist["#text"];
> >         "🗣️ " + recentTrack.artist["#text"];
> >       $("a#trackartist")
> >         .html(artistFormatted)
> >         .attr("title", "Artist : " + recentTrack.artist["#text"]);
> >       $("img#trackart").attr("src", recentTrack.image[2]["#text"]);
> >     },
> >     error: function(resp) {
> >       $("a#tracktitle").html(
> >         "<img src='https://api.iconify.design/streamline-emojis:muted-speaker.svg?color=%23888888'>" + "Silence!"
> >       );
> >       $("img#trackart").attr("src", "🧑🏻‍💻");
> >       var artistFormatted =
> >         "Sam Foreman";
> >       $("a#trackartist")
> >         .html(artistFormatted)
> >         .attr("href", "https://samforeman.me");
> >     }
> >   });
> > };
> > &#10;// Get the new one.
> > getSetLastFM();
> > // Start the countdown.
> > setInterval(getSetLastFM, 10 * 5000);
> > </script> <div class="nowplayingcard">
> > <div class="nowplayingcontainer-inner">
> > <img id="trackart" src="#">
> > <div class="trackInfo">
> > <a id="tracktitle"></a>
> > <a href="#" id="trackartist"></a>
> > </div>
> > </div>
> > </div>

</div>

</div>

- <details closed>

  <summary>

  👀 <strong>If you’re curious</strong>
  </summary>

  - <details closed>

    <summary>

    🔥 What I work on

    </summary>

    As a member of the [AI / ML Group] at [ALCF][(ALCF)], I work on:

    <div class="flex-container">

    <div class="flex-container">

    - 🤖 🧪 [AI + Science][2]

    - 🎲 [Building better sampling methods for Lattice QCD]

    - 🧬 [Genome-Scale Language Models]

      - [ GenSLM]

      - 🥇 [ACM Gordon Bell Special Prize]

    </div>

    <div class="flex-container">

    - 🌍 [Foundation models for long term climate forecasting]

    - 🏃‍♂️ [Scaling Large Language Models]

    - 🏎️ [Distributed training across thousands of GPUs]

    </div>

    </div>

    </details>

  - <details closed>

    <summary>

    📍 How I got here

    </summary>

    My [current research] focuses on using deep generative modeling to
    help build better sampling algorithms in lattice gauge theory. In
    particular, I’m interested in building gauge equivariant neural
    network architectures and using inductive priors to incorporate
    physical symmetries into machine learning models.

    I received my PhD in Physics from the University of Iowa in 2019 and
    my thesis was on [Learning Better Physics: A Machine Learning
    Approach to Lattice Gauge Theory].

    Prior to this, I completed two bachelors degrees (Engineering
    Physics and Applied Mathematics, 2015) at The University of Illinois
    at Urbana-Champaign. My undergraduate dissertation was titled
    [Energy Storage in Quantum Resonators] and was supervised by
    Professor [Alfred Hübler] within the Center for Complex Systems
    Research at UIUC.

    This work ultimately resulted in a [patent] !!

    </details>

</details>

> [!TIP]
>
> ### <span class="dim-text">❤️‍🩹 Status</span>
>
> <span class="highlight">yellow</span>
> <span class="highlight-pink">pink</span>
> <span class="highlight-green">green</span>
> <span class="highlight-blue">blue</span>
> <span class="circle-sketch-highlight">circle</span>
>
> ``` python
> import datetime
> from rich import print
> now = datetime.datetime.now()
> day = now.strftime("%Y-%m-%d")
> time = now.strftime("%H:%M:%S")
> print(' '.join([
>     "[#838383]Last Updated[/]:",
>     f"[#E599F7]{day}[/]",
>     "[#838383]@[/]",
>     f"[#00CCFF]{time}[/]"
> ]))
> ```
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #838383; text-decoration-color: #838383">Last Updated</span>: <span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">2024</span><span style="color: #e599f7; text-decoration-color: #e599f7">-</span><span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">10</span><span style="color: #e599f7; text-decoration-color: #e599f7">-</span><span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">19</span> <span style="color: #838383; text-decoration-color: #838383">@</span> <span style="color: #00ccff; text-decoration-color: #00ccff; font-weight: bold">12:34:18</span>
> </pre>
>
> <div style="text-align:center;">
>
> <span class="dim-text">© Copyright [Sam
> Foreman][<span class="icon dim-text" style="font-size: 1.5rem; padding-right:0pt;"></span>]</span>
>
> <img src='https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fsamforeman.me&count_bg=%23838383&title_bg=%23303030&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false'>
>
> </div>

### 📝 Work

\[**NOTE**\]: <span class="dim-text">*You can find a full list of my
publications on my [Google Scholar][5]*</span>.

- [Intro to HPC Bootcamp: Engaging New Communities Through Energy
  Justice Projects]  
  <span class="dim-text">Journal of Computational Science, *2024*</span>

- [Thorough Characterization and Analysis of Large Transformer Model
  Training At-Scale]  
  <span class="dim-text">Proc. ACM Meas. Anal. Comput. Syst.
  *03/2024*</span>

- [MLMC: Machine Learning Monte Carlo for Lattice Gauge Theory]  
  <span class="dim-text">**S. Foreman** et al. [Lattice, 2023
  (Proceedings)], *12/2023*</span>

- [Protein Generation via Genome-scale Language Models with Bio-physical
  Scoring]  
  <span class="dim-text">@ SC’23, *11/2023*</span>

-  [**DeepSpeed4Science Initiative**: Enabling Large-Scale Scientific
  Discovery \[…\]][14]  
  <span class="dim-text">@ [NeurIPS 2023 AI For Science Workshop],
  *10/2023*</span>

  - [ DeepSpeed4Science.ai Blog Post]

  - [ Loooooooong Sequence Lengths]

- [Comprehensive Performance Study of LLMs on Novel AI Accelerators]  
  <span class="dim-text">M. Emani, **S. Foreman**, et al., [IPDPS 2024],
  *10/2023*</span>

- [Exploratory Analysis of Climate Data with
  `ClimRR`][Foundation models for long term climate forecasting]  
  <span class="dim-text">**S. Foreman**, [Intro to HPC Bootcamp @
  NERSC], *08/2023*</span>

- 🏆 [**GenSLMs: Genome-scale language models reveal SARS-Cov-2
  evolutionary dynamics**]  
  <span class="dim-text">@ SC’22 *10/2022*</span>

  - 🥇 [ACM Gordon Bell Special Prize]

- [Lattice QCD and Particle Physics]  
  <span class="dim-text">A.S. Kronfeld et al., *07/2022*</span>

- [Applications of ML to Lattice QFT]  
  <span class="dim-text">D. Boyda, S. Calí, **S. Foreman**, et al.,
  \[[*arXiv:2202.05838*][Applications of ML to Lattice QFT]\],
  *02/2022*</span>

- [LeapFrogLayers: Trainable Framework for Effective Sampling]  
  <span class="dim-text">**S. Foreman**, X.Y. Jin, J.C. Osborn,
  [Lattice, *2021*]</span>

- [HMC with Normalizing Flows][] \[[slides]\]  
  <span class="dim-text">**S. Foreman** et al., [Lattice,
  *2021*][15]</span>

- [Deep Learning Hamiltonian Monte Carlo][] \[[+ poster]\]  
  <span class="dim-text">**S. Foreman**, X.Y. Jin, & J.C. Osborn, @
  [SimDL Workshop @ ICLR], *2021*</span>

- [Machine Learning and Neural Networks for Field Theory]  
  <span class="dim-text">**S. Foreman**, X.Y. Jin, & J.C. Osborn,
  [SnowMass], *2020*</span>

- [Examples of renormalization group transformations for image sets]  
  <span class="dim-text">**S. Foreman** et al., Physical Review E.,
  *2018*</span>

- [RG inspired Machine Learning for lattice field theory]  
  <span class="dim-text">**S. Foreman** et al., [*arXiv:1710.02079*],
  *2017*</span>

- [Large Energy Density in Three-Plate Nanocapacitors due to Coulomb
  Blockade]  
  <span class="dim-text">**S. Foreman** et al., *J. Appl. Phys*,
  *2018*</span>

### 🦜 Talks

<div id="listing-talks">

</div>

#### 📆 2024

> [!TIP]
>
> ### <span class="dim-text">[**AuroraGPT**] @ [*HPC User Forum*, 2024][] \[09/2024\]</span>
>
> <div class="reveal-full-page">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://samforeman.me/talks/hpc-user-forum/slides" title="AuroraGPT" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="aspect-ratio: 1.5;">
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Training LLMs at Scale**] @ [*ATPESC*, 2024][] \[08/2024\]</span>
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://samforeman.me/talks/llms-at-scale/slides" title="Training LLMs at Scale" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>

> [!TIP]
>
> ### <span class="dim-text">[**LLMs on Polaris**] @ [*Center for Scientific Foundation Models*, Summer School 24’][] \[07/2024\]</span>
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://samforeman.me/talks/llms-on-polaris/slides" title="LLMs on Polaris" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>

> [!TIP]
>
> ### <span class="dim-text">[**Parallel Training Techniques**] @ [*AI-4-Science Training Series*][] \[03/2024\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/parallel-training-slides" title="Parallel Training Techniques" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**LLMs from Scratch**] @ [LLM Tutorial Workshop][] \[02/2024\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/llm-workshop-talk" title="LLMs from Scratch" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

#### 📆 2023

> [!TIP]
>
> ### <span class="dim-text">[**Creating Small(-ish) LLMs**] @ [LLM Tutorial Workshop (1)][] \[11/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/LLM-tutorial" title="Creating Small(-ish) LLMs" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Exascale Science on Aurora**] @ [Intel oneAPI Workshop @ UIC][] \[10/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/oneapi-talk" title="Exascale Science on Aurora" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**LLM Lunch Talk**] @ [ALCF Hands On HPC Workshop][Intel oneAPI Workshop @ UIC] \[10/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/llm-lunch-talk/#/section" title="LLMs on Polaris" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Scaling LLMs for Science**] @ [Data-Intensive Computing + AI/ML at Scale][] \[08/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/scaling4science/#/section" title="Scaling LLMs for Science and Ongoing Collaborations" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**MLMC: Machine Learning Monte Carlo**] @ [Lattice 2023][] \[07/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/lattice23/#/title-slide" title="MLMC: Machine Learning Monte Carlo" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Generative Modeling and Efficient Sampling**] @ [PASC23][] \[07/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/lqcd-pasc23/" title="Generative Modeling and Efficient Sampling" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Efficient Sampling for LGT**] @ [Deep Fridays @ U. Bologna][] \[04/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/deep-fridays/" title="Efficient Sampling for LGT" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

#### 📆 2022

> [!TIP]
>
> ### <span class="dim-text">[**Large Scale Training**] @ [AI4Science on Supercomputers (ALCF)][] \[11/2022\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/ai4sci-large-scale-training/#" title="Large Scale Training" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Hyperparameter Management**] @ [ALCF SDL Workshop][] \[10/2022\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/hparam-management-sdl2022" title="Hyperparameter Management" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Statistical Learning**] @ [ATPESC 2022][] \[08/2022\]</span>
>
> - [📕 accompanying notebook]
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/ATPESC-StatisticalLearning/#/" title="Statistical Learning" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Scientific Data Science: An Emerging Symbiosis**] @ ANL (05/2022)</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/anl-job-talk" title="Scientific Data Science" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Machine Learning in HEP**] @ UNC Greensboro \[03/2022\]</span>
>
> - [**Machine Learning in HEP**], at UNC Greensboro, March 2022
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/physicsSeminar" title="Machine Learning in HEP" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="width:100%!important; border:none;border-radius:0.25rem;">
>
> </iframe>
>
> </div>

#### 📆 2021

> [!TIP]
>
> ### <span class="dim-text">[**Accelerated Sampling Methods for LGT**], @ [DWQ @ 25 \[BNL\]][16] \[12/2021\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/l2hmc-dwq25/" title="Accelerated Sampling Methods for LGT" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Training Topological Samplers for LGT**] @ [ML4HEP, ECT\* Trento][] \[09/2021\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://saforem2.github.io/l2hmc_talk_ect2021" title="Training Topological Samplers for LGT" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**l2hmc-qcd**][Building better sampling methods for Lattice QCD] @ MIT Lattice Group Seminar \[2021\]</span>
>
> [**l2hmc-qcd**][Building better sampling methods for Lattice QCD] at
> the *MIT Lattice Group Seminar*, 2021

> [!TIP]
>
> ### <span class="dim-text">[**Deep Learning HMC for Improved Gauge Generation**] @ [ML in LQCD Workshop][] \[2021\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://slides.com/samforeman/dlhmc/embed" title="Deep Learning HMC for Improved Gauge Generation" scrolling="no" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

#### 📆 2020

> [!TIP]
>
> ### <span class="dim-text">[**Machine Learning for Lattice QCD**] @ U. Iowa \[2020\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" allow="picture-in-picture" src="https://slides.com/samforeman/l2hmc-qcd/embed" title="Machine Learning for Lattice QCD" align="center" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen>
>
> </iframe>
>
> </div>

### 📬 Posts

<div id="listing-posts">

</div>

### 📦 Projects

> [!TIP]
>
> ### <span class="dim-text-11">📊 GitHub Stats</span>
>
> <div class="columns"
> style="display: flex; flex-direction: row; align-items: center; text-align:left;">
>
> <div class="column" style="text-align: left;width: 57%;">
>
> [![][17]][2]
>
> </div>
>
> <div class="column" style="text-align: left; width: 43%">
>
> [![][18]][19]
>
> </div>
>
> </div>
>
> [![][20]][21] <img
> src="https://github-readme-activity-graph.vercel.app/graph?username=saforem2&amp;theme=minimal&amp;bg_color=00000000&amp;line=838383&amp;color=838383&amp;days=30&amp;point=838383&amp;hide_border=true&amp;hide_title=true&amp;area=true.png"
> id="img-stretch" />
>
> <details closed>
>
> <summary>
>
> Even More !!
> </summary>
>
> <details closed>
>
> <summary>
>
> Wakatime
> </summary>
>
> [![][22]][2]
>
> </details>
>
> ![][23]
>
> </details>

> [!TIP]
>
> ### <span class="dim-text-11">📂 [`saforem2/`]</span>
>
> <div class="grid"
> style="grid-template-columns: repeat(var(--bs-columns, 2), 1fr);">
>
> [![][24]][Building better sampling methods for Lattice QCD]
>
> [![][25]][26]
>
> [![][27]][Scaling Large Language Models]
>
> [![][28]][29]
>
> [![][30]][31]
>
> [![][32]][33]
>
> [![][34]][35]
>
> [![][36]][37]
>
> [![][38]][39]
>
> [![][40]][41]
>
> [![][42]][43]
>
> [![][36]][37]
>
> [![][44]][45]
>
> [![][46]][47]
>
> [![][48]][49]
>
> [![][50]][51]
>
> [![][52]][53]
>
> [![][54]][55]
>
> [![][56]][AI4Science on Supercomputers (ALCF)]
>
> [![][57]][Distributed training across thousands of GPUs]
>
> [![][58]][59]
>
> [![][60]][31]
>
> </div>

### 🪖 Experience

#### 🎪 Events

- Organizer for:

  - [SC24 Workshop: High Performance Python for Science at Scale
    (HPPSS)], November 2024

  - [SC23 Workshop: High Performance Python for Science at Scale
    (HPPSS)], November 2023

  - [Machine Learning and Quantum Computing for Earth Sciences] at
    17th U. S. National Congress on Computational Mechanics, July 2023

#### 👔 Employment

<div id="tbl-experience">

Table 1: 📟 Experience

| Position                                  |       @        | Start | End  |
|:------------------------------------------|:--------------:|:-----:|:----:|
| [Assistant Computational Scientist][ALCF] | [ALCF][(ALCF)] | 2022  |  –   |
| Postdoc                                   |      ALCF      | 2019  | 2022 |
| Graduate Researcher                       |     [ANL]      | 2018  | 2019 |

</div>

#### 🍎 School

<div id="tbl-education">

Table 2: 🎓 Education

| Degree |      In       |         @          | End  |
|:-------|:-------------:|:------------------:|:----:|
| [PhD]  |   [Physics]   | University of Iowa | 2019 |
| B.Sc   | [Physics][61] |       [UIUC]       | 2015 |
| B.Sc   |    [Math]     |        UIUC        | 2015 |

</div>

### 🎶 Music

<div class="flex-container" style="align-items: baseline;">

[![][62]][63]

<div class="flex-container" style="flex-direction: column; width: 50%;">

> [!TIP]
>
> ### <span style="color:#D41109;">[![][12]][13]</span>
>
> <script>
> /**
>   Developed by Prashant Shrestha
>   + https://prashant.me
> */
> var lastfmData = {
>   baseURL:
>     "https://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks&user=",
>   // Your Last.fm Username
>   user: "saforem2",
>   // Your API key
>   api_key: "1dbc15037c1fe71ce06acbb3f73adc75",
>   additional: "&format=json&limit=1"
> };
> &#10;var getSetLastFM = function() {
>   $.ajax({
>     type: "GET",
>     url:
>       lastfmData.baseURL +
>       lastfmData.user +
>       "&api_key=" +
>       lastfmData.api_key +
>       lastfmData.additional,
>     dataType: "json",
>     success: function(resp) {
>       var recentTrack = resp.recenttracks.track[0];
>       var formatted =
>         // "<img src='https://api.iconify.design/streamline-emojis:musical-notes.svg?color=%23888888'>" + recentTrack.name;
>         "🎶 " + recentTrack.name;
>       $("a#tracktitle")
>         .html(formatted)
>         .attr("href", recentTrack.url)
>         .attr("title", recentTrack.name + " by " + recentTrack.artist["#text"])
>         .attr("target", "_blank");
> &#10;      var artistFormatted =
>         // "<img src='https://api.iconify.design/material-symbols:person.svg?color=%23888888'>" + recentTrack.artist["#text"];
>         "🗣️ " + recentTrack.artist["#text"];
>       $("a#trackartist")
>         .html(artistFormatted)
>         .attr("title", "Artist : " + recentTrack.artist["#text"]);
>       $("img#trackart").attr("src", recentTrack.image[2]["#text"]);
>     },
>     error: function(resp) {
>       $("a#tracktitle").html(
>         "<img src='https://api.iconify.design/streamline-emojis:muted-speaker.svg?color=%23888888'>" + "Silence!"
>       );
>       $("img#trackart").attr("src", "🧑🏻‍💻");
>       var artistFormatted =
>         "Sam Foreman";
>       $("a#trackartist")
>         .html(artistFormatted)
>         .attr("href", "https://samforeman.me");
>     }
>   });
> };
> &#10;// Get the new one.
> getSetLastFM();
> // Start the countdown.
> setInterval(getSetLastFM, 10 * 5000);
> </script> <div class="nowplayingcard">
> <div class="nowplayingcontainer-inner">
> <img id="trackart" src="#">
> <div class="trackInfo">
> <a id="tracktitle"></a>
> <a href="#" id="trackartist"></a>
> </div>
> </div>
> </div>

<span class="stretch">[<img src="https://lastfm-recently-played.vercel.app/api?user=saforem2" >][13]</span>

</div>

</div>

</div>

  [ALCF]: https://alcf.anl.gov/about/people/sam-foreman
  [1]: ./assets/avatar.webp
  [<span class="orcid-green" style="background: none!important;"></span>]:
    https://orcid.org/0000-0002-9981-0876
  [<span class="icon dim-text" style="font-size: 1.5rem; padding-right:0pt;"></span>]:
    https://samforeman.me
  [2]: https://github.com/saforem2/
  [3]: https://www.twitter.com/saforem2
  [4]: mailto:///foremans@anl.gov
  [5]: https://scholar.google.com/citations?user=vV_1zDwAAAAJ&hl=en
  [6]: https://open.spotify.com/user/saforem2
  [7]: https://www.last.fm/user/saforem2
  [8]: https://linkedin.com/in/saforem2
  [9]: https://outlook.office.com/bookwithme/user/450ab3e5d58a4e7f84c802cc4c7205e6@anl.gov?anonymous&ep=plink
  [(ALCF)]: https://alcf.anl.gov
  [here]: talks/index.qmd
  [make them]: ./posts/dope-slides/index.qmd
  [10]: https://spotify-github-profile.kittinanx.com/api/view?uid=saforem2&cover_image=true&theme=novatorem&show_offline=false&background_color=none&interchange=true.png
  [11]: https://lastfm.com/user/saforem2
  [12]: https://api.iconify.design/logos:lastfm.svg?color=%23888888
  [13]: https://last.fm/user/saforem2
  [AI / ML Group]: https://www.alcf.anl.gov/about/people/group/506
  [Building better sampling methods for Lattice QCD]: https://github.com/saforem2/l2hmc-qcd
  [Genome-Scale Language Models]: https://www.biorxiv.org/content/10.1101/2022.10.10.511571v2
  [GenSLM]: https://github.com/ramanathanlab/genslm
  [ACM Gordon Bell Special Prize]: https://www.acm.org/media-center/2022/november/gordon-bell-special-prize-covid-research-2022
  [Foundation models for long term climate forecasting]: https://saforem2.github.io/climate-analysis
  [Scaling Large Language Models]: https://github.com/argonne-lcf/Megatron-DeepSpeed
  [Distributed training across thousands of GPUs]: https://github.com/argonne-lcf/mlprof
  [current research]: https://saforem2.github.io/l2hmc-qcd
  [Learning Better Physics: A Machine Learning Approach to Lattice Gauge Theory]:
    https://iro.uiowa.edu/esploro/outputs/doctoral/Learning-better-physics-a-machine-learning/9983776792002771
  [Energy Storage in Quantum Resonators]: https://aip.scitation.org/doi/10.1063/1.5009698
  [Alfred Hübler]: https://en.wikipedia.org/wiki/Alfred_H%C3%BCbler
  [patent]: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=vV_1zDwAAAAJ&pagesize=80&citation_for_view=vV_1zDwAAAAJ:SeFeTyx0c_EC
  [Intro to HPC Bootcamp: Engaging New Communities Through Energy Justice Projects]:
    https://jocse.org/downloads/jocse-15-1-10.pdf
  [Thorough Characterization and Analysis of Large Transformer Model Training At-Scale]:
    https://doi.org/10.1145/3639034
  [MLMC: Machine Learning Monte Carlo for Lattice Gauge Theory]: https://arxiv.org/abs/2312.08936
  [Lattice, 2023 (Proceedings)]: https://indico.fnal.gov/event/57249/
  [Protein Generation via Genome-scale Language Models with Bio-physical Scoring]:
    https://dl.acm.org/doi/abs/10.1145/3624062.3626087
  [14]: https://arxiv.org/abs/2310.04610
  [NeurIPS 2023 AI For Science Workshop]: https://ai4sciencecommunity.github.io/neurips23.html
  [DeepSpeed4Science.ai Blog Post]: https://deepspeed4science.ai/2023/09/18/model-showcase-genslms/
  [Loooooooong Sequence Lengths]: ./posts/AuroraGPT/long-sequences/index.qmd
  [Comprehensive Performance Study of LLMs on Novel AI Accelerators]: https://arxiv.org/abs/2310.04607
  [IPDPS 2024]: https://www.ipdps.org/
  [Intro to HPC Bootcamp @ NERSC]: https://github.com/NERSC/intro-HPC-bootcamp-2023
  [**GenSLMs: Genome-scale language models reveal SARS-Cov-2 evolutionary dynamics**]:
    https://www.biorxiv.org/content/10.1101/2022.10.10.511571v1.abstract
  [Lattice QCD and Particle Physics]: https://arxiv.org/abs/2207.07641
  [Applications of ML to Lattice QFT]: https://arxiv.org/abs/2202.05838
  [LeapFrogLayers: Trainable Framework for Effective Sampling]: https://arxiv.org/abs/2112.01582
  [Lattice, *2021*]: https://indico.cern.ch/event/1006302
  [HMC with Normalizing Flows]: https://arxiv.org/abs/2112.01586
  [slides]: https://indico.cern.ch/event/1006302/contributions/4380743/
  [15]: https://indico.cern.ch/event/1006302/
  [Deep Learning Hamiltonian Monte Carlo]: https://arxiv.org/abs/2105.03418
  [+ poster]: https://simdl.github.io/posters/57-supp_DLHMC_Foreman_SimDL-ICLR2021_poster1.pdf
  [SimDL Workshop @ ICLR]: https://simdl.github.io/
  [Machine Learning and Neural Networks for Field Theory]: https://bit.ly/snowmass_ml2020
  [SnowMass]: https://snowmass21.org/
  [Examples of renormalization group transformations for image sets]: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.052129
  [RG inspired Machine Learning for lattice field theory]: https://arxiv.org/abs/1710.02079
  [*arXiv:1710.02079*]: https://www.arxiv.or/abs/1710.02079
  [Large Energy Density in Three-Plate Nanocapacitors due to Coulomb Blockade]:
    https://doi.org/10.1063/1.5009698
  [**AuroraGPT**]: https://samforeman.me/talks/hpc-user-forum/slides
  [*HPC User Forum*, 2024]: https://www.hpcuserforum.com/hpc-user-forum-fall-2024/
  [**Training LLMs at Scale**]: https://samforeman.me/talks/llms-at-scale/slides
  [*ATPESC*, 2024]: https://extremecomputingtraining.anl.gov/atpesc-2024/
  [**LLMs on Polaris**]: https://samforeman.me/talks/llms-on-polaris/slides
  [*Center for Scientific Foundation Models*, Summer School 24’]: https://scifm.ai/summer_school.html
  [**Parallel Training Techniques**]: https://github.com/saforem2/parallel-training-slides
  [*AI-4-Science Training Series*]: https://github.com/argonne-lcf/ai-science-training-series/tree/main/06_parallel_training
  [**LLMs from Scratch**]: https://saforem2.github.io/llm-workshop-talk
  [LLM Tutorial Workshop]: https://github.com/argonne-lcf/llm-workshop
  [**Creating Small(-ish) LLMs**]: https://saforem2.github.io/LLM-tutorial
  [LLM Tutorial Workshop (1)]: https://github.com/brettin/llm_tutorial
  [**Exascale Science on Aurora**]: https://saforem2.github.io/oneapi-talk
  [Intel oneAPI Workshop @ UIC]: https://www.alcf.anl.gov/events/alcf-hands-hpc-workshop
  [**LLM Lunch Talk**]: https://saforem2.github.io/llm-lunch-talk
  [**Scaling LLMs for Science**]: https://saforem2.github.io/scaling4science
  [Data-Intensive Computing + AI/ML at Scale]: https://events.cels.anl.gov/event/426/overview
  [**MLMC: Machine Learning Monte Carlo**]: https://saforem2.github.io/lattice23
  [Lattice 2023]: https://indico.fnal.gov/event/57249/contributions/271305/
  [**Generative Modeling and Efficient Sampling**]: https://saforem2.github.io/lqcd-pasc23/
  [PASC23]: https://pasc23.pasc-conference.org/
  [**Efficient Sampling for LGT**]: https://saforem2.github.io/deep-fridays
  [Deep Fridays @ U. Bologna]: https://www.cs.unibo.it/~asperti/deep_fridays.html
  [**Large Scale Training**]: https://saforem2.github.io/ai4sci-large-scale-training
  [AI4Science on Supercomputers (ALCF)]: https://github.com/argonne-lcf/ai-science-training-series
  [**Hyperparameter Management**]: https://saforem2.github.io/hparam-management-sdl2022/
  [ALCF SDL Workshop]: https://www.alcf.anl.gov/events/2022-alcf-simulation-data-and-learning-workshop
  [**Statistical Learning**]: https://saforem2.github.io/ATPESC-StatisticalLearning
  [ATPESC 2022]: https://extremecomputingtraining.anl.gov/
  [📕 accompanying notebook]: https://github.com/argonne-lcf/ATPESC_MachineLearning/blob/master/00_statisticalLearning/src/atpesc/notebooks/statistical_learning.ipynb
  [**Scientific Data Science: An Emerging Symbiosis**]: https://saforem2.github.io/anl-job-talk/
  [**Machine Learning in HEP**]: https://saforem2.github.io/physicsSeminar
  [**Accelerated Sampling Methods for LGT**]: https://saforem2.github.io/l2hmc-dwq25/
  [16]: https://indico.bnl.gov/event/13576/
  [**Training Topological Samplers for LGT**]: https://saforem2.github.io/l2hmc_talk_ect2021
  [ML4HEP, ECT\* Trento]: https://indico.ectstar.eu/event/77/contributions/2349/
  [**Deep Learning HMC for Improved Gauge Generation**]: https://bit.ly/mainz21
  [ML in LQCD Workshop]: https://bit.ly/mainz21_overview
  [**Machine Learning for Lattice QCD**]: https://slides.com/samforeman/l2hmc-qcd/embed
  [17]: https://raw.githubusercontent.com/saforem2/github-stats/master/generated/overview.svg
  [18]: https://github-readme-stats.vercel.app/api/top-langs/?username=saforem2&layout=compact&langs_count=10&theme=transparent&hide_title=true&hide_border=true&text_color=838383.png
  [19]: https://github.com/saforem2/github-readme-stats
  [20]: https://github-readme-streak-stats.herokuapp.com?user=saforem2&theme=transparent&hide_border=true&card_width=800&card_height=200&stroke=838383&currStreakNum=838383&dates=838383&currStreakLabel=838383&background=EB545400&border=83838300&ring=8383836F&fire=FF5252&sideNums=838383&sideLabels=838383
  [21]: https://git.io/streak-stats
  [22]: https://github-readme-stats.vercel.app/api/wakatime?username=saforem2&show_icons=true&include_all_commits=true&title_color=838383&hide_border=true&layout=compact&theme=transparent&text_color=838383.png
  [23]: https://raw.githubusercontent.com/saforem2/saforem2/main/github-metrics.svg
  [`saforem2/`]: https://github.com/saforem2?tab=repositories
  [24]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=l2hmc-qcd&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [25]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=ezpz&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [26]: https://github.com/saforem2/ezpz
  [27]: https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=Megatron-DeepSpeed&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [28]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=wordplay&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [29]: https://github.com/saforem2/wordplay
  [30]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=personal_site&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [31]: https://github.com/saforem2/personal_site
  [32]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=ambivalent&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [33]: https://github.com/saforem2/ambivalent
  [34]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=enrich&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [35]: https://github.com/saforem2/enrich
  [36]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=public&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [37]: https://github.com/saforem2/public
  [38]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=lattice23&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [39]: https://github.com/saforem2/lattice23
  [40]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=llm-workshop-talk&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [41]: https://github.com/saforem2/llm-workshop-talk
  [42]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=quarto-site-template&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [43]: https://github.com/saforem2/quarto-site-template
  [44]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=nvim&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [45]: https://github.com/saforem2/nvim
  [46]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=glitz&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [47]: https://github.com/saforem2/glitz
  [48]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=saforem2&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [49]: https://github.com/saforem2/saforem2
  [50]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=awesome-stars&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [51]: https://github.com/saforem2/awesome-stars
  [52]: https://github-readme-stats.vercel.app/api/pin/?username=nftqcd&repo=fthmc&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [53]: https://github.com/nftqcd/fthmc
  [54]: https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=CompPerfWorkshop&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383.png
  [55]: https://github.com/argonne-lcf/CompPerfWorkshop
  [56]: https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=ai-science-training-series&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [57]: https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=mlprof&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [58]: https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=user-guides&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383.png
  [59]: https://github.com/argonne-lcf/user-guides
  [60]: https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=personal_site&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&icon_color=838383&title_color=838383.png
  [SC24 Workshop: High Performance Python for Science at Scale (HPPSS)]:
    https://hppss.github.io/SC24/
  [SC23 Workshop: High Performance Python for Science at Scale (HPPSS)]:
    https://hppss.github.io/SC23/
  [Machine Learning and Quantum Computing for Earth Sciences]: https://17.usnccm.org/702
  [ANL]: https://anl.gov
  [PhD]: https://bit.ly/sam-foreman-phd
  [Physics]: https://physics.uiowa.edu/graduate/phd-physics
  [61]: https://grainger.illinois.edu/academics/undergraduate/majors-and-minors/physics
  [UIUC]: https://illinois.edu/
  [Math]: https://math.illinois.edu/
  [62]: https://spotify-github-profile.kittinanx.com/api/view?uid=saforem2&cover_image=true&theme=default&show_offline=false&background_color=1c1c1c&interchange=false.png
  [63]: https://github.com/kittinan/spotify-github-profile
