Sam Foreman
2025-05-23

<link rel="preconnect" href="https://fonts.googleapis.com">

<div style="font-size:1.0em; text-align: center;">

<span class="profile-avatar"><img width=75 height=75 src="./assets/avatar-100x100.webp" width="100%" aria-label="Sam Foreman" alt="Sam Foreman"></img></span>

<span style="font-size: 1.5rem; color: var(--dim-text)!important; padding-bottom: 0pt;"><span class="dim-text">👋
Hi, I’m Sam!</span> [<span class="orcid-green"
style="background: none!important;"></span>]</span>

<div class="flex-container"
style="display: flex; flex-direction: row; align-items: center; text-align: center!important; justify-content: center; gap: 5pt; background-color: var(--link-bg-color);font-size:1.5rem;">

<a style="color: #838383;" href="https://samforeman.me"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:house" aria-label="Homepage" title="Homepage"></iconify-icon></a>

<a style="color: #838383;" href="https://github.com/saforem2" ><iconify-icon loading="lazy" role="img" inline="true" icon="ph:github-logo" aria-label="GitHub" title="GitHub"></iconify-icon></a>

<a style="color: #838383;" href="https://twitter.com/saforem2"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:twitter-logo" aria-label="Twitter" title="Twitter"></iconify-icon></a>

<a style="color: #838383;" href="https://bsky.com/samforeman"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:butterfly" aria-label="Bluesky" title="Bluesky"></iconify-icon></a>

<a style="color: #838383;" href="https://scholar.google.com/citations?user=vV_1zDwAAAAJ&hl=en"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:graduation-cap" aria-label="Google Scholar" title="Google Scholar"></iconify-icon></a>

<a style="color: #838383;" href="mailto:foremans@anl.gov"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:envelope-open" aria-label="Email" title="Email"></iconify-icon></a>

<a style="color: #838383;" href="https://outlook.office.com/"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:calendar" aria-label="Schedule Time" title="Email"></iconify-icon></a>

<a style="color: #838383;" href="https://linkedin.com/in/saforem2"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:linkedin-logo" aria-label="LinkedIn" title="LinkedIn"></iconify-icon></a>

<a style="color: #838383;" href="https://open.spotify.com/user/saforem2"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:spotify-logo" aria-label="Spotify" title="Spotify"></iconify-icon></a>

<a style="color: #838383;" href="https://www.last.fm/user/saforem2"><iconify-icon loading="lazy" role="img" inline="true" icon="ph:lastfm-logo" aria-label="LastFM" title="LastFM"></iconify-icon></a>

</div>

</div>

<div class="panel-tabset"
style="justify-content: center; loading='lazy';">

### 🧑🏻‍💻 About

<div class="flex-container"
style="width: 100%; justify-content: space-between; align-items: flex-start;">

<div class="column" style="width: 54%;">

- [Computational scientist][ALCF] @ Argonne National Laboratory
  - AI / ML [Group] @ [ALCF][1]
  - Working on:
    - 🧪 {AI, HPC} for [science]
    - 🚀 [training large models] on [supercomputers]

</div>

<div class="column">

> [!TIP]
>
> ### 🎤 <span class="dim-text">Recent Talks</span>
>
> <span class="dim-text" style="font-size:1em;">\[[here]\] ( + how I
> [make them]! )</span>

> [!TIP]
>
> ### <span style="color:#1CD760;"><img src="./assets/spotify-green.svg" class="inline-icon img-fluid" height="24" width="24" style="height:1.25rem; width: auto; vertical-align:text-top;" alt="spotify" /> Now Playing</span>
>
> <a href="https://open.spotify.com/user/saforem2" target="_blank"><img loading="lazy" src="https://spotify-github-profile.kittinanx.com/api/view?uid=saforem2&cover_image=true&theme=novatorem&show_offline=false&background_color=none&interchange=true" alt="Now Playing" /></a>
>
> > [!TIP]
> >
> > ### <a href="https://last.fm/user/saforem2" target="_blank"><img src="https://api.iconify.design/logos:lastfm.svg" alt="last.fm" style="overflow: visible;"/></a>
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

> [!TIP]
>
> ### <span class="dim-text">➕ More</span>
>
> - <details closed>
>
>   <summary>
>
>   🔥 What I work on
>
>   </summary>
>
>   As a member of the [AI / ML Group][Group] at [ALCF][1], I work on:
>
>   <div class="flex-container">
>
>   <div class="flex-container">
>
>   - 🤖 🧪 [AI + Science]
>
>   - 🎲 [Building better sampling methods for Lattice QCD]
>
>   - 🧬 [Genome-Scale Language Models]
>
>     - [ GenSLM]
>
>     - 🥇 [ACM Gordon Bell Special Prize]
>
>   </div>
>
>   <div class="flex-container">
>
>   - 🌍 [Foundation models for long term climate forecasting]
>
>   - 🏃‍♂️ [Scaling Large Language Models]
>
>   - 🏎️ [Distributed training across thousands of GPUs]
>
>   </div>
>
>   </div>
>
>   </details>
>
> - <details closed>
>
>   <summary>
>
>   📍 How I got here
>
>   </summary>
>
>   My [current research] focuses on using deep generative modeling to
>   help build better sampling algorithms in lattice gauge theory. In
>   particular, I’m interested in building gauge equivariant neural
>   network architectures and using inductive priors to incorporate
>   physical symmetries into machine learning models.
>
>   I received my PhD in Physics from the University of Iowa in 2019 and
>   my thesis was on [Learning Better Physics: A Machine Learning
>   Approach to Lattice Gauge Theory].
>
>   Prior to this, I completed two bachelors degrees (Engineering
>   Physics and Applied Mathematics, 2015) at The University of Illinois
>   at Urbana-Champaign. My undergraduate dissertation was titled
>   [Energy Storage in Quantum Resonators] and was supervised by
>   Professor [Alfred Hübler] within the Center for Complex Systems
>   Research at UIUC.
>
>   This work ultimately resulted in a [patent] !!
>
>   </details>
>
> > [!NOTE]
> >
> > ### Headings
> >
> > # Heading 1
> >
> > Content 1
> >
> > ## Heading 2
> >
> > Content 2
> >
> > ### Heading 3
> >
> > Content 3
> >
> > #### Heading 4
> >
> > Content 4
> >
> > ##### Heading 5
> >
> > Content 5
> >
> > ###### Heading 6
> >
> > Content 6
> >
> > Inspired by my neovim config:
> >
> > ![][2]
>
> ``` python
> import datetime
> from rich import print
> now = datetime.datetime.now()
> print(
>     ' '.join([
>         "[#838383]Last Updated[/]:",
>         f"[#E599F7]{now.strftime("%Y-%m-%d")}[/]",
>         "[#838383]@[/]",
>         f"[#00CCFF]{now.strftime("%H:%M:%S")}[/]", 
>     ])
> )
> ```
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #838383; text-decoration-color: #838383">Last Updated</span>: <span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">2025</span><span style="color: #e599f7; text-decoration-color: #e599f7">-</span><span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">05</span><span style="color: #e599f7; text-decoration-color: #e599f7">-</span><span style="color: #e599f7; text-decoration-color: #e599f7; font-weight: bold">23</span> <span style="color: #838383; text-decoration-color: #838383">@</span> <span style="color: #00ccff; text-decoration-color: #00ccff; font-weight: bold">07:20:50</span>
> </pre>
>
> <div style="text-align:center;">
>
> <iframe src="https://github.com/sponsors/saforem2/button" title="Sponsor saforem2" height="32" width="114" style="border: 0; border-radius: 6px;">
>
> </iframe>
>
> <img alt="hits" src="https://hitscounter.dev/api/hit?url=samforeman.me&label=samforeman.me&icon=check2-square&color=%236c757d">
>
> <span class="dim-text">© Copyright [Sam Foreman]</span>
>
> </div>

### 📝 Work

> [!NOTE]
>
> <span style="color:#4582ec;">You can find a full list of my
> publications on my [Google Scholar]</span>

- [**MProt-DPO: Breaking the ExaFLOPS Barrier for Multimodal Protein
  Design Workflows with Direct Preference Optimization**]
  <span class="dim-text">G. Dharuman, K. Hippe, A. Brace, **S.
  Foreman**, et al. @ [SC’24]</span>

  -  [<span class="highlight-pink">2024 Gordon Bell Finalist</span>]

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
  Discovery \[…\]][3]  
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

-  [**GenSLMs: Genome-scale language models reveal SARS-Cov-2
  evolutionary dynamics**]  
  <span class="dim-text">@ SC’22 *10/2022*</span>

  - 🥇 [<span class="highlight-pink">ACM Gordon Bell Special
    Prize</span>]

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
  *2021*][4]</span>

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

## References

> [!TIP]
>
> ### <span class="dim-text">📓 References</span>
>
> - References:
>   - (Dharuman et al. 2024)
>   - (Parete-Koon et al. 2024)
>   - (Cheng et al. 2024)
>   - (Zvyagin et al. 2023)
>   - (Dharuman et al. 2023)
>   - (Emani et al. 2023)
>   - (Song et al. 2023)
>   - (Sam Foreman, Jin, and Osborn)
>   - (Boyda et al. 2022)
>   - (Kronfeld et al. 2022)
>   - (Shanahan, Terao, and Whiteson 2022)
>   - (S. Foreman, Jin, and Osborn 2022)
>   - (Sam Foreman et al. 2021)
>   - (Sam Foreman, Jin, and Osborn 2020)
>   - (S. A. Foreman 2019)
>   - (Sam Foreman et al. 2018)
>   - (Hubler et al. 2018)
>   - (Samuel Foreman et al. 2018)
>   - (Liu et al. 2017)
>   - (Deamont and Foreman 2014)
>
> <div id="refs" class="references csl-bib-body hanging-indent"
> entry-spacing="0">
>
> <div id="ref-boyda2022applications" class="csl-entry">
>
> Boyda, Denis, Salvatore Calı̀, Sam Foreman, Lena Funcke, Daniel C
> Hackett, Yin Lin, Gert Aarts, et al. 2022. “Applications of Machine
> Learning to Lattice Quantum Field Theory.” *arXiv Preprint
> arXiv:2202.05838*.
>
> </div>
>
> <div id="ref-cheng2024thorough" class="csl-entry">
>
> Cheng, Scott, Jun-Liang Lin, Murali Emani, Siddhisanket Raskar, Sam
> Foreman, Zhen Xie, Venkatram Vishwanath, and Mahmut Taylan Kandemir.
> 2024. “Thorough Characterization and Analysis of Large Transformer
> Model Training at-Scale.” *Proceedings of the ACM on Measurement and
> Analysis of Computing Systems* 8 (1): 1–25.
>
> </div>
>
> <div id="ref-deamont2014superconductivity" class="csl-entry">
>
> Deamont, George, and Sam Foreman. 2014. “Superconductivity of in and
> Sn Samples.”
>
> </div>
>
> <div id="ref-dharuman2024mprot" class="csl-entry">
>
> Dharuman, Gautham, Kyle Hippe, Alexander Brace, Sam Foreman, Väinä
> Hatanpää, Varuni K Sastry, Huihuo Zheng, et al. 2024. “MProt-DPO:
> Breaking the ExaFLOPS Barrier for Multimodal Protein Design Workflows
> with Direct Preference Optimization.” In *2024 SC24: International
> Conference for High Performance Computing, Networking, Storage and
> Analysis SC*, 74–86. IEEE Computer Society.
>
> </div>
>
> <div id="ref-dharuman2023protein" class="csl-entry">
>
> Dharuman, Gautham, Logan Ward, Heng Ma, Priyanka V Setty, Ozan
> Gokdemir, Sam Foreman, Murali Emani, et al. 2023. “Protein Generation
> via Genome-Scale Language Models with Bio-Physical Scoring.” In
> *Proceedings of the SC’23 Workshops of the International Conference on
> High Performance Computing, Network, Storage, and Analysis*, 95–101.
>
> </div>
>
> <div id="ref-emani2023comprehensive" class="csl-entry">
>
> Emani, Murali, Sam Foreman, Varuni Sastry, Zhen Xie, Siddhisanket
> Raskar, William Arnold, Rajeev Thakur, Venkatram Vishwanath, and
> Michael E Papka. 2023. “A Comprehensive Performance Study of Large
> Language Models on Novel AI Accelerators.” *arXiv Preprint
> arXiv:2310.04607*.
>
> </div>
>
> <div id="ref-foreman2018rg" class="csl-entry">
>
> Foreman, Sam, Joel Giedt, Yannick Meurice, and Judah Unmuth-Yockey.
> 2018. “RG-Inspired Machine Learning for Lattice Field Theory.” In *EPJ
> Web of Conferences*, 175:11025. EDP Sciences.
>
> </div>
>
> <div id="ref-foreman2021hmc" class="csl-entry">
>
> Foreman, Sam, Taku Izubuchi, Luchang Jin, Xiao-Yong Jin, James C
> Osborn, and Akio Tomiya. 2021. “HMC with Normalizing Flows.” *arXiv
> Preprint arXiv:2112.01586*.
>
> </div>
>
> <div id="ref-foreman2023mlmc" class="csl-entry">
>
> Foreman, Sam, Xiao-Yong Jin, and James Osborn. “MLMC: Machine Learning
> Monte Carlo for Lattice Gauge Theory.” In *40th International
> Symposium on Lattice Field Theory (Lattice 2023) (Batavia, IL, United
> States, 07/31/2023 - 08/04/2023)*.
>
> </div>
>
> <div id="ref-foreman2020machine" class="csl-entry">
>
> Foreman, Sam, Xiao-Yong Jin, and James C Osborn. 2020. “Machine
> Learning and Neural Networks for Field Theory.”
>
> </div>
>
> <div id="ref-foreman2019learning" class="csl-entry">
>
> Foreman, Samuel Alfred. 2019. “Learning Better Physics: A Machine
> Learning Approach to Lattice Gauge Theory.” PhD thesis, University of
> Iowa.
>
> </div>
>
> <div id="ref-foreman2018examples" class="csl-entry">
>
> Foreman, Samuel, Joel Giedt, Yannick Meurice, and Judah Unmuth-Yockey.
> 2018. “Examples of Renormalization Group Transformations for Image
> Sets.” *Physical Review E* 98 (5): 052129.
>
> </div>
>
> <div id="ref-2022slft.confE.508F" class="csl-entry">
>
> Foreman, S., X. y. Jin, and J. Osborn. 2022.
> “<span class="nocase">LeapfrogLayers: A Trainable Framework for
> Effective Topological Sampling</span>.” In *The 38th International
> Symposium on Lattice Field Theory*, 508.
> <https://doi.org/10.22323/1.396.0508>.
>
> </div>
>
> <div id="ref-hubler2018large" class="csl-entry">
>
> Hubler, A, S Foreman, J Liu, and L Wortsmann. 2018. “Large Energy
> Density in Three-Plate Nanocapacitors Due to Coulomb Blockade.”
> *Journal of Applied Physics* 123 (10).
>
> </div>
>
> <div id="ref-kronfeld2022lattice" class="csl-entry">
>
> Kronfeld, Andreas S, Tanmoy Bhattacharya, Thomas Blum, Norman H
> Christ, Carleton DeTar, William Detmold, Robert Edwards, et al. 2022.
> “Lattice QCD and Particle Physics.” *arXiv Preprint arXiv:2207.07641*.
>
> </div>
>
> <div id="ref-liu2017energy" class="csl-entry">
>
> Liu, Jiaqi, Alfred W Hubler, Samuel Alfred Foreman, and Katharina Ott.
> 2017. “Energy Storage in Quantum Resonators.”
>
> </div>
>
> <div id="ref-parete2024intro" class="csl-entry">
>
> Parete-Koon, Suzanne, Michael Sandoval, Kellen Leland, Subil Abraham,
> Mary Ann Leung, Rebecca Hartman-Baker, Paige Kinsley, et al. 2024.
> “Intro to HPC Bootcamp: Engaging New Communities Through Energy
> Justice Projects.” *Journal of Computational Science Education* 15
> (1).
>
> </div>
>
> <div id="ref-shanahan2022snowmass" class="csl-entry">
>
> Shanahan, Phiala, Kazuhiro Terao, and Daniel Whiteson. 2022. “Snowmass
> 2021 Computational Frontier CompF03 Topical Group Report: Machine
> Learning.” *arXiv Preprint arXiv:2209.07559*.
>
> </div>
>
> <div id="ref-song2023deepspeed4science" class="csl-entry">
>
> Song, Shuaiwen Leon, Bonnie Kruft, Minjia Zhang, Conglong Li, Shiyang
> Chen, Chengming Zhang, Masahiro Tanaka, et al. 2023.
> “DeepSpeed4Science Initiative: Enabling Large-Scale Scientific
> Discovery Through Sophisticated AI System Technologies.” *arXiv
> Preprint arXiv:2310.04610*.
>
> </div>
>
> <div id="ref-zvyagin2023genslms" class="csl-entry">
>
> Zvyagin, Maxim, Alexander Brace, Kyle Hippe, Yuntian Deng, Bin Zhang,
> Cindy Orozco Bohorquez, Austin Clyde, et al. 2023. “GenSLMs:
> Genome-Scale Language Models Reveal SARS-CoV-2 Evolutionary Dynamics.”
> *The International Journal of High Performance Computing Applications*
> 37 (6): 683–705.
>
> </div>
>
> </div>

### 🦜 Talks

> [!TIP]
>
> ### \[HTML ⇆ Reveal.js\]
>
> Convert from HTML to slideshow version of a page by appending
> `/slides` to the end of its URL, e.g.
>
> - HTML: <https://samforeman.me/talks/ai-for-science-2024/>
> - Slides: <https://samforeman.me/talks/ai-for-science-2024/slides>

<div id="listing-talks">

</div>

## 📆 2025

> [!TIP]
>
> ### <span class="dim-text">[**LLMs on Aurora: 🌌 AuroraGPT**] @ [*2025 ALCF INCITE GPU Hackathon*][] \[05/2025\]</span>
>
> <div class="reveal-full-page">
>
> <iframe class="slide-deck reveal-full-page" loading="lazy" src="https://samforeman.me/talks/incite-hackathon-2025/AuroraGPT/slides#/section" title="LLMs on Aurora: AuroraGPT" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**LLMs on Aurora: 🍋 ezpz**] @ [*2025 ALCF INCITE GPU Hackathon*][] \[05/2025\]</span>
>
> <div class="reveal-full-page">
>
> <iframe class="slide-deck reveal-full-page" loading="lazy" src="https://samforeman.me/talks/incite-hackathon-2025/ezpz/slides#/section" title="🍋 ezpz on Aurora" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**AuroraGPT: Foundation Models for Science**] @ [*Foundation Models for the Electric Grid*][] \[02/2025\]</span>
>
> <div class="reveal-full-page">
>
> <iframe class="slide-deck reveal-full-page" loading="lazy" src="/talks/aurora-gpt-fm-for-electric-grid/slides.html" title="AuroraGPT: Foundation Models for Science" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

## 📆 2024

> [!TIP]
>
> ### <span class="dim-text">[**Parallel Training Methods**] @ [*AI-for-Science on Supercomputers*][*Foundation Models for the Electric Grid*] \[11/2024\]</span>
>
> <div class="reveal-full-page">
>
> <iframe class="slide-deck reveal-full-page" loading="lazy" src="/talks/ai-for-science-2024/slides.html" title="Parallel Training Methods" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**AuroraGPT**] @ [*2024 ALCF Hands-On HPC Workshop*][] \[10/2024\]</span>
>
> <div class="reveal-full-page">
>
> <iframe class="slide-deck" loading="lazy" src="/talks/AuroraGPT/alcf-hpc-workshop-2024/slides.html" title="AuroraGPT" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**Machine Learning and Foundation Models at Scale**] @ [*2024 ALCF Hands-On HPC Workshop*][] \[10/2024\]</span>
>
> <div class="reveal-full-page">
>
> <iframe class="slide-deck" loading="lazy" src="https://samforeman.me/talks/alcf-hpc-workshop-2024/slides#/section" title="Machine Learning and Foundation Models at Scale" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

> [!TIP]
>
> ### <span class="dim-text">[**AuroraGPT**][5] @ [*HPC User Forum*, 2024][] \[09/2024\]</span>
>
> <iframe class="slide-deck reveal-full-page" loading="lazy" src="/talks/hpc-user-forum/slides.html" title="AuroraGPT" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>

> [!TIP]
>
> ### <span class="dim-text">[**Training LLMs at Scale**] @ [*ATPESC*, 2024][] \[08/2024\]</span>
>
> <iframe class="slide-deck" loading="lazy" src="/talks/llms-at-scale/slides.html" title="Training LLMs at Scale" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>

> [!TIP]
>
> ### <span class="dim-text">[**LLMs on Polaris**] @ [*Center for Scientific Foundation Models*, Summer School 24’][] \[07/2024\]</span>
>
> <iframe class="slide-deck" loading="lazy" src="/talks/llms-on-polaris/slides.html" title="LLMs on Polaris" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>

> [!TIP]
>
> ### <span class="dim-text">[**Parallel Training Techniques**] @ [*AI-4-Science Training Series*][] \[03/2024\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/parallel-training-slides" title="Parallel Training Techniques" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/llm-workshop-talk" title="LLMs from Scratch" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

## 📆 2023

> [!TIP]
>
> ### <span class="dim-text">[**Creating Small(-ish) LLMs**] @ [LLM Tutorial Workshop (1)][] \[11/2023\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/LLM-tutorial" title="Creating Small(-ish) LLMs" align="center" frameborder="0" webkitallowfullscreen allowfullscreen>
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/oneapi-talk" title="Exascale Science on Aurora" align="center" frameborder="0" webkitallowfullscreen allowfullscreen>
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/llm-lunch-talk/#/section" title="LLMs on Polaris" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/scaling4science/#/section" title="Scaling LLMs for Science and Ongoing Collaborations" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/lattice23/#/title-slide" title="MLMC: Machine Learning Monte Carlo" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/lqcd-pasc23/" title="Generative Modeling and Efficient Sampling" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/deep-fridays/" title="Efficient Sampling for LGT" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

## 📆 2022

> [!TIP]
>
> ### <span class="dim-text">[**Large Scale Training**] @ [AI4Science on Supercomputers (ALCF)][] \[11/2022\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/ai4sci-large-scale-training/#" title="Large Scale Training" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/hparam-management-sdl2022" title="Hyperparameter Management" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/ATPESC-StatisticalLearning/#/" title="Statistical Learning" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/anl-job-talk" title="Scientific Data Science" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/physicsSeminar" title="Machine Learning in HEP" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="width:100%!important; ">
>
> </iframe>
>
> </div>

## 📆 2021

> [!TIP]
>
> ### <span class="dim-text">[**Accelerated Sampling Methods for LGT**], @ [DWQ @ 25 \[BNL\]][6] \[12/2021\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/l2hmc-dwq25/" title="Accelerated Sampling Methods for LGT" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://saforem2.github.io/l2hmc_talk_ect2021" title="Training Topological Samplers for LGT" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <iframe class="slide-deck" loading="lazy" src="https://slides.com/samforeman/dlhmc/embed" title="Deep Learning HMC for Improved Gauge Generation" scrolling="no" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
>
> </iframe>
>
> </div>

## 📆 2020

> [!TIP]
>
> ### <span class="dim-text">[**Machine Learning for Lattice QCD**] @ U. Iowa \[2020\]</span>
>
> <div class="embedded-slide">
>
> <iframe class="slide-deck" loading="lazy" src="https://slides.com/samforeman/l2hmc-qcd/embed" title="Machine Learning for Lattice QCD" align="center" frameborder="0" webkitallowfullscreen allowfullscreen style="aspect-ratio:1.3671875;">
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
> <div class="flex-container" style="flex-flow: wrap;">
>
> <a href="https://github.com/saforem2"><img loading="lazy" src="https://raw.githubusercontent.com/saforem2/github-stats/master/generated/overview.svg"></a>
> <a href="https://github.com/saforem2/github-readme-stats"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/top-langs/?username=saforem2&layout=compact&langs_count=10&theme=transparent&hide_title=true&hide_border=true&text_color=838383"></a>
>
> </div>
>
> <div style="width: 100%; text-align: center;">
>
> <a href="https://git.io/streak-stats"><img loading="lazy" align="center" width="100%" src="https://streak-stats.demolab.com?user=saforem2&theme=shadow-blue&hide_border=true&card_width=800&card_height=200&stroke=838383&currStreakNum=838383&dates=838383&currStreakLabel=838383&ring=838383&fire=FF5252&sideNums=838383&sideLabels=838383&date_format=n%2Fj%5B%2FY%5D&background=EB545400" alt="GitHub Streak" /></a>
>
> <img loading="lazy" width="100%" alt="Github Contributions " src="https://github-readme-activity-graph.vercel.app/graph?username=saforem2&theme=minimal&bg_color=00000000&line=838383&color=838383&days=30&point=838383&hide_border=true&hide_title=true&area=true">
>
> </div>
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
> <a href="https://github.com/saforem2/"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/wakatime?username=saforem2&show_icons=true&include_all_commits=true&title_color=838383&hide_border=true&layout=compact&theme=transparent&text_color=838383"></a>
>
> </details>
>
> <img loading="lazy" src="https://raw.githubusercontent.com/saforem2/saforem2/main/github-metrics.svg">
>
> </details>

> [!TIP]
>
> ### <span class="dim-text-11">📂 [`saforem2/`]</span>
>
> <div class="flex-container" style="flex-flow: wrap;">
>
> <a href="https://github.com/argonne-lcf/Megatron-DeepSpeed"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=Megatron-DeepSpeed&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/ezpz"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=ezpz&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/mmm"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=mmm&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/ambivalent"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=ambivalent&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/l2hmc-qcd"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=l2hmc-qcd&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/personal_site"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=personal_site&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/wordplay"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=wordplay&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/enrich"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=enrich&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/public"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=public&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/lattice23"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=lattice23&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="github.com/saforem2/llm-workshop-talk"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=llm-workshop-talk&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/quarto-site-template"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=quarto-site-template&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/starter"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=starter&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/glitz"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=glitz&theme=transparent&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/glitz"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=glitz&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/saforem2/awesome-stars"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=saforem2&repo=awesome-stars&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/nftqcd/fthmc"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=nftqcd&repo=fthmc&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/argonne-lcf/CompPerfWorkshop"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=CompPerfWorkshop&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/argonne-lcf/ai-science-training-series"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=ai-science-training-series&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/argonne-lcf/mlprof"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=mlprof&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> <a href="https://github.com/argonne-lcf/user-guides"><img loading="lazy" src="https://github-readme-stats.vercel.app/api/pin/?username=argonne-lcf&repo=user-guides&theme=transparent&show_icons=true&include_all_commits=true&hide_border=true&line_height=5&card_width=300px&text_color=838383&title_color=838383"></a>
>
> </div>

### 🪖 Experience

#### 🎓 Education

- **Ph.D., Physics**
  - University of Iowa, 2019  
  - [*Learning Better Physics: A Machine Learning Approach to Lattice
    Gauge Theory*]
- **B.S. in Engineering Physics**
  - University of Illinois at Urbana-Champaign, 2015
  - [Energy Storage in Quantum Resonators (US Patent \#US9741492B2)]
- **B.S. in Applied Mathematics**
  - University of Illinois at Urbana-Champaign, 2015

#### 🧑‍🔬 Professional Experience

- **Assistant Computational Scientist**
  - Argonne National Laboratory, Argonne Leadership Computing Facility
    (ALCF)  
  - *Lemont, IL \| 2022 – Present*
    - Research lead on scaling large language models (LLMs) and
      generative AI for science on supercomputers (Aurora, Frontier,
      LUMI, Leonardo, …).
    - Optimize large-scale training of foundation models and language
      models for scientific applications.  
    - Collaborate with interdisciplinary teams to enhance simulation
      efficiency and scalability.
    - Focus on AI and HPC for scientific applications, including:
      - Developing improved sampling algorithms for lattice quantum
        chromodynamics (QCD)
      - Training large language models on supercomputers
    - <https://www.alcf.anl.gov/about/people/sam-foreman>
- **Postdoctoral Researcher**
  - Argonne National Laboratory, Argonne Leadership Computing Facility
    (ALCF)  
  - *Lemont, IL \| 2019 – 2022*
    - Applied deep learning to lattice gauge theory and quantum field
      simulations.
    - Developed ML-enhanced Monte Carlo methods for QCD.
    - Engaged in AI-for-Science collaborations with national labs and
      university partners.
- **Graduate Researcher**
  - Argonne National Laboratory, Math and Computer Sciences (MCS)
  - *Lemont, IL \| 2018 – 2019*  
  - Collaborated with ALCF while completing Ph.D., integrating ML into
    physical sciences workflows.

#### 🏆 Awards and Honors

- **ACM Gordon Bell Special Prize for High Performance Computing-Based
  COVID-19 Research**, 2022
  - Recognized for contributions to the GenSLMs project, which developed
    genome-scale language models to study SARS-CoV-2 evolutionary
    dynamics.
  - <https://www.acm.org/media-center/2022/november/gordon-bell-special-prize-covid-research-2022>
- **Finalist, ACM Gordon Bell Prize**, 2024
  - Acknowledged for the MProt-DPO project, which achieved over 4
    ExaFLOP sustained performance in multimodal protein design workflows
    using Direct Preference Optimization.
  - <https://sc.cels.anl.gov/gordon-bell-argonne-team-breaks-new-ground-in-ai-driven-protein-design/>
- **DOE Office of Science Graduate Student Research Fellow**, 2018
  - Awarded by the Department of Energy for outstanding research
    contributions during graduate studies.

#### 📚 Publications[^1]

- [**MProt-DPO: Breaking the ExaFLOPS Barrier for Multimodal Protein
  Design Workflows with Direct Preference Optimization**][7]
- [**GenSLMs: Genome-Scale Language Models Reveal SARS-CoV-2
  Evolutionary Dynamics**][8]
- [**Applications of Machine Learning to Lattice Quantum Field
  Theory**][Applications of ML to Lattice QFT]
- [**HMC with Normalizing Flows**][HMC with Normalizing Flows]
- [**Deep Learning Hamiltonian Monte
  Carlo**][Deep Learning Hamiltonian Monte Carlo]
- [**Examples of Renormalization Group Transformations for Image
  Sets**][Examples of renormalization group transformations for image sets]

#### 🎤 Selected Talks[^2]

- [**AuroraGPT: Foundation Models for Science**][9] @ [*Foundation
  Models for the Electric Grid*][] \[02/2025\]
- [**Parallel Training Methods**][10] @ [*AI-for-Science on
  Supercomputers*][*Foundation Models for the Electric Grid*]
  \[11/2024\]
- [**AuroraGPT**][11] @ [*HPC User Forum*, 2024][] \[09/2024\]
- [**Machine Learning and Foundation Models at Scale**][12] @ [*2024
  ALCF Hands-On HPC Workshop*][] \[10/2024\]
- [**Training LLMs at Scale**][13] @ [*ATPESC*, 2024][] \[08/2024\]
- [**LLMs from Scratch**] @ [LLM Tutorial Workshop][] \[02/2024\]
- [**Exascale Science on Aurora**] @ [Intel oneAPI Workshop @ UIC][]
  \[10/2023\]
- [**Scaling LLMs for Science**] @ [Data-Intensive Computing + AI/ML at
  Scale][] \[08/2023\]
- [**MLMC: Machine Learning Monte Carlo**] @ [Lattice 2023][]
  \[07/2023\]
- [**Generative Modeling and Efficient Sampling**] @ [PASC23][]
  \[07/2023\]

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

| Position                                  |     @     | Start | End  |
|:------------------------------------------|:---------:|:-----:|:----:|
| [Assistant Computational Scientist][ALCF] | [ALCF][1] | 2022  |  –   |
| Postdoc                                   |   ALCF    | 2019  | 2022 |
| Graduate Researcher                       |   [ANL]   | 2018  | 2019 |

📟 Experience

</div>

#### 🍎 School

<div id="tbl-education">

Table 2: 🎓 Education

| Degree |      In       |         @          | End  |
|:-------|:-------------:|:------------------:|:----:|
| [PhD]  |   [Physics]   | University of Iowa | 2019 |
| B.Sc   | [Physics][14] |       [UIUC]       | 2015 |
| B.Sc   |    [Math]     |        UIUC        | 2015 |

🎓 Education

</div>

### 🎶 Music

<div class="container"
style="display: grid; text-align:center; gap: 10px; grid-template-columns: repeat(2, minmax(120px, 1fr)); grid-template-rows: masonry;">

<a href="https://github.com/kittinan/spotify-github-profile"><img loading="lazy" src="https://spotify-github-profile.kittinanx.com/api/view?uid=saforem2&cover_image=true&loading=lazy&theme=default&show_offline=false&background_color=1c1c1c&interchange=false" /></a>

<a href="https://last.fm/user/saforem2"><img loading="lazy" src="https://lastfm-recently-played.vercel.app/api?user=saforem2" align="center" /></a>

<iframe loading="lazy" width="auto" src="https://descent.live/saforem2" style="width: 100%; border: none; height: min(800px, calc(0.8*100vh)); border-radius: 4pt;">

</iframe>

<a href="https://music-profile.rayriffy.com"><img loading="lazy" src="https://music-profile.rayriffy.com/theme/dark.svg?uid=002028.5a338f21979147c78f6193b6138a1ec7.1532" align="center" /></a>

</div>

</div>

[^1]: *See full list on [Google Scholar][15]*

[^2]: *See full list at: [samforeman.me/talks]*

  [ALCF]: https://alcf.anl.gov/about/people/sam-foreman
  [<span class="orcid-green" style="background: none!important;"></span>]:
    https://orcid.org/0000-0002-9981-0876
  [Group]: https://www.alcf.anl.gov/about/people/group/506
  [1]: https://alcf.anl.gov
  [science]: https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=vV_1zDwAAAAJ
  [training large models]: https://samforeman.me/talks/AuroraGPT/alcf-hpc-workshop-2024/slides.html
  [supercomputers]: https://www.alcf.anl.gov/aurora
  [here]: talks/index.qmd
  [make them]: ./posts/dope-slides/index.qmd
  [AI + Science]: https://github.com/saforem2/
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
  [2]: ./assets/neovim-headings.png
  [Sam Foreman]: https://samforeman.me
  [Google Scholar]: https://scholar.google.com/citations?user=vV_1zDwAAAAJ&hl=en
  [**MProt-DPO: Breaking the ExaFLOPS Barrier for Multimodal Protein Design Workflows with Direct Preference Optimization**]:
    https://doi.org/10.1109/SC41406.2024.00013
  [SC’24]: https://sc24.supercomputing.org/
  [<span class="highlight-pink">2024 Gordon Bell Finalist</span>]: https://sc24.supercomputing.org/2024/10/presenting-the-finalists-for-the-2024-gordon-bell-prize/
  [Intro to HPC Bootcamp: Engaging New Communities Through Energy Justice Projects]:
    https://jocse.org/downloads/jocse-15-1-10.pdf
  [Thorough Characterization and Analysis of Large Transformer Model Training At-Scale]:
    https://doi.org/10.1145/3639034
  [MLMC: Machine Learning Monte Carlo for Lattice Gauge Theory]: https://arxiv.org/abs/2312.08936
  [Lattice, 2023 (Proceedings)]: https://indico.fnal.gov/event/57249/
  [Protein Generation via Genome-scale Language Models with Bio-physical Scoring]:
    https://dl.acm.org/doi/abs/10.1145/3624062.3626087
  [3]: https://arxiv.org/abs/2310.04610
  [NeurIPS 2023 AI For Science Workshop]: https://ai4sciencecommunity.github.io/neurips23.html
  [DeepSpeed4Science.ai Blog Post]: https://deepspeed4science.ai/2023/09/18/model-showcase-genslms/
  [Loooooooong Sequence Lengths]: ./posts/AuroraGPT/long-sequences/index.qmd
  [Comprehensive Performance Study of LLMs on Novel AI Accelerators]: https://arxiv.org/abs/2310.04607
  [IPDPS 2024]: https://www.ipdps.org/
  [Intro to HPC Bootcamp @ NERSC]: https://github.com/NERSC/intro-HPC-bootcamp-2023
  [**GenSLMs: Genome-scale language models reveal SARS-Cov-2 evolutionary dynamics**]:
    https://www.biorxiv.org/content/10.1101/2022.10.10.511571v1.abstract
  [<span class="highlight-pink">ACM Gordon Bell Special Prize</span>]: https://www.acm.org/media-center/2023/november/gordon-bell-special-prize-covid-research-2022
  [Lattice QCD and Particle Physics]: https://arxiv.org/abs/2207.07641
  [Applications of ML to Lattice QFT]: https://arxiv.org/abs/2202.05838
  [LeapFrogLayers: Trainable Framework for Effective Sampling]: https://arxiv.org/abs/2112.01582
  [Lattice, *2021*]: https://indico.cern.ch/event/1006302
  [HMC with Normalizing Flows]: https://arxiv.org/abs/2112.01586
  [slides]: https://indico.cern.ch/event/1006302/contributions/4380743/
  [4]: https://indico.cern.ch/event/1006302/
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
  [**LLMs on Aurora: 🌌 AuroraGPT**]: ./talks/incite-hackathon-2025/AuroraGPT/index.html
  [*2025 ALCF INCITE GPU Hackathon*]: https://www.alcf.anl.gov/events/alcf-incite-gpu-hackathon
  [**LLMs on Aurora: 🍋 ezpz**]: ./talks/incite-hackathon-2025/ezpz/index.html
  [**AuroraGPT: Foundation Models for Science**]: ./talks/aurora-gpt-fm-for-electric-grid/index.html
  [*Foundation Models for the Electric Grid*]: https://www.alcf.anl.gov/alcf-ai-science-training-series
  [**Parallel Training Methods**]: ./talks/ai-for-science-2024/index.html
  [**AuroraGPT**]: ./talks/AuroraGPT/alcf-hpc-workshop-2024/index.html
  [*2024 ALCF Hands-On HPC Workshop*]: https://www.alcf.anl.gov/events/2024-alcf-hands-hpc-workshop
  [**Machine Learning and Foundation Models at Scale**]: ./talks/alcf-hpc-workshop-2024/index.html
  [5]: ./talks/hpc-user-forum/index.html
  [*HPC User Forum*, 2024]: https://www.hpcuserforum.com/hpc-user-forum-fall-2024/
  [**Training LLMs at Scale**]: ./talks/llms-at-scale/
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
  [6]: https://indico.bnl.gov/event/13576/
  [**Training Topological Samplers for LGT**]: https://saforem2.github.io/l2hmc_talk_ect2021
  [ML4HEP, ECT\* Trento]: https://indico.ectstar.eu/event/77/contributions/2349/
  [**Deep Learning HMC for Improved Gauge Generation**]: https://bit.ly/mainz21
  [ML in LQCD Workshop]: https://bit.ly/mainz21_overview
  [**Machine Learning for Lattice QCD**]: https://slides.com/samforeman/l2hmc-qcd/embed
  [`saforem2/`]: https://github.com/saforem2?tab=repositories
  [*Learning Better Physics: A Machine Learning Approach to Lattice Gauge Theory*]:
    https://www.proquest.com/openview/95d7f7c12da8da8aa5ead3ac0f6ca0e8/1?cbl=18750&diss=y&pq-origsite=gscholar
  [Energy Storage in Quantum Resonators (US Patent \#US9741492B2)]: https://patents.google.com/patent/US9741492B2/en
  [7]: https://www.researchgate.net/publication/387390653_MProt-DPO_Breaking_the_ExaFLOPS_Barrier_for_Multimodal_Protein_Design_Workflows_with_Direct_Preference_Optimization
  [8]: https://doi.org/10.1177/10943420231184990
  [9]: https://samforeman.me/talks/aurora-gpt-fm-for-electric-grid/
  [10]: https://samforeman.me/talks/ai-for-science-2024/
  [11]: https://samforeman.me/talks/hpc-user-forum/
  [12]: https://samforeman.me/talks/alcf-hpc-workshop-2024/
  [13]: https://samforeman.me/talks/llms-at-scale/
  [SC24 Workshop: High Performance Python for Science at Scale (HPPSS)]:
    https://hppss.github.io/SC24/
  [SC23 Workshop: High Performance Python for Science at Scale (HPPSS)]:
    https://hppss.github.io/SC23/
  [Machine Learning and Quantum Computing for Earth Sciences]: https://17.usnccm.org/702
  [ANL]: https://anl.gov
  [PhD]: https://bit.ly/sam-foreman-phd
  [Physics]: https://physics.uiowa.edu/graduate/phd-physics
  [14]: https://grainger.illinois.edu/academics/undergraduate/majors-and-minors/physics
  [UIUC]: https://illinois.edu/
  [Math]: https://math.illinois.edu/
  [15]: https://scholar.google.com/citations?user=7vBs2ZwAAAAJ
  [samforeman.me/talks]: https://samforeman.me/talks/
