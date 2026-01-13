# 🍋 <code>ezpz</code>
Sam Foreman
2026-01-10

- [AMD Timeline](#amd-timeline)
- [Intel Timeline](#intel-timeline)

In ancient times[^1], back in ~ 2022–2023, virtually all (production)
PyTorch code was designed to run on NVIDIA GPUs.

In April 2023, AMD announced day-zero support for PyTorch 2.0 within the
ROCm 6.0 ecosystem, leveraging new features like TorchDynamo for
performance

``` mermaid
gantt
    title AMD and Intel PyTorch Enablement Timeline
    dateFormat  YYYY
    axisFormat  %Y
    %% ROCm                                             :vert, amd1, 2021-03-04, 1d
    %% Installable PyTorch ROCm Python packages         :milestone, amd2, 2021-03-04, 1d

    section AMD ROCm and PyTorch
      Torch7 era and early CUDA to HIP ports        :milestone,amd1, 2012, 2016
      ROCm 1.0 and HIPIFY tooling                   :milestone,amd2, 2016, 2020
      Official PyTorch ROCm Python packages         :milestone,amd3, 2021, 2022
      PyTorch Foundation governance participation   :milestone,amd4, 2022, 2023
      Triton ecosystem support                      :milestone,amd6, 2023, 2024
      MI300x PyTorch guidance                       :milestone,amd7, 2024, 2024
      %% PyTorch 2.0 day zero ROCm support             :milestone,amd5, 2023, 2023
      %% Torchtune on AMD GPUs guide                   :milestone,amd8, 2024, 2024
      %% PyTorch on Windows public preview             :milestone,amd9, 2025, 2025
      %% AMD PyTorch on Windows ROCm 7.1.1             :milestone,amd10, 2025, 2025
      %% MI450X rack scale roadmap                     :milestone,amd11, 2026, 2026
      %% MI500 series future roadmap                   :milestone,amd12, 2027, 2028

    section Intel and PyTorch
      Initial PyTorch contributions                :i2,2018, 2019
      Intel Extension for PyTorch launch           :i3,2020, 2024
      VTune ITT API integration in PyTorch         :i4,2022, 2022
      PyTorch Foundation Premier membership        :i5,2023, 2023
      Prototype native Intel GPU support           :i6,2024, 2024
      Solid native Intel GPU support               :i7,2025, 2025
      IPEX feature upstreaming completion          :i8,2025, 2025
      Intel Extension for PyTorch end of life      :i9,2026, 2026

```

``` mermaid
gantt
    title PyTorch Vendor Integration Timeline AMD vs Intel
    dateFormat  YYYY-MM-DD
    axisFormat  %Y


section PyTorch Releases
    1.8                                             :crit,milestone, pt180, 2021-03-04, 1d
    1.12                                            :done,milestone, pt1120, 2022-06-28, 1d
    2.0                                             :crit,milestone, pt200, 2023-03-15, 1d
    2.4                                             :done,milestone,pt24, 2024-07-24, 1d
    2.5                                             :crit,milestone, pt250, 2024-10-17, 1d
    2.6                                             :milestone,done,2025-01-29, 1d
    2.7                                             :done,milestone, pt270, 2025-04-23, 1d
    2.8                                             :crit,milestone, pt280, 2025-08-06, 1d
    2.9                                            :done,milestone, pt290, 2025-10-15, 1d
    2.10                                           :milestone, pt210, 2026-01-15, 1d

section AMD
    ROCm                                             :vert, amd1, 2021-03-04, 1d
    Installable PyTorch ROCm Python packages         :milestone, amd2, 2021-03-04, 1d
    %% ROCm development                                 :amd25, 2021-03-04, 2022-06-28,
    ROCm marked stable                               :vert,done,amd3,2022-06-28,
    %% PyTorch 1.12 ROCm marked stable                  :done,milestone,amd3 2022-06-28, 1d
    %% PyTorch joins Linux Foundation (AMD board)       :done, amd4, 2022-09-01, 2022-12-31
    %% Torch 2.0 TorchInductor MI250 Performance        :vert,amd5,2023-03-15, 1d
    %% PyTorch 2.0 day zero ROCm support                :done, amd6, 2023-04-01, 2023-12-31
    %% Triton kernel ecosystem support                  :done, amd7, 2023-06-01, 2024-03-31
    %% MI300x PyTorch deployment guides                 :done, amd8, 2024-06-01, 2024-08-31
    %% Torchtune LLM fine tuning on AMD GPUs             :done, amd9, 2024-10-01, 2024-10-31
    %% Expanded ROCm wheel variants in PyTorch 2.8 2.9    :milestone, amd10, 2025-10-15, 1d

section Intel
    %% Intel Extension for PyTorch launched               :milestone, int1, 2020-01-01, 1d
    Incremental Intel GPU improvements begin           :milestone, int2, 2024-07-24, 1d
    Native Intel GPU support announced in PyTorch 2.5  :milestone, int3, 2024-10-17, 1d
    XPU                                                :vert, 2024-10-17, 1d
    Intel GPU eager and compile parity in PyTorch 2.7  :milestone, int4, 2025-04-23, 1d
    XCCL Backend                                       :vert,done, 2025-08-06,
    IPEX discontinued                                  :int5, 2025-08-06, 2026-03-31
    IPEX end of life                                   :milestone, int6, 2026-03-31, 1d

```

> Intel: Mar 2026 (planned): IPEX end-of-lifemove to native PyTorch

    %% --- Forward-looking roadmap (AMD) ---
    AMD-->>PT: 2026: MI450X rack-scale target (2H 2026 competitiveness)
    AMD-->>PT: Post-2026: MI500 series plans (major AI perf increase)



    ``` mermaid
    gantt
        title AMD and Intel PyTorch Enablement Timeline
        dateFormat  YYYY
        axisFormat  %Y
        %% ROCm                                             :vert, amd1, 2021-03-04, 1d
        %% Installable PyTorch ROCm Python packages         :milestone, amd2, 2021-03-04, 1d

        section AMD ROCm and PyTorch
          Torch7 era and early CUDA to HIP ports        :milestone,amd1, 2012, 2016
          ROCm 1.0 and HIPIFY tooling                   :milestone,amd2, 2016, 2020
          Official PyTorch ROCm Python packages         :milestone,amd3, 2021, 2022
          PyTorch Foundation governance participation   :milestone,amd4, 2022, 2023
          Triton ecosystem support                      :milestone,amd6, 2023, 2024
          MI300x PyTorch guidance                       :milestone,amd7, 2024, 2024
          %% PyTorch 2.0 day zero ROCm support             :milestone,amd5, 2023, 2023
          %% Torchtune on AMD GPUs guide                   :milestone,amd8, 2024, 2024
          %% PyTorch on Windows public preview             :milestone,amd9, 2025, 2025
          %% AMD PyTorch on Windows ROCm 7.1.1             :milestone,amd10, 2025, 2025
          %% MI450X rack scale roadmap                     :milestone,amd11, 2026, 2026
          %% MI500 series future roadmap                   :milestone,amd12, 2027, 2028

        section Intel and PyTorch
          Initial PyTorch contributions                :i2,2018, 2019
          Intel Extension for PyTorch launch           :i3,2020, 2024
          VTune ITT API integration in PyTorch         :i4,2022, 2022
          PyTorch Foundation Premier membership        :i5,2023, 2023
          Prototype native Intel GPU support           :i6,2024, 2024
          Solid native Intel GPU support               :i7,2025, 2025
          IPEX feature upstreaming completion          :i8,2025, 2025
          Intel Extension for PyTorch end of life      :i9,2026, 2026

``` mermaid
gantt
    title AMD and Intel PyTorch Enablement Timeline
    dateFormat  YYYY
    axisFormat  %Y

    section amd AMD
      Torch7 era and early CUDA to HIP ports        :2012, 2016
      ROCm 1.0 and HIPIFY tooling                   :2016, 2020
      Official PyTorch ROCm Python packages         :2021, 2022
      PyTorch Foundation governance participation   :2022, 2023
      ROCm                                          :vert, 2023, 2023
      PyTorch 2.0 day zero ROCm support             :milestone,crit, 2023, 2023
      Triton ecosystem support                      :2023, 2024
      MI300x PyTorch guidance                       :2024, 2024
      Torchtune on AMD GPUs guide                   :2024, 2024
      PyTorch on Windows public preview             :2025, 2025
      AMD PyTorch on Windows ROCm 7.1.1             :2025, 2025
      MI450X rack scale roadmap                     :2026, 2026
      MI500 series future roadmap                   :2027, 2028

    section intel Intel
      Initial PyTorch contributions                :2018, 2019
      Intel Extension for PyTorch launch           :2020, 2024
      VTune ITT API integration in PyTorch         :2022, 2022
      PyTorch Foundation Premier membership        :2023, 2023
      Prototype native Intel GPU support           :2024, 2025
      Solid native Intel GPU support               :milestone,crit, 2025, 2025
      X{PU,CCL}                                    :vert, 2025, 2025
      IPEX feature upstreaming completion          :2025, 2025
      Intel Extension for PyTorch end of life      :2026, 2026
```

–\>

## AMD Timeline

- Pre-2021: Early Efforts and Torch7
  - 2012: Torch7 was released, a precursor to PyTorch, written in C++
    and CUDA.
  - ROCm 1.0: AMD demonstrated the ability to port CUDA code to HIP
    (AMD’s C++ dialect for GPU computing) using the HIPIFY tool,
    including ports of Caffe and Torch7.
- 2021-2022: Official Support and Foundation
  - March 2021: PyTorch for the AMD ROCm platform became officially
    available as a Python package, simplifying installation on supported
    Linux systems.
  - September 2022: The PyTorch project joined the independent Linux
    Foundation, with AMD participating as a founding member of the
    PyTorch Foundation governing board.
- 2023: PyTorch 2.0 Integration
  - April 2023: AMD announced day-zero support for PyTorch 2.0 within
    the ROCm 6.0 ecosystem, leveraging new features like TorchDynamo for
    performance improvements.
  - OpenAI Triton Support: The ecosystem grew to include support for
    OpenAI Triton, a key component for high-performance AI workloads.
- 2024-2025: Expanding Accessibility (Windows & Consumer GPUs)
  - June 2024: AMD released guides and information on running PyTorch
    models on AMD MI300x systems, highlighting near drop-in
    compatibility with code written for Nvidia GPUs.
  - September 2025: AMD released a public preview of PyTorch on Windows,
    enabling native AI inference on select consumer Radeon RX 7000 and
    9000 series GPUs and Ryzen AI APUs, without needing workarounds like
    WSL2.
  - October 2024: AMD released a “how-to” guide for using Torchtune, a
    PyTorch library for fine-tuning LLMs, on AMD GPUs.
  - November 2025: Release of AMD Software: PyTorch on Windows Edition
    7.1.1, featuring an update to AMD ROCm 7.1.1.
- Future/Upcoming
  - 2026: AMD is working on its next generation MI450X rack-scale
    solution, which aims to be competitive with NVIDIA’s high-end
    offerings by the second half of 2026.
  - Post-2026: The company has also detailed plans for future MI500
    series data center GPUs, targeting a significant increase in AI
    performance

## Intel Timeline

- 2018: Intel begins contributing to the open-source PyTorch framework.
- 2020: The Intel® Extension for PyTorch\* (IPEX) is launched as a
  separate package to provide optimized performance on Intel CPUs and
  GPUs.
- October 2022[^2]: PyTorch 1.13 is released with integrated support for
  Intel® VTune™ Profiler’s ITT APIs.
- August 2023[^3]: Intel joins the PyTorch Foundation as a Premier
  member, deepening its commitment to the ecosystem.
- July 2024: PyTorch 2.4 debuts with initial (prototype) native support
  for Intel GPUs (client and data center).
- April 2025: PyTorch 2.7 establishes a solid foundation for Intel GPU
  support in both eager and graph modes (torch.compile) on Windows and
  Linux.
- August 2025: Active development of the separate Intel® Extension for
  PyTorch\* ceases following the PyTorch 2.8 release, as most features
  are now upstreamed into the main PyTorch project.
- End of March 2026 (Planned): The Intel® Extension for PyTorch\*
  project will officially reach end-of-life. Users are strongly
  recommended to use native PyTorch directly.

This made sense at the time, as NVIDIA had the vast majority of the GPU
market share and was the only major GPU manufacturer.

This was before the advent of

we were still in the early days of trying to run PyTorch on

I’ve been working on the 🍋 [`ezpz`](https://ezpz.cool) package for a
while now,

[^1]: Even now, in 2026, a lot of code is still NVIDIA-centric and is
    rarely designed with multi-platform support in mind.

[^2]: [PyTorch 1.13
    release](https://pytorch.org/blog/pytorch-1-13-release/)

[^3]: [Intel Joins the PyTorch
    Foundation](https://www.edge-ai-vision.com/2023/08/driving-pytorch-and-ai-everywhere-intel-joins-the-pytorch-foundation/)
