# 📊 `pbs-tui`: TUI for PBS Job Scheduler Monitoring
Sam Foreman
2025-09-17

- [👀 Overview](#eyes-overview)
- [🐣 Getting Started](#hatching_chick-getting-started)
- [✨ Features](#sparkles-features)
  - [🎹 Key bindings](#musical_keyboard-key-bindings)
  - [🧪 Sample mode](#test_tube-sample-mode)
  - [Headless / automated runs](#headless--automated-runs)
  - [Inline snapshot mode](#inline-snapshot-mode)
- [Architecture](#architecture)
- [Development notes](#development-notes)
- [Screenshots](#screenshots)

<div style="text-align: center;">

`pbs-tui`

</div>

<div id="fig-pbs-tui">

<div class="dark-content">

![pbs-tui](https://github.com/user-attachments/assets/49bbdd98-9670-4064-bea1-948dda949b64.png)

</div>

<div class="light-content">

![pbs-tui](https://github.com/user-attachments/assets/4f1c4503-2478-436e-91da-13c421da7523.png)

</div>

Figure 1: A terminal dashboard for monitoring PBS Pro schedulers

</div>

## 👀 Overview

A terminal user interface built with
[Textual](https://textual.textualize.io/) for monitoring [PBS
Pro](https://altair.com/pbs-professional) schedulers at the [Argonne
Leadership Computing Facility](https://alcf.anl.gov).

The dashboard surfaces job, queue, and node activity in a single view
and refreshes itself automatically so operators can track workload
health in real time.

## 🐣 Getting Started

- Try it with uv:

  ``` bash
  # install uv if necessary
  # curl -LsSf https://astral.sh/uv/install.sh | sh
  uv run --with pbs-tui pbs-tui
  ```

- Or install and run:

  ``` bash
  python3 -m pip install pbs-tui
  pbs-tui
  ```

## ✨ Features

- **Live PBS data** – prefers the JSON (`-F json`) output of
  `qstat`/`pbsnodes` and falls back to XML or text parsing so schedulers
  without newer flags continue to work.

  - **Automatic refresh** – updates every 30 seconds by default with a
    manual refresh binding (`r`).
  - **Summary cards** – quick totals for job states, node states, and
    queue health.

- **Inline snapshot** – render the current queue as a Rich table with
  `pbs-tui --inline`

  - **Save to file** – write the snapshot to a Markdown file with
    `pbs-tui --inline --file snapshot.md`

- **Fallback sample data** – optional bundled data makes it easy to demo
  the interface without connecting to a production scheduler
  (`PBS_TUI_SAMPLE_DATA=1`).

### 🎹 Key bindings

<div id="tbl-keys">

Table 1: Use the arrow keys/`PageUp`/`PageDown` to move through rows
once a table has focus.

|  Key  | Action                   |
|:-----:|:-------------------------|
|  `q`  | Quit the application     |
|  `r`  | Refresh immediately      |
|  `j`  | Focus the jobs table     |
|  `n`  | Focus the nodes table    |
|  `u`  | Focus the queues table   |
| `^-p` | Open the command palette |

</div>

### 🧪 Sample mode

If you want to explore the UI without a live PBS cluster, export
`PBS_TUI_SAMPLE_DATA=1` (or pass `force_sample=True` to
`PBSDataFetcher`). The application will display bundled example jobs,
nodes, and queues along with a warning banner indicating that the data
is synthetic.

### Headless / automated runs

For automated testing or CI environments without an interactive terminal
you can run the TUI in headless mode by exporting `PBS_TUI_HEADLESS=1`.
Pairing this with `PBS_TUI_AUTOPILOT=quit` presses the `q` binding
automatically after startup so `pbs-tui` exits cleanly once the
interface has rendered its first update.

### Inline snapshot mode

When running non-interactively you can emit a Rich-rendered table
summarising the active PBS jobs instead of starting the Textual
interface:

``` bash
PBS_TUI_SAMPLE_DATA=1 pbs-tui --inline
```

The command prints a table that can be pasted into terminals that
support Unicode box drawing. Pass `--file snapshot.md` alongside
`--inline` to also write an aligned Markdown table to `snapshot.md` for
sharing in chat or documentation systems. Any warnings raised while
collecting data are written to standard error so they remain visible in
logs.

## Architecture

- `pbs_tui.fetcher.PBSDataFetcher` orchestrates `qstat`/`pbsnodes`
  calls, preferring JSON output and falling back to XML/text before
  converting everything into structured dataclasses (`Job`, `Node`,
  `Queue`).
- `pbs_tui.app.PBSTUI` is the Textual application that renders the
  dashboard, periodically asks the fetcher for new data, and updates the
  widgets.
- `pbs_tui.samples.sample_snapshot` provides the demonstration snapshot
  used when PBS commands cannot be executed.

The UI styles are defined in `pbs_tui/app.tcss`. Adjust the CSS to
change layout or theme attributes.

## Development notes

- The application refresh interval defaults to 30 seconds. Pass a
  different value to `PBSTUI(refresh_interval=...)` if desired.
- Errors encountered while running PBS commands are surfaced in the
  status bar so operators can quickly see when data is stale.
- When both PBS utilities are unavailable and the fallback is disabled,
  the UI will show an empty dashboard with an error message in the
  status bar.

## Screenshots

- `pbs-tui`:

  <img loading="lazy" width="95%" alt="<code>pbs-tui</code>”
  src=“https://github.com/user-attachments/assets/419cecb6-25a1-4007-8456-38bd80fb4ae7”
  /\>

- Keys and Help Panel:

  <img loading="lazy" width="95%" alt="Keys and Help Panel" src="https://github.com/user-attachments/assets/d521d137-1135-4503-bcc0-2b9dba35d252" />

- Command palette:

  <img loading="lazy" width="95%" alt="Command palette" src="https://github.com/user-attachments/assets/5804c99a-621a-4cce-adde-092f6d324824" />

- theme support:

  <img loading="lazy" width="95%" alt="theme support" src="https://github.com/user-attachments/assets/d4009439-2ea7-49f5-9c75-5d25f7b13771" />
