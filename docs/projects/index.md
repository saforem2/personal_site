# 📚 Projects
Sam Foreman

> [!TIP]
>
> ### <span class="dim-text-11">📂 [`saforem2/`](https://github.com/saforem2?tab=repositories)</span>
>
> <style>
>   .repo-grid {
>     display: flex;
>     flex-wrap: wrap;
>     gap: 1rem;
>     justify-content: flex-start;
>     align-items: stretch;
>     margin-top: 1.5rem;
>   }
> &#10;  .repo-card {
>     flex: 1 1 260px; /* responsive: min width ~260px */
>     max-width: 360px;
>     border-radius: 0pt;
>     padding: 1rem 1.1rem;
>     border: 1px solid var(--bs-border-color, #dee2e6);
>     background-color: var(--bs-body-bg, #ffffff);
>     box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
>     display: flex;
>     flex-direction: column;
>     transition: transform 0.15s ease, box-shadow 0.15s ease,
>                 border-color 0.15s ease, background-color 0.15s ease;
>   }
> &#10;  .repo-card:hover {
>     transform: translateY(-3px);
>     box-shadow: 3pt 3pt black;
>     /* border-color: var(--bs-primary, #0d6efd); */
>   }
> &#10;  .repo-header {
>     display: flex;
>     align-items: center;
>     justify-content: space-between;
>     gap: 0.4rem;
>     margin-bottom: 0.35rem;
>   }
> &#10;  .repo-name {
>     color: inherit;
>     background: none;
>     font-size: 1.0rem;
>     font-weight: 600;
>     margin: 0;
>     word-break: break-word;
>     border: 1pt solid rgba(0,0,0,0.0) !important;
>     padding-inline: 1pt;
> }
> &#10;  .repo-name a {
>     text-decoration: none;
>     color: var(--bs-link-color, #0d6efd);
>   }
> &#10;  .repo-name a:hover {
>     text-decoration: underline;
>     border: none;
>   }
> &#10;  .repo-badge {
>     font-size: 0.7rem;
>     padding: 0.15rem 0.45rem;
>     /* border-radius: 999px; */
>     border: 1px solid var(--bs-border-color, #838383);
>     background: var(--bs-secondary-bg, rgba(108, 117, 125, 0.1));
>     white-space: nowrap;
>   }
> &#10;  .repo-description {
>     font-size: 0.9rem;
>     margin: 0.3rem 0 0.6rem;
>     color: var(--bs-secondary-color, #6c757d);
>     min-height: 2.2em; /* keep rows roughly aligned */
>   }
> &#10;  .repo-meta {
>     display: flex;
>     flex-wrap: wrap;
>     align-items: center;
>     gap: 0.6rem;
>     margin-top: auto;
>     font-size: 0.8rem;
>     color: var(--bs-secondary-color, #6c757d);
>   }
> &#10;  .repo-meta span {
>     display: inline-flex;
>     align-items: center;
>     gap: 0.2rem;
>   }
> &#10;  .repo-language-dot {
>     width: 0.55rem;
>     height: 0.55rem;
>     /* border-radius: 999px; */
>     background: currentColor;
>   }
> &#10;  /* Dark mode tweaks when using Quarto's theme toggle */
>   body[data-mode="dark"] .repo-card {
>     background-color: var(--bs-body-bg, #111827);
>     border-color: var(--bs-border-color, #374151);
>     box-shadow: 0 6px 18px rgba(0, 0, 0, 0.6);
>   }
> &#10;  body[data-mode="dark"] .repo-badge {
>     background: var(--bs-secondary-bg, #1f2933);
>   }
> &#10;  .repo-grid-loading {
>     font-size: 0.9rem;
>     color: var(--bs-secondary-color, #6c757d);
>     margin-top: 0.75rem;
>   }
> </style>
>
> <div class="repo-grid-container">
>
> <h2 class="anchored">
>
> <a href="https://github.com/saforem2/"><code>saforem2</code></a>s
> GitHub Repositories
> </h2>
>
> <p class="repo-grid-loading" id="repo-grid-status">
>
> Loading repositories from GitHub…
> </p>
>
> <div id="repo-grid" class="repo-grid" data-github-user="saforem2">
>
> </div>
>
> </div>
>
> <script>
>   (async function () {
>     const grid = document.getElementById("repo-grid");
>     const status = document.getElementById("repo-grid-status");
>     if (!grid) return;
> &#10;    const username = grid.getAttribute("data-github-user") || "saforem2";
>     const apiUrl = `https://api.github.com/users/${username}/repos?per_page=100&sort=updated`;
> &#10;    try {
>       const res = await fetch(apiUrl);
>       if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
>       const repos = await res.json();
> &#10;      // Filter out forks if you don't want them:
>       const filtered = repos.filter(r => !r.fork);
> &#10;      if (filtered.length === 0) {
>         if (status) status.textContent = "No repositories found.";
>         return;
>       }
>       if (status) status.remove();
> &#10;      // Small helper: nice abbreviations for numbers
>       function formatNumber(n) {
>         if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
>         if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, "") + "k";
>         return String(n);
>       }
> &#10;      filtered.forEach(repo => {
>         const card = document.createElement("div");
>         card.className = "repo-card";
> &#10;        const header = document.createElement("div");
>         header.className = "repo-header";
> &#10;        const title = document.createElement("h3");
>         title.className = "repo-name";
>         const link = document.createElement("a");
>         link.href = repo.html_url;
>         link.textContent = repo.name;
>         link.target = "_blank";
>         link.rel = "noopener noreferrer";
>         title.appendChild(link);
> &#10;        const badge = document.createElement("span");
>         badge.className = "repo-badge";
>         badge.textContent = repo.private ? "Private" : "Public";
> &#10;        header.appendChild(title);
>         header.appendChild(badge);
> &#10;        const desc = document.createElement("p");
>         desc.className = "repo-description";
>         desc.textContent = repo.description || "No description provided.";
> &#10;        const meta = document.createElement("div");
>         meta.className = "repo-meta";
> &#10;        if (repo.language) {
>           const langSpan = document.createElement("span");
>           const dot = document.createElement("span");
>           dot.className = "repo-language-dot";
>           langSpan.appendChild(dot);
>           const langText = document.createElement("span");
>           langText.textContent = repo.language;
>           langSpan.appendChild(langText);
>           meta.appendChild(langSpan);
>         }
> &#10;        const starsSpan = document.createElement("span");
>         starsSpan.innerHTML = "⭐️ " + formatNumber(repo.stargazers_count || 0);
>         meta.appendChild(starsSpan);
> &#10;        const updatedSpan = document.createElement("span");
>         const updatedDate = new Date(repo.updated_at);
>         updatedSpan.textContent = "Updated " + updatedDate.toLocaleDateString(undefined, {
>           year: "numeric",
>           month: "short",
>           day: "numeric"
>         });
>         meta.appendChild(updatedSpan);
> &#10;        card.appendChild(header);
>         card.appendChild(desc);
>         card.appendChild(meta);
>         grid.appendChild(card);
>       });
>     } catch (err) {
>       console.error(err);
>       if (status) {
>         status.textContent = "Failed to load repositories from GitHub.";
>       }
>     }
>   })();
> </script>
