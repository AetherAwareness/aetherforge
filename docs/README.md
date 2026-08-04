# AetherForge documentation (GitHub Pages)

This folder is the **documentation site** for AetherForge.

## Enable on GitHub

1. Push to `main` (or `master`).
2. **Settings → Pages → Build and deployment → Source: GitHub Actions**
3. Workflow: [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)
4. Search-replace `AetherAwareness` in:
   - `docs/_config.yml` (`url`, `baseurl`, `aux_links`, `gh_edit_repository`)
   - Root `README.md` badges and links
   - `HF_README.md`

`baseurl` should be `/aetherforge` for `https://AetherAwareness.github.io/aetherforge/`  
(or empty if using a custom domain / user site).

## Local preview

```bash
cd docs
bundle install
bundle exec jekyll serve
# open http://127.0.0.1:4000/aetherforge/
```

## Site map

| Path | Page |
|------|------|
| `index.md` | Home |
| `getting-started.md` | Install & first run |
| `architecture.md` | Pipeline & MoE |
| `guides/*` | Studio, Flash, postures, hardware, Vast |
| `reference/*` | CLI, config, recipes |
| `safety.md` | Safety & privacy |
| `changelog.md` | Versions |
| `faq.md` | FAQ |
| `demo/*` | Screenshots & gif |

Long-form internal notes also live here (`FLASH_0731_TRAIN.md`, `BROAD_WORK.md`) and are linked from guides.
