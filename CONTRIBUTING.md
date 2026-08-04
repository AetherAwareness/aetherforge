# Contributing to AetherForge

Thanks for helping improve open MoE post-training tooling.

**License:** contributions are accepted under the same **PolyForm Noncommercial License 1.0.0** as the project (© AetherAwareness). By opening a PR you agree your contribution is licensed under those terms. Commercial redistribution still requires a separate grant — see [COMMERCIAL.md](COMMERCIAL.md).

## Development setup

```bash
git clone https://github.com/AetherAwareness/aetherforge.git
cd aetherforge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

## Guidelines

1. **Industry-agnostic core** — no hard-coded medical/finance/legal content in Python modules. Put domain material in YAML packs under `configs/domains/` or `packs/`.
2. **Tests** — add unit tests for new selection/forensics/config behavior; keep dry-run pipeline green.
3. **Docs** — user-facing features need a short note under `docs/` (and README link if major).
4. **Spend safety** — remote paths must not train without explicit `--exec` / confirmation.
5. **Style** — match existing modules; prefer small, focused PRs.

## Docs site (GitHub Pages)

```bash
cd docs
bundle install
bundle exec jekyll serve
# http://127.0.0.1:4000/aetherforge/
```

Or rely on `.github/workflows/pages.yml` after enabling Pages in repo settings.

## PR checklist

- [ ] `pytest tests/ -q` passes  
- [ ] New CLI flags documented in `docs/reference/cli.md`  
- [ ] Config keys in `docs/reference/config.md` if applicable  
- [ ] No secrets committed (`~/.aetherforge` is local only)  

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
