# Domain packs

Industry content for AetherForge lives **here** (or in `configs/domains/*.yaml`) — never inside the Python trainer core.

## Minimal pack (YAML)

```yaml
domain: energy_markets
description: Day-ahead bidding and grid congestion risk
topics:
  - nodal price spike after transmission outage
  - renewable forecast error on imbalance cost
keywords:
  - nodal
  - congestion
  - imbalance
  - forecast
actions:
  - Recompute PTDF-sensitive positions for affected nodes.
  - Bound imbalance exposure under forecast RMSE.
specialists: [trader, risk, grid_ops]
high_stakes: true
```

## Wire it in

```bash
# As a domain config (data.* keys)
aetherforge train -c configs/base.yaml -c configs/domains/example_logistics.yaml

# Or as a pure pack file
aetherforge train -c configs/base.yaml \
  -o data.domain=energy_markets \
  -o data.domain_pack=packs/energy_markets.yaml
```

Copy `_template` → your field. Train any industry the same way.
