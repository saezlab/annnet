# Multilayer measures

Measures and dynamics over the supra-graph of a multilayer network, the graph
that `anet.layers.supra()` builds with one row per node-layer placement. Each
function takes the graph first.

```python
from annnet.algorithms import multilayer

pc = multilayer.participation_coefficient(anet)
lam2 = multilayer.sweep_coupling_regime(anet, [0.1, 1.0, 10.0])
```

::: annnet.algorithms.multilayer
    options:
      show_root_heading: false
      members_order: source
