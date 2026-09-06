# Method adapters

Thin adapters onto established method packages. See
[Joining measurements](../../explanations/joining-measurements.md) for the loop
they sit in.

!!! warning "Experimental"
    Nothing under `annnet.experimental` is part of the stable public surface.
    This package is going to leave this repository, and the import path will
    change.

**No adapter reimplements any arithmetic.** That belongs to the method package,
and each has a test pinning numeric parity against calling it directly on the
equivalent DataFrame. What an adapter contributes is the declaration before, the
reading off the graph, and the additive write-back after.

```bash
pip install 'annnet[decoupler]'   # or annnet[corneto], or annnet[methods]
```

## decoupler

::: annnet.experimental.sysbio.methods.decoupler.run
::: annnet.experimental.sysbio.methods.decoupler.regulon
::: annnet.experimental.sysbio.methods.decoupler.DecouplerResult
    options:
      show_root_heading: true

## CORNETO / CARNIVAL

::: annnet.experimental.sysbio.methods.corneto.run
::: annnet.experimental.sysbio.methods.corneto.pkn
::: annnet.experimental.sysbio.methods.corneto.CarnivalResult
    options:
      show_root_heading: true

## Use cases

The composed loop, for someone reading the documentation rather than building a
pipeline. Each is a composition over the adapters and nothing else.

::: annnet.experimental.sysbio.usecases.tf_activity
::: annnet.experimental.sysbio.usecases.causal_subnetwork
::: annnet.experimental.sysbio.usecases.activity_to_obs
