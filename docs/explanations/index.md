# Explanations

This section is a conceptual guide to annnet.

- [Design principles](design-philosophy.md): the decisions that shape annnet.
- [Package architecture](architecture-overview.md): how the object model and package modules fit together.
- [Internal representation](internal-representation.md): the in-memory graph model.
- [Mutation and derived state](mutation-and-derived-state.md): how writes propagate through registries, overlays, and caches.
- [Incidence representation](math-incidence.md): the graph formalism underneath the package.
- [Multilayer and multi-aspect graphs](math-multilayer.md): how annnet models layered graph state.
- [Slices and views](managers-and-views.md): how one graph can hold several contexts without duplication.
- [Interoperability](interoperability.md): what annnet keeps, what other tools expect, and how conversion works.
- [Storage and IO](io-annnet.md): native persistence and exchange formats.
- [Tracking changes](history-and-diffs.md): mutation history, snapshots, and diffs.

Use the [API reference](../reference/index.md) when you need exact signatures and details about parameters, attributes, and methods.
