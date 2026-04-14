# Explanations

This section is a conceptual guide to annnet. A good reading order is:

1. [Design principles](design-philosophy.md): the decisions that shape annnet.
2. [Package architecture](architecture-overview.md): how the object model and package modules fit together.
3. [Internal representation](internal-representation.md): the actual structural SSOT and compatibility boundary.
4. [Mutation and derived state](mutation-and-derived-state.md): how writes propagate through registries, overlays, and caches.
5. [Incidence representation](math-incidence.md): the graph formalism underneath the package.
6. [Multilayer and multi-aspect graphs](math-multilayer.md): how annnet models layered graph state.
7. [Slices and views](managers-and-views.md): how one graph can hold several contexts without duplication.
8. [Interoperability](interoperability.md): what annnet keeps, what other tools expect, and how conversion works.
9. [Storage and IO](io-annnet.md): native persistence and exchange formats.
10. [Tracking changes](history-and-diffs.md)
11. [Typical analysis patterns](common-patterns.md)

Use the [API reference](../reference/index.md) when you need exact signatures rather than concepts.
