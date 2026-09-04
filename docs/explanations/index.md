# Explanations

This section is a conceptual guide to annnet.

Read these three first. They are the ones that say what an AnnNet *is*.

- [Design philosophy](design-philosophy.md): what the package optimises for, and what it refuses to.
- [Architecture overview](architecture-overview.md): the modules, and which of them a caller ever names.
- [Internal representation](internal-representation.md): the canonical stores, what derives from them, and the eight attribute tables.

Then, on reading and changing a graph:

- [Reading the graph](reading-the-graph.md): the frame as the default answer, and which namespace builds which one.
- [Mutation and derived state](mutation-and-derived-state.md): what a write invalidates, and what a read rebuilds.
- [Aspects, order, and windows](aspects-and-windows.md): ordinal against categorical aspects, and selecting layers as a query.
- [Incidence and the matrices](math-incidence.md): what each named matrix means, and where a self-loop and a boundary edge land.
- [Multilayer and multi-aspect graphs](math-multilayer.md): how annnet models layered graph state.
- [Slices and views](managers-and-views.md): how one graph can hold several contexts without duplication.
- [Interoperability](interoperability.md): what annnet keeps, what other tools expect, and how conversion works.
- [Storage and IO](io-annnet.md): native persistence and exchange formats.
- [Tracking changes](history-and-diffs.md): mutation history, snapshots, and diffs.
- [Adding edges](add-edges.md): accepted edge input forms and dispatch rules.

Use the [API reference](../reference/index.md) when you need exact signatures and details about parameters, attributes, and methods.
