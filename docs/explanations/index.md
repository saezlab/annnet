# Explanations

This section explains how annnet represents a graph, how its state changes, and
how graph data moves between annnet and other tools. Follow the sections in
order for the main path, or jump directly to a topic.

## 1. Understand the model

### Principles and architecture

- [Design principles](design-philosophy.md): the choices that shape the package.
- [Package architecture](architecture-overview.md): the main modules and how they fit together.
- [Internal representation](internal-representation.md): canonical storage, indices, and attribute tables.

### Identity and representation

- [Edge tables, ids, and formats](edge-tables-and-formats.md): what an edge identity means and what each format can preserve.
- [Incidence representation](math-incidence.md): how graph structure appears in incidence matrices.

## 2. Work with graph state

### Inspect and change

- [Reading the graph](reading-the-graph.md): choose the right frame and namespace for a query.
- [Adding edges](add-edges.md): accepted edge inputs and how they are dispatched.
- [Mutation and derived state](mutation-and-derived-state.md): what writes maintain and what reads derive.

### Layers and context

- [Aspects, order, and windows](aspects-and-windows.md): organize layers and select them as a query.
- [Multilayer and multi-aspect graphs](math-multilayer.md): model structure across layer coordinates.
- [Slices and views](managers-and-views.md): work with contexts without duplicating the graph.

### Values

- [Node-layer values](values-and-scale.md): attach, read, and interpret values on node-layer pairs.

## 3. Connect and persist

### Interoperate

- [Interoperability](interoperability.md): convert graphs while making projections explicit.
- [Storage and IO](io-annnet.md): persist native state and exchange it with other formats.

### Track changes

- [Tracking changes](history-and-diffs.md): use mutation history, snapshots, and diffs.

When you need exact signatures, parameters, attributes, or methods, use the
[API reference](../reference/index.md). For runnable examples, use the
[Notebook Gallery](../tutorials/index.md).
