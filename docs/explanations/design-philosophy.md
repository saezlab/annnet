# Design principles

annnet is built around a small set of design choices.

## Keep the full problem intact for longer

Many graph datasets are not naturally simple graphs. They may mix directed and
undirected relations, hyperedges, repeated edges, multiple conditions, and rich
annotations. annnet is designed to keep those structures in one container
instead of forcing an early reduction to a simpler graph type.

## Separate structure from annotations

Topology and annotations are both first-class, but they are stored differently.
Structure is represented as sparse graph data. Annotations live in tabular
structures, which makes typed columns, bulk updates, filtering, and export much
more practical than per-node or per-edge dictionaries.

## Make context explicit

Slices and multilayer state are part of the model, not external bookkeeping.
That means different conditions, views, or layer assignments remain attached to
the same graph object instead of being scattered across copies or side tables.

## Interoperate rather than replace

annnet is not meant to replace NetworkX, igraph, graph-tool, or tabular tools.
It acts as a high-fidelity source of truth and converts into other ecosystems
when you need their algorithms, file formats, or workflows.

## Preserve structure on disk

The native storage layout is meant to round-trip the graph faithfully, including
annotations, slices, multilayer state, and history-related metadata. Persistence
is part of the design, not an afterthought.

## Consequence

The result is a richer container than a minimal graph API. That is a deliberate
tradeoff: annnet optimizes for expressive scientific graph data and controlled
conversion, not for the smallest possible abstraction.
