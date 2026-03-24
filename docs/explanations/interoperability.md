# Interoperability

In annnet, interoperability means preserving as much graph meaning as possible
when moving between annnet and other tools, backends, or file formats.

## Why this matters

Most external graph tools expect a simpler structure than annnet can represent.
They may assume:

- only binary edges
- one global graph context
- no edge-entities
- limited multilayer support
- dictionary-style attributes rather than aligned tables

Interoperability in annnet is therefore not just file export. It is controlled
translation between models with different expressive power.

## Two kinds of interoperability

annnet separates two jobs:

- runtime backends for in-memory computation
- IO and exchange formats for persistence or data transfer

That separation is important because converting to another in-memory graph
backend is not the same thing as writing a file format.

## Runtime backends

Lazy proxies such as `G.nx`, `G.ig`, and `G.gt` expose the current annnet graph
through external graph libraries only when needed.

```python
# NetworkX
bc = G.nx.betweenness_centrality(G)

# Get a concrete backend graph with options
nxG = G.nx.backend(
    directed=True,
    hyperedge_mode="skip",
    slice="toy",
    simple=True,
)
```

This is useful when annnet is your source of truth but another library provides
a specific algorithm or workflow you need.

## Conversion is always a choice of projection

When exporting from annnet, you often have to choose how to project richer
structure into a simpler target:

- keep or collapse parallel edges
- drop, expand, or reify hyperedges
- select one slice or flatten several contexts
- keep only a subset of attributes
- preserve directedness exactly or coerce to the target model

Those choices are part of interoperability. They are not incidental details.

## Manifests and round-tripping

annnet uses manifests to preserve reconstruction details when a target backend
cannot represent the original graph directly.

```python
import annnet as an

nxG, manifest = an.to_nx(G, directed=True, hyperedge_mode="skip")
G2 = an.from_nx(nxG, manifest)
```

A manifest is especially useful when hyperedges, slices, or multiedges are
projected into a simpler graph shape.

## Relation to storage and IO

Interoperability is the broad concept of controlled conversion. [Storage and
IO](io-annnet.md) explains the concrete persistence and exchange mechanisms,
including the native `.annnet` format and the other supported file and table
interfaces.
