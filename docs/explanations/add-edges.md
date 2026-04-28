# Adding edges

`add_edges` is the single entry point for all edge creation in annnet. It handles
binary edges, directed and undirected hyperedges, stoichiometric coefficients,
supra-node (multilayer) edges, and edge-entity placeholders — all through one
method, dispatching on the shape of the input.

## Dispatch

When you call `G.add_edges(...)`, the first thing that happens is a shape check:

```
G.add_edges(*args, **kwargs)
    │
    ├─ single arg that is a list/generator of dicts or tuples?
    │       ├─ all items are hyperedge dicts?  → batch hyperedge path
    │       ├─ all items are binary?           → batch binary path (optimised)
    │       └─ mixed?                          → item-by-item loop
    │
    └─ everything else                         → single-edge path
```

The single-edge path returns a `str` (the edge ID).  
The batch path always returns a `list[str]`.

---

## Input forms — single edge

### Binary edge

```python
G.add_edges("A", "B")
G.add_edges("A", "B", weight=0.5, edge_id="e1", directed=True)
```

Both endpoints must be strings. If either vertex does not exist yet, it is
created automatically.

### Supra-node edge (multilayer)

```python
G.add_edges(("A", ("healthy",)), ("B", ("treated",)))
```

Each endpoint is a `(vertex_id, layer_coord_tuple)` pair. Required when the graph
was created with `aspects=`. The multilayer kind is inferred:

| Condition | Kind |
|---|---|
| Same vertex ID, different layers | `coupling` |
| Same layer, different vertex IDs | `intra` |
| Different vertex ID and layer | `inter` |

In a multilayer graph, passing a bare string ID raises `ValueError`.

### Undirected hyperedge

```python
G.add_edges(["A", "B", "C"])
```

`src` is a list; `tgt` defaults to `None`. All members get `+weight` in the
incidence column.

### Directed hyperedge

```python
G.add_edges(["A", "B"], ["C", "D"])   # tail → head
```

First list is the tail (source side, `+weight`), second is the head (target side, `-weight`).

### Stoichiometric edge

```python
# Negative coeff → source side, positive coeff → target side
G.add_edges({"glucose": -1.0, "atp": -1.0, "glucose-6p": 1.0, "adp": 1.0})

# Explicit two-dict form
G.add_edges({"A": -1.0, "B": -2.0}, {"C": 3.0})
```

Literal coefficient values are written directly into the incidence matrix column.
This form is **single-edge only**.

### Edge-entity placeholder (single)

```python
G.add_edges(None, None, as_entity=True, edge_id="virtual_e1", role="enzyme")
```

Creates a named edge that has no incident vertices yet, but occupies a row in the
entity space so it can later be used as an endpoint.
`edge_id` is required here — there is nothing to derive an auto-ID from.

---

## Input forms — batch

Pass a single list (or generator) as the first argument. No positional `src`/`tgt`.

### Binary batch

```python
# 2-tuples
G.add_edges([("A", "B"), ("C", "D")])

# 3-tuples  (weight in position 2)
G.add_edges([("A", "B", 0.5), ("C", "D", 2.0)])

# dicts  (keys: source/target or src/tgt)
G.add_edges([
    {"source": "A", "target": "B", "weight": 0.5, "edge_id": "e1"},
    {"src": "C", "tgt": "D"},
])
```

### Hyperedge batch

```python
# Undirected
G.add_edges([
    {"members": ["A", "B", "C"]},
    {"members": ["B", "D"], "edge_id": "h2"},
])

# Directed (tail → head)
G.add_edges([
    {"tail": ["A", "B"], "head": ["C"]},
])
```

### Edge-entity placeholder batch

Pass dicts with no `source`/`target` and set `as_entity=True` at the batch level.
Each item is registered as a null-endpoint entity — a connectable row in the entity
space with no incidence column.

```python
G.add_edges(
    [
        {"edge_id": "EE1", "role": "enzyme", "pathway": "glycolysis"},
        {"edge_id": "EE2", "role": "enzyme", "pathway": "tca"},
    ],
    as_entity=True,
    slice="Healthy",
)
```

Any keys other than `edge_id`, `slice`, `weight`, `edge_directed`, `edge_type`,
and `propagate` are stored as edge attributes. `as_entity=True` is required; omitting
it when items have no source/target raises `ValueError`.

### Mixed batch

A list that contains both binary and hyperedge items is accepted, but dispatches
item-by-item and loses the bulk-path optimisation.

---

## Parameters

### Single-edge parameters

| Parameter | Default | Description |
|---|---|---|
| `weight` | `1.0` | Incidence coefficient. Written as `+weight` on the source side, `−weight` on the target side (directed), or `+weight` on both sides (undirected). Ignored for the stoich form. |
| `edge_id` | auto | Explicit ID. If the ID exists → update in-place. If new → create. |
| `directed` | `None` | Per-edge override. `None` inherits the graph default, then falls back to `True`. |
| `parallel` | `"update"` | Policy when `edge_id` is `None` and the same endpoints already have an edge. See below. |
| `slice` | active slice | Slice that receives the edge. Auto-created if it doesn't exist. |
| `as_entity` | `False` | Register the edge as a connectable entity row in the incidence matrix. |
| `propagate` | `"none"` | Slice propagation after insertion. See below. |
| `flexible` | `None` | Data-driven direction policy dict. See below. |
| `**attrs` | — | Arbitrary edge attributes upserted into the edge attribute table. |

### Batch-level defaults

| Parameter | Default | Description |
|---|---|---|
| `default_weight` | `1.0` | Weight for items that don't carry one. |
| `default_edge_directed` | `None` | Directedness for items that don't specify it. |
| `default_propagate` | `"none"` | Propagate policy for items that don't specify it. |
| `slice` | active slice | Default slice for items that don't specify one. |
| `as_entity` | `False` | Apply to all inserted edges. |

---

## The `parallel` policy

Applies **only in the single-edge path and only when `edge_id` is `None`**.

| Value | Behaviour |
|---|---|
| `"update"` (default) | If an edge with the same endpoint set already exists, return that edge's ID and update it in-place. No new edge is created. |
| `"parallel"` | Always create a new edge, even if one already exists between the same endpoints. |
| `"error"` | Raise `ValueError` if any edge already exists between the same endpoints. |

When you supply an explicit `edge_id`:

- The ID already exists → always update in-place (parallel policy is ignored).
- The ID is new → always create. If `parallel="error"` and the same endpoints are
  already connected by a different edge, `ValueError` is raised.

---

## The `propagate` policy

Controls which slices receive the edge after insertion.

| Value | Behaviour |
|---|---|
| `"none"` (default) | Edge is added only to the specified slice. |
| `"shared"` | Edge is added to every slice that already contains **both** endpoints. |
| `"all"` | Edge is added to every slice that contains **either** endpoint. |

---

## Flexible direction

An edge's effective direction can be driven by one of its own attribute values.

```python
G.add_edges("A", "B", flexible={
    "var": "score",       # which edge attribute to read
    "threshold": 0.0,     # decision boundary
    "scope": "edge",      # "edge" (read from this edge) or "global"
    "above": "forward",   # direction when var > threshold
    "tie": "undirected",  # direction when var == threshold
})
```

`flexible` is **single-edge path only**. Setting it causes `_apply_flexible_direction`
to run immediately after the edge is stored.

---

## What cannot be done

| Limitation | Detail |
|---|---|
| Stoich form in batch mode | The stoichiometric dict is normalised only in `_parse_edge_inputs`, which is not called by the batch path. |
| `parallel` policy per batch item | The batch path is optimised for throughput and has no per-item dedup logic. |
| `flexible` direction in batch mode | Not wired into the batch path. |
| Bare string IDs in a multilayer graph | `_add_edge_impl` raises `ValueError` if any endpoint is not a supra-node when `G.is_multilayer` is `True`. |
| Items without `source`/`target` without `as_entity=True` | Batch items with no endpoints raise `ValueError` unless `as_entity=True` is set. |
| Single-edge `src=None, tgt=None` without `edge_id` | In the single-edge path, `edge_id` is mandatory for null-endpoint entities — the auto-ID counter has no input to derive from. In batch mode, `edge_id` can be omitted and an auto-ID is assigned. |
| Passing an exhausted generator | The dispatcher materialises the generator once. An already-exhausted generator produces an empty batch silently. |

---

## How the incidence matrix is written

For a directed binary edge `A → B` with `weight=2.0`:

```
Rows:      A     B
Column:  +2.0  -2.0
```

For an undirected edge `{A, B}`:

```
Rows:      A     B
Column:  +2.0  +2.0
```

For a directed hyperedge with `tail=[A,B]`, `head=[C]`, `weight=1.0`:

```
Rows:      A      B      C
Column:  +1.0   +1.0   -1.0
```

See [Incidence representation](math-incidence.md) for the full matrix algebra.

---

## See also

- [Incidence representation](math-incidence.md) — matrix layout, sign convention, operators
- [Multilayer and multi-aspect graphs](math-multilayer.md) — supra-node form in depth
- [Slices and views](managers-and-views.md) — how `slice=` and `propagate=` interact with slice state
- API reference: [`AnnNet.add_edges`][annnet.core.graph.AnnNet.add_edges]
