## Architecture Overview

`graph.py` implements a **sparse incidence-matrix graph** with first-class support for:

- Directed/undirected/mixed edges
- Parallel edges (multigraph)
- Hyperedges (k-ary relations)
- Multilayer networks (Kivelä framework)
- Slices (named subgraph partitions)
- Rich attributes via Polars DataFrames
- History/versioning with automatic mutation logging
- Lazy interop with NetworkX, igraph, graph-tool

---

## 1. Core Data Model

### 1.1 Incidence Matrix Representation

Unlike adjacency-matrix graphs (V×V), this library uses an **incidence matrix** (V×E):

```
         e0   e1   e2   e3   (edges)
    v0 [ +1   0    0   +1 ]
    v1 [ -1  +1    0    0 ]
    v2 [  0  -1   +1    0 ]
    v3 [  0   0   -1   -1 ]
(vertices)
```

**Why incidence matrix?**

- Natural representation for hyperedges (edge connects >2 vertices)
- Efficient for sparse graphs
- Supports parallel edges trivially (each edge = separate column)
- Encodes direction via sign: `+w` = source/head, `-w` = target/tail
- Undirected edges: `+w` on all incident vertices

### 1.2 Storage Format

```python
self._matrix = scipy.sparse.dok_matrix((n, e), dtype=np.float32)
```

**DOK (Dictionary of Keys)** sparse format:

- Stores only non-zero entries as `{(row, col): value}`
- O(1) random access for insertion/update
- Converts to CSR/CSC for fast row/column operations
- Pre-allocated capacity with geometric growth to reduce resizes

### 1.3 Entity System

The graph has a unified **entity** concept covering both vertices and edge-entities:

```python
self.entity_to_idx = {}    # entity_id (str) -> row index (int)
self.idx_to_entity = {}    # row index -> entity_id
self.entity_types = {}     # entity_id -> "vertex" | "edge"
self._num_entities = 0     # actual count (≤ matrix rows)
```

**Edge-entities** are hybrid vertex-edge objects that can be endpoints of other edges, enabling metagraph patterns.

### 1.4 Edge System

```python
self.edge_to_idx = {}        # edge_id (str) -> column index
self.idx_to_edge = {}        # column index -> edge_id
self.edge_definitions = {}   # edge_id -> (source, target, edge_type)
self.edge_weights = {}       # edge_id -> float
self.edge_directed = {}      # edge_id -> bool (per-edge override)
self._num_edges = 0          # actual count (≤ matrix columns)
```

**Parallel edges**: Multiple edges between same endpoints get unique `edge_id` values, each with its own column.

---

## 2. Directionality Model

### 2.1 Three Levels of Direction Control

1. **Graph default**: `Graph(directed=True|False|None)`
2. **Per-edge override**: `add_edge(..., edge_directed=True|False)`
3. **Mixed graphs**: Some edges directed, others undirected

### 2.2 Incidence Encoding

|Edge Type|Source Cell|Target Cell|
|---|---|---|
|Directed|`+weight`|`-weight`|
|Undirected|`+weight`|`+weight`|
|Self-loop|`+weight`|(same cell)|

### 2.3 Resolution Logic

```python
def _is_directed_edge(self, edge_id):
    return self.edge_directed.get(edge_id, self.directed or True)
```

---

## 3. Hyperedges

### 3.1 Definition

Hyperedges connect arbitrary sets of vertices (not just pairs):

```python
self.hyperedge_definitions = {}  # edge_id -> metadata dict
self.edge_kind = {}              # edge_id -> "hyper" | "binary" | "intra" | "inter" | "coupling"
```

### 3.2 Two Hyperedge Types

**Undirected hyperedge** (set membership):

```python
G.add_hyperedge(members=["a", "b", "c", "d"], edge_id="meeting_1")
# Stored as: {"members": {"a","b","c","d"}, "directed": False}
# Matrix: all members get +weight in the column
```

**Directed hyperedge** (head → tail):

```python
G.add_hyperedge(head=["author1", "author2"], tail=["reviewer"], edge_id="review_1")
# Stored as: {"head": {...}, "tail": {...}, "directed": True}
# Matrix: head vertices get +weight, tail vertices get -weight
```

### 3.3 Stoichiometry Support (SBML)

For biochemical reactions with coefficients:

```python
G.set_hyperedge_coeffs("reaction_1", {"A": -2, "B": -1, "C": +1, "D": +3})
# Writes exact coefficients into incidence column
```

---

## 4. Slices (Subgraph Partitions)

### 4.1 Concept

Slices are **named membership sets** that partition or overlap the graph:

```python
self._slices = {
    "default": {"vertices": set(), "edges": set(), "attributes": {}},
    "team_alpha": {"vertices": {"alice", "bob"}, "edges": {"e1"}, "attributes": {"dept": "eng"}},
    ...
}
self._current_slice = "default"  # active slice for new elements
```

### 4.2 Slice Operations

|Operation|Method|Result|
|---|---|---|
|Union|`slice_union(["A", "B"])`|Elements in A ∪ B|
|Intersection|`slice_intersection(["A", "B"])`|Elements in A ∩ B|
|Difference|`slice_difference("A", "B")`|Elements in A \ B|
|Create from op|`create_slice_from_operation("C", result)`|New slice from result|

### 4.3 Per-Slice Edge Weights

Same edge can have different weights in different contexts:

```python
G.set_edge_slice_attrs("context_a", "e1", weight=1.0)
G.set_edge_slice_attrs("context_b", "e1", weight=5.0)

G.get_effective_edge_weight("e1", slice="context_a")  # 1.0
G.get_effective_edge_weight("e1", slice="context_b")  # 5.0
```

---

## 5. Multilayer Networks (Kivelä Framework)

### 5.1 Multi-Aspect Structure

The library implements the full Kivelä multilayer model:

```python
G.set_aspects(
    aspects=["time", "relation"],           # aspect names
    elem_layers={
        "time": ["t1", "t2", "t3"],          # elementary layers per aspect
        "relation": ["friendship", "work"]
    }
)
# Creates layer space: {(t1,friendship), (t1,work), (t2,friendship), ...}
```

### 5.2 Vertex-Layer Presence (V_M)

Vertices exist in specific layers:

```python
self._V = set()   # all vertex IDs
self._VM = set()  # {(vertex_id, layer_tuple), ...}

G.add_presence("alice", ("t1", "friendship"))
G.has_presence("alice", ("t1", "friendship"))  # True
```

### 5.3 Edge Types in Multilayer

|Type|Description|Example|
|---|---|---|
|**Intra-layer**|Both endpoints in same layer|alice↔bob in (t1, friendship)|
|**Inter-layer**|Endpoints in different layers|alice@L1 → bob@L2|
|**Coupling**|Same vertex across layers|alice@L1 ↔ alice@L2|

```python
G.add_intra_edge_nl("alice", "bob", ("t1", "friendship"))
G.add_inter_edge_nl("alice", ("t1", "F"), "bob", ("t2", "F"))
G.add_coupling_edge_nl("alice", ("t1", "F"), ("t1", "W"))
```

### 5.4 Supra-Adjacency Matrix

Flattens multilayer to single matrix where rows/cols = (vertex, layer) pairs:

```python
G.ensure_vertex_layer_index()  # builds stable row mapping
A = G.supra_adjacency()        # CSR matrix
L = G.supra_laplacian()        # L = D - A
```

### 5.5 Tensor View

4-index representation: A[u, α, v, β] = weight of edge (u@α) → (v@β)

```python
tensor = G.adjacency_tensor_view()
# Returns: {vertices, layers, ui, ai, vi, bi, w}
# where (ui[k], ai[k]) → (vi[k], bi[k]) with weight w[k]

# Round-trip conversions:
A_supra = G.flatten_to_supra(tensor)
tensor_back = G.unflatten_from_supra(A_supra)
```

---

## 6. Attribute System (Polars DataFrames)

### 6.1 Attribute Tables

```python
self.vertex_attributes = pl.DataFrame(schema={"vertex_id": pl.Utf8})
self.edge_attributes = pl.DataFrame(schema={"edge_id": pl.Utf8})
self.slice_attributes = pl.DataFrame(schema={"slice_id": pl.Utf8})
self.edge_slice_attributes = pl.DataFrame(schema={
    "slice_id": pl.Utf8, "edge_id": pl.Utf8, "weight": pl.Float64
})
self.graph_attributes = {}  # global metadata dict
```

### 6.2 Reserved vs Pure Attributes

**Structural keys** (filtered out of attribute tables):

```python
_EDGE_RESERVED = {"edge_id", "source", "target", "weight", "edge_type", 
                  "directed", "slice", "members", "head", "tail", ...}
```

Only **pure attributes** (user data) go into DataFrames.

### 6.3 Upsert Mechanism

The `_upsert_row()` method handles insert-or-update with:

- Automatic column creation for new attributes
- Type inference and casting
- Per-table caching for fast existence checks

```python
G.set_vertex_attrs("alice", age=30, role="engineer")
G.get_vertex_attrs("alice")  # {"age": 30, "role": "engineer"}
```

### 6.4 Views (Read-Only)

```python
G.vertices_view()  # Polars DF of vertex attributes
G.edges_view(include_weight=True, include_directed=True)
G.slices_view()
G.aspects_view()   # Kivelä aspects
G.layers_view()    # Kivelä layers
```

---

## 7. Traversal & Queries

### 7.1 Neighbor Access

```python
G.neighbors("alice")       # all adjacent vertices
G.out_neighbors("alice")   # successors (directed out-edges)
G.in_neighbors("alice")    # predecessors (directed in-edges)
G.successors("alice")      # alias for out_neighbors
G.predecessors("alice")    # alias for in_neighbors
```

### 7.2 Edge Queries

```python
G.has_vertex("alice")                    # bool
G.has_edge(edge_id="e1")                 # bool
G.has_edge(source="a", target="b")       # (bool, [edge_ids])
G.get_edge_ids("alice", "bob")           # [edge_id, ...]
G.incident_edges("alice")                # [edge_indices]
G.degree("alice")                        # int
```

### 7.3 Listings

```python
G.vertices()              # [vertex_id, ...]
G.edges()                 # [edge_id, ...]
G.edge_list()             # [(src, tgt, eid, weight), ...]
G.get_directed_edges()    # [eid, ...] where directed=True
G.get_undirected_edges()  # [eid, ...] where directed=False
```

---

## 8. Subgraphs & Views

### 8.1 Materialized Subgraphs

```python
# Vertex-induced: keep vertices + edges fully inside
sub = G.subgraph(["alice", "bob", "carol"])

# Edge-induced: keep edges + incident vertices
sub = G.edge_subgraph(["e1", "e2", "e3"])

# Combined filter
sub = G.extract_subgraph(vertices=[...], edges=[...])
```

### 8.2 Lazy Views (GraphView)

```python
view = G.view(vertices=["alice", "bob"], edges=None, slices=["team_a"])
# Returns GraphView object - filters without copying
```

---

## 9. Spectral Methods

### 9.1 Matrix Construction

```python
A = G.supra_adjacency()                    # adjacency
L = G.supra_laplacian(kind="comb")         # L = D - A (combinatorial)
L = G.supra_laplacian(kind="norm")         # normalized Laplacian
P = G.transition_matrix()                  # row-stochastic P = D⁻¹A
```

### 9.2 Eigenvalue Analysis

```python
lambda2, fiedler = G.algebraic_connectivity()  # 2nd smallest eigenvalue
vals, vecs = G.k_smallest_laplacian_eigs(k=6)
lambda_max, v = G.dominant_rw_eigenpair()      # largest eigenvalue of P
```

### 9.3 Dynamics

```python
# Random walk step: p' = p @ P
p1 = G.random_walk_step(p0)

# Diffusion step: x' = x - τ * L @ x
x1 = G.diffusion_step(x0, tau=0.1, kind="comb")
```

### 9.4 Coupling Regime Analysis

```python
# Sweep coupling strength ω and measure spectral properties
results = G.sweep_coupling_regime(
    scales=[0.1, 0.5, 1.0, 2.0],
    metric="algebraic_connectivity"
)
```

---

## 10. History & Versioning

### 10.1 Automatic Mutation Logging

Every mutating method is wrapped to log:

```python
{
    "version": 42,
    "ts_utc": "2025-12-05T10:30:00.123456Z",
    "mono_ns": 123456789,  # monotonic nanoseconds
    "op": "add_vertex",
    "vertex_id": "alice",
    "attributes": {"age": 30},
    "result": "alice"
}
```

### 10.2 Control

```python
G.enable_history(True)   # enable/disable logging
G.clear_history()        # clear in-memory log
G.mark("checkpoint")     # insert manual marker
```

### 10.3 Export

```python
G.history(as_df=True)                    # Polars DataFrame
G.export_history("log.parquet")          # .parquet, .ndjson, .json, .csv
```

### 10.4 Snapshots & Diffs

```python
snap1 = G.snapshot(label="before")
# ... mutations ...
snap2 = G.snapshot(label="after")

diff = G.diff("before", "after")
diff.added_vertices    # set of new vertex IDs
diff.removed_edges     # set of deleted edge IDs
```

---

## 11. Interoperability

### 11.1 Adapter System

Located in separate adapter files:

| Adapter                       | Format                                 |
| ----------------------------- | -------------------------------------- |
| `networkx_adapter.py`         | NetworkX Graph/DiGraph                 |
| `igraph_adapter.py`           | igraph.Graph                           |
| `graphtool_adapter.py`        | graph_tool.Graph                       |
| `GraphML_adapter.py`          | GraphML XML                            |
| `json_adapter.py`             | JSON                                   |
| `dataframe_adapter.py`        | all Narwhals python dataframe packages |
| `GraphDir_Parquet_adapter.py` | Directory of Parquet files             |
| `SIF_adapter.py`              | Simple Interaction Format              |
| `cx2_adapter.py`              | CX2 (Cytoscape)                        |
| `SBML_adapter.py`             | Systems Biology Markup Language        |
| `sbml_cobra_adapter.py`       | COBRA metabolic models (WIP)           |

### 11.2 Conversion Methods

```python
nx_g = G.to_networkx()
gt_g = G.to_graphtool()
cx_g = G.to_cx2()
```

### 11.3 Lazy Proxies

Call algorithms without explicit conversion:

```python
# Proxy intercepts calls, converts G, runs algorithm
G.nx.shortest_path(G, "alice", "bob")
G.nx.louvain_communities(G)
G.ig.community_multilevel(G, weights="weight")
G.gt.module.algorithm(...)
```

---

## 12. Manager APIs

### 12.1 Accessor Properties

```python
G.slices   # SliceManager - slice CRUD operations
G.layers   # LayerManager - multilayer operations  
G.idx      # IndexManager - entity↔row, edge↔col lookups
G.cache    # CacheManager - CSR/CSC materialization
```

### 12.2 AnnData-like API

For bioinformatics workflows:

```python
G.X        # incidence matrix (like AnnData.X)
G.obs      # vertex attributes (observations)
G.var      # edge attributes (variables)
G.uns      # unstructured metadata (graph_attributes)
```

---

## 13. Memory Layout

### 13.1 Pre-allocation Strategy

```python
G = Graph(n=1000, e=5000)  # pre-allocate capacity
```

Matrix grows geometrically (1.5x) to avoid frequent resizes:

```python
def _grow_rows_to(target):
    if target > rows:
        new_rows = max(target, rows + max(8, rows >> 1))
        self._matrix.resize((new_rows, cols))
```

### 13.2 Memory Estimation

```python
G.memory_usage()  # bytes estimate
# Counts: matrix nnz × 12 bytes + dict entries × 100 bytes + DF sizes
```

### 13.3 String Interning

Hot strings (vertex IDs, edge IDs, slice IDs) are interned via `sys.intern()` for faster dict lookups.

---

## 14. Composite Vertex Keys

### 14.1 Definition

```python
G.set_vertex_key("type", "name")  # declare key fields
```

### 14.2 Lookup by Attributes

```python
G.add_vertex("v1", type="person", name="Alice")
vid = G.get_vertex_by_attrs(type="person", name="Alice")  # "v1"
```

### 14.3 Edge Endpoints by Attributes

```python
G.add_edge(
    source={"type": "person", "name": "Alice"},
    target={"type": "org", "name": "Acme"}
)
# Automatically resolves to vertex IDs via composite key
```

---

## 15. Thread Safety & Caveats

### 15.1 Pre-allocation vs Actual Size

```python
G._matrix.shape      # (capacity_rows, capacity_cols) - may be larger
G._num_entities      # actual vertex/entity count
G._num_edges         # actual edge count
```

### 15.2 DOK vs CSR

- **DOK**: Good for random writes (building phase)
- **CSR**: Good for row operations (query phase)

The library auto-converts as needed, but `tocsr()` has overhead.

---

## 16. File I/O

### 16.1 Native Format

```python
G.write("graph.annnet")           # lossless save
G_loaded = Graph.read("graph.annnet")
```
**.annnet Shema: **
```
graph.annnet/
├── manifest.json                 # root descriptor (format, counts, compression, etc.)
├── structure/
│   ├── incidence.zarr/           # Zarr v3 group holding COO [coordinate list] arrays
│   │   ├── zarr.json             # Zarr v3 group metadata (includes group attributes)
│   │   ├── row/                  # Zarr array (int32) of entity indices (COO row)
│   │   ├── col/                  # Zarr array (int32) of edge indices   (COO col)
│   │   └── data/                 # Zarr array (float32) of weights      (COO data)
│   │   # group attributes include: {"shape": [n_entities, n_edges]}
│   ├── entity_to_idx.parquet     # entity_id → row index
│   ├── idx_to_entity.parquet     # row index → entity_id
│   ├── entity_types.parquet      # entity_id → "vertex" | "edge"
│   ├── edge_to_idx.parquet       # edge_id → column index
│   ├── idx_to_edge.parquet       # column index → edge_id
│   ├── edge_definitions.parquet  # edge_id → (source, target, edge_type) for simple edges
│   ├── edge_weights.parquet      # edge_id → weight
│   ├── edge_directed.parquet     # edge_id → bool | null
│   ├── edge_kind.parquet         # edge_id → "binary" | "hyper" | "intra" | "inter"
│   └── hyperedge_definitions.parquet
│       # columns: edge_id, directed(bool), members(List[Utf8]) OR head(List[Utf8]), tail(List[Utf8])
│
├── tables/
│   ├── vertex_attributes.parquet      # vertex-level DF [dataframe]
│   ├── edge_attributes.parquet        # edge-level DF
│   ├── slice_attributes.parquet       # slice metadata
│   └── edge_slice_attributes.parquet  # (slice_id, edge_id, weight)
│
├── layers/                           # Kivela Multilayer Structures
│   ├── metadata.json                 # {"aspects": [...], "elem_layers": {...}}
│   ├── vertex_presence.parquet       # vertex_id, layer(List[str])
│   ├── edge_layers.parquet           # edge_id, layer_1(List[str]), layer_2(List[str]|null)
│   ├── elem_layer_attributes.parquet # attributes for elementary layers
│   ├── aspect_attributes.json        # attributes for aspects
│   ├── tuple_layer_attributes.parquet # layer(List[str]), attributes(JSON)
│   └── vertex_layer_attributes.parquet # vertex_id, layer(List[str]), attributes(JSON)
│
├── slices/                           # Subgraph Views
│   ├── registry.parquet              # slice_id, name, metadata…
│   ├── vertex_memberships.parquet    # (slice_id, vertex_id)
│   └── edge_memberships.parquet      # (slice_id, edge_id, weight)
│
├── cache/                            # optional materialized views
│   ├── csr.zarr/                     # CSR [compressed sparse row] cache
│   └── csc.zarr/                     # CSC [compressed sparse column] cache
│
├── audit/
│   ├── history.parquet               # operation log (nested payloads stringified to JSON [JavaScript Object Notation])
│   ├── snapshots/                    # optional labeled snapshots
│   └── provenance.json               # creation time, software versions, etc.
│
└── uns/                              # unstructured metadata & results
    ├── graph_attributes.json
    └── results/`
```

### 16.2 Via Adapters

```python
## Example:
# Import
from annnet.io import read_graphml, read_sif, read_json
G = read_graphml("graph.graphml")

# Export  
from annnet.io import write_graphml
write_graphml(G, "output.graphml")
```

---

## 17. Summary: Data Structure Map

```
Graph
├── _matrix (DOK sparse)           # V×E incidence matrix
├── entity_to_idx / idx_to_entity  # vertex ↔ row mapping
├── entity_types                   # "vertex" | "edge" per entity
├── edge_to_idx / idx_to_edge      # edge ↔ column mapping
├── edge_definitions               # (src, tgt, type) per edge
├── edge_weights                   # weight per edge
├── edge_directed                  # direction per edge
├── edge_kind                      # "binary"|"hyper"|"intra"|"inter"|"coupling"
├── hyperedge_definitions          # {members} or {head, tail} per hyperedge
├── _slices                        # slice_id → {vertices, edges, attributes}
├── vertex_attributes (Polars DF)  # pure vertex attributes
├── edge_attributes (Polars DF)    # pure edge attributes
├── slice_attributes (Polars DF)   # slice metadata
├── edge_slice_attributes (Polars) # per-slice edge overrides
├── graph_attributes (dict)        # global metadata
├── aspects / elem_layers          # Kivelä multilayer structure
├── _VM                            # vertex-layer presence set
├── _history                       # mutation log
└── _snapshots                     # named state snapshots
```

---
## 18. Design Philosophy

1. **Incidence over adjacency**: Generalizes to hypergraphs naturally
2. **Slices as first-class**: Logical partitions without data duplication
3. **Multilayer native**: Kivelä framework built-in, not bolted on
4. **Polars for attributes**: Columnar storage, fast filtering, type safety
5. **Lazy interop**: Don't pay conversion cost until needed
6. **History by default**: Audit trail without explicit opt-in
7. **Pre-allocation**: Amortized O(1) insertions for known sizes