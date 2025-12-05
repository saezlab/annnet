# HowTo / FAQ

This FAQ explains why annnet uses an incidence matrix, and how that naturally supports hyperedges, vertex→edge edges, parallel edges, weights, and directionality.

## Why incidence matrix?

We represent the graph by a sparse incidence matrix `B ∈ R^{|V| × |E|}` where each column corresponds to an edge and each row corresponds to an entity (a vertex or an edge‑entity).

Intuition:
- Each edge is a column; its non‑zeros mark which entities it connects.
- Direction is encoded by the sign:
  - `+w` for a source/head membership
  - `−w` for a target/tail membership
  - Undirected membership uses `+w` for all incident entities
- Parallel edges are simply additional columns with the same endpoints.
- Hyperedges set multiple `+w` or `−w` entries in the column.

Small example with weights and mixed edge types:

```
Entities (rows):   a,  b,  c,  d
Edges (columns):  e1, e2, e3

B =
      e1  e2  e3
 a   +2   0  +1   # a is source in e1 (w=2), member in undirected e3 (w=1)
 b   -2  +1  +1   # b is target in e1 (w=2), member in e2/e3 (w=1)
 c    0  +1  +1   # c is member in e2/e3
 d    0  -2  +1   # d is target in e2 (w=2), member in e3

e1: directed a → b (weight 2)
e2: directed b → d (weight 2) but encoded as +1 on b and −2 on d (e.g., normalized head)
e3: undirected hyperedge {a,b,c,d} (weight 1)
```

From `B`, you can construct familiar operators:
- Undirected Laplacian: `L = B W Bᵀ` with diagonal `W = diag(w_e)`
- Directed adjacency (one option): `A = B⁺ W (B⁻)ᵀ` where `B⁺ = max(B,0)`, `B⁻ = max(−B,0)`

## How do hyperedges work?

Hyperedges are columns with more than two non‑zeros:
- Undirected hyperedge over members `M` puts `+w` in rows for all `v ∈ M`.
- Directed hyperedge with head `H` and tail `T` puts `+w` in rows `v ∈ H` and `−w` in rows `v ∈ T`.
- Stoichiometric coefficients (e.g., SBML) write exact per‑endpoint weights into `B[:, e]`.

This generalizes natively: no need to reify hyperedges into auxiliary nodes unless exporting to a format that requires it.

## Can edges connect to edges (vertex→edge, edge→edge)?

Yes. annnet has a unified entity set: rows correspond to both vertices and edge‑entities. You can register an edge‑entity (row) and then create edges whose endpoints are vertices, edge‑entities, or mixed.

Examples:

```python
G.add_edge_entity("e_meta", description="signal")
G.add_edge("e_meta", "C", edge_type="vertex_edge", edge_directed=True)
# Edge→edge is similar: endpoints can be previously declared edge‑entities
```

In the incidence matrix, those edge‑entities occupy rows like any vertex, so vertex→edge and edge→edge edges are just additional columns with non‑zeros at the appropriate rows.

## How are parallel edges represented?

Each parallel edge is its own column with a distinct ID. Aggregations are applied only when you explicitly request a simple backend graph (e.g., `G.nx.backend(..., simple=True)`), where multiple columns between the same endpoints are combined (e.g., summed weights).

## How are weights handled?

- Column scale represents the edge weight. For binary edges, you either store the weight at the head (directed) or split symmetrically (undirected). For hyperedges, coefficients go into the respective incidence entries.
- annnet also supports per‑slice overrides (e.g., the same edge has different effective weights in different contexts) via `edge_slice_attributes`.

## How do I get adjacency and Laplacian?

Common constructions from the incidence matrix:

```
Let W = diag(w_e) be diagonal weights per edge.
Undirected Laplacian:   L = B W Bᵀ
Directed adjacency:     A = B⁺ W (B⁻)ᵀ  with  B⁺ = max(B,0),  B⁻ = max(−B,0)
Row‑stochastic:         P = D⁻¹ A,  where D = diag(A 1)
```

annnet also provides multilayer (supra) versions of these matrices when layers/aspects are defined.
