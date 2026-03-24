# Incidence representation

annnet uses a sparse incidence matrix because it can represent ordinary edges, hyperedges, parallel edges, and edge-entities without switching data structures.

## Basic idea

- Let `V` be the set of entities and `E` the set of edges.
- The incidence matrix is `B ∈ R^{|V|×|E|}`.
- Each column is one edge.
- Each row is one entity. In annnet, entities include both ordinary vertices and edge-entities.

Sign convention:

- Directed edge `e`: for each head or source endpoint `u`, set `B[u,e] = +w`.
- Directed edge `e`: for each tail or target endpoint `v`, set `B[v,e] = -w`.
- Undirected edge `e`: for each endpoint `u`, set `B[u,e] = +w`.

## What this makes possible

- Hyperedges become columns with more than two non-zero entries.
- Parallel edges become additional columns with the same endpoint set but different edge IDs.
- Vertex-to-edge and edge-to-edge relations work because edge-entities live in the same row space as ordinary vertices.
- Stoichiometric coefficients can be written directly into the corresponding incidence column.

## Example

```
Entities (rows):   a,  b,  c,  d
Edges (columns):  e1, e2, e3

B =
      e1  e2  e3
 a   +2   0  +1
 b   -2  +1  +1
 c    0  +1  +1
 d    0  -2  +1
```

- `e1`: directed `a → b` with weight `2`
- `e2`: directed edge with positive membership on `b` and negative membership on `d`
- `e3`: undirected hyperedge over `{a, b, c, d}`

## Hyperedges

- Undirected hyperedge over members `M`: put `+w` in every row for `v ∈ M`.
- Directed hyperedge with head `H` and tail `T`: put `+w` in rows for `H` and `-w` in rows for `T`.
- SBML-style stoichiometric coefficients can be stored directly as endpoint-specific values in the same column.

There is no need to reify hyperedges into auxiliary nodes unless you export to a format that requires that shape.

## Edge-entities

annnet can represent vertex→edge and edge→edge relations because edges can themselves appear as entities.

```python
G.add_edge_entity("e_meta", description="signal")
G.add_edge("e_meta", "C", edge_type="vertex_edge", edge_directed=True)
```

Once an edge-entity has a row in the incidence matrix, it behaves like any other endpoint.

## Parallel edges and weights

- Parallel edges are separate columns with distinct IDs.
- Edge weights scale the relevant column.
- Hyperedges can carry endpoint-specific coefficients.
- Slice-specific overrides can change the effective weight in a given slice without changing the base structure.

If you later ask for a simple backend graph, those parallel edges may be collapsed during conversion, but they remain distinct in annnet itself.

## Operators

Let `W = diag(w_e)` be a diagonal matrix of edge weights.

- Undirected Laplacian: `L = B W Bᵀ`
- Directed adjacency:
  - `B⁺ = max(B,0)`
  - `B⁻ = max(-B,0)`
  - `A = B⁺ W (B⁻)ᵀ`
- Row-stochastic transition: `P = D⁻¹ A`, with `D = diag(A 1)`

Other directed operators are possible, but these constructions capture the main idea used throughout annnet.

## Multilayer extension

For multilayer graphs, the same logic is applied to the supra incidence matrix over `(vertex, layer)` pairs. See [Multilayer and multi-aspect graphs](math-multilayer.md).

