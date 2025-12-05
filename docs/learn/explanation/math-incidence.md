# Math: incidence matrix and operators

This page summarizes the core linear algebra behind annnet's incidence representation.

## Incidence matrix

- Let V be the set of entities (vertices and edge‑entities) and E the set of edges.
- The incidence matrix is `B ∈ R^{|V|×|E|}`; each column corresponds to an edge `e ∈ E`.

Sign convention (one convenient choice):
- Directed edge e: for each head/source endpoint `u`, set `B[u,e] = +w_e`; for each tail/target endpoint `v`, set `B[v,e] = −w_e`.
- Undirected edge e: for each endpoint `u`, set `B[u,e] = +w_e`.
- Hyperedges: set non‑zeros for all members (undirected) or heads/tails (directed) accordingly.

Weights and coefficients:
- Edge weights `w_e` scale the column; stoichiometric coefficients (SBML) write directly into `B[:,e]` as endpoint‑specific values.

## Native support for complex structures

- Hyperedges: columns with more than two non‑zeros; directed hyperedges use signed entries to indicate head→tail.
- Parallel edges: just more columns with the same endpoint sets (distinct edge IDs).
- Vertex→edge and edge→edge edges: edge‑entities occupy rows in V, so edges can connect any combination of rows.

## Operators from incidence

Let `W = diag(w_e)` be a diagonal matrix of per‑edge weights (or use the column norms if coefficients vary per endpoint).

- Undirected Laplacian: `L = B W Bᵀ` (positive semidefinite; generalizes to hyperedges).
- Directed adjacency (one construct):
  - Split `B` into positive and negative parts: `B⁺ = max(B,0)`, `B⁻ = max(−B,0)`.
  - Define `A = B⁺ W (B⁻)ᵀ` so that arcs go from heads to tails.
- Row‑stochastic transition: `P = D⁻¹ A` with `D = diag(A 1)` where `1` is the all‑ones vector.

Notes:
- Other directed Laplacians exist; the incidence construction above is compatible with several standard definitions.
- In multilayer settings, construct the supra incidence `B̃` over `(vertex, layer)` pairs and apply the same formulas.

