# Math: multilayer operators

Let `L = A₁ × A₂ × … × A_k` be the cartesian product of aspects, where each aspect `A_i` has a finite set of elementary layers. A layer index is a tuple `α ∈ L`.

## Supra incidence and adjacency

- Build the set of present vertex‑layers `V×M = {(v, α) : v has presence in layer α}`.
- Define supra incidence `B̃ ∈ R^{|V×M| × |E|}` by mapping each original endpoint `(v, α)` of edge `e` to a row in `B̃` with the same sign convention as the monolayer case.

From `B̃`, obtain:
- Supra Laplacian: `L̃ = B̃ W B̃ᵀ` (undirected or symmetrized case).
- Directed supra adjacency: `Â = B̃⁺ W (B̃⁻)ᵀ`.

## Tensor representation

Alternatively, use a 4‑index tensor view `𝒜[u, α, v, β]` giving the weight of arcs from `(u, α)` to `(v, β)`.

- Fold to supra adjacency by indexing `(u,α)` and `(v,β)` pairs.
- Unfold from supra adjacency back to tensor by inverting the index mapping.

## Edge types in multilayer

- Intra‑layer: endpoints share the same layer tuple α.
- Inter‑layer: endpoints lie in different layer tuples.
- Coupling: endpoint is the same vertex across two layers `(v, α) ↔ (v, β)`.

These categories determine which blocks of `Â` are populated and how presence constraints apply.

