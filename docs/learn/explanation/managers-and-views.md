# Managers and views

annnet exposes several manager accessors and view utilities to organize functionality.

## SliceManager (`G.slices`)
- Create/delete: `G.add_slice(id, **attrs)`, `G.drop_slice(id)`
- Activate: `G.slices.active = id`
- List: `G.list_slices(include_default=True)`
- Memberships: `G.add_vertex_to_slice(id, v)`, `G.add_edge_to_slice(id, e)`
- Set algebra: `G.slice_union(ids)`, `G.slice_intersection(ids)`, `G.slice_difference(a, b)`
- Per‑slice edge overrides: `G.set_edge_slice_attrs(id, e, weight=...)`, `G.get_effective_edge_weight(e, slice=id)`

## LayerManager (`G.layers`)
- Define aspects/layers: `G.set_aspects(aspects=[...], elem_layers={...})`
- Presence: `G.add_presence(v, layer_tuple)`, `G.has_presence(v, layer_tuple)`
- Edge types: `G.add_intra_edge_nl(u, v, layer)`, `G.add_inter_edge_nl(u, layer_u, v, layer_v)`, `G.add_coupling_edge_nl(v, layer_a, layer_b)`
- Matrices: `G.supra_adjacency()`, `G.supra_laplacian(kind="comb"|"norm")`, `G.adjacency_tensor_view()`

## IndexManager (`G.idx`)
- Vertex/entity maps: `entity_to_idx`, `idx_to_entity`, `entity_types`
- Edge maps: `edge_to_idx`, `idx_to_edge`
- Queries: `G.get_edge_ids(u, v)`, `G.has_vertex(v)`, `G.has_edge(edge_id=...)`

## CacheManager (`G.cache`)
- Materialize CSR/CSC: `G.cache.ensure_csr()`, `G.cache.ensure_csc()`
- Clear caches: `G.cache.clear()`

## Graph views and subgraphs
- Read‑only filter view: `G.view(vertices=[...], edges=[...], slices=[...])`
- Materialized subgraphs:
  - Vertex‑induced: `G.subgraph(vertices)`
  - Edge‑induced: `G.edge_subgraph(edges)`
  - Combined: `G.extract_subgraph(vertices=[...], edges=[...])`

## Direction and hyperedges
- Edge direction: graph default via `Graph(directed=...)`, per‑edge override via `add_edge(..., edge_directed=...)`
- Hyperedges:
  - Undirected: `G.add_hyperedge(members=[...], weight=..., tag=...)`
  - Directed head→tail: `G.add_hyperedge(head=[...], tail=[...], weight=...)`
  - Stoichiometry: `G.set_hyperedge_coeffs(eid, {vertex: coeff, ...})`

## Backend interoperability (proxies)
- NetworkX: `G.nx.<algo>(G)`; concrete graph via `G.nx.backend(directed=..., hyperedge_mode="skip|expand", slice=..., simple=True)`
- igraph: `G.ig.<algo>(G)` (if python‑igraph installed)
- graph‑tool: `G.gt.<module>.<algo>(...)` (if graph‑tool installed)

See Architecture Overview for deeper context and data structures.


## Cheatsheet

| Area | Method | One‑liner |
| --- | --- | --- |
| Slices | `G.add_slice(id, **attrs)` | Create a slice with optional metadata |
| Slices | `G.slices.active = id` | Set current active slice |
| Slices | `G.slice_union(ids)` | Union of multiple slices (membership) |
| Slices | `G.set_edge_slice_attrs(id, e, weight=...)` | Override edge attrs in a slice |
| Layers | `G.set_aspects(aspects, elem_layers)` | Define Kivelä aspects/layers |
| Layers | `G.add_presence(v, layer_tuple)` | Declare vertex presence in a layer |
| Layers | `G.supra_adjacency()` | Build supra‑adjacency matrix |
| Index | `G.get_edge_ids(u, v)` | List edge IDs between two vertices |
| Index | `G.has_vertex(v)` / `G.has_edge(...)` | Membership checks |
| Cache | `G.cache.ensure_csr()` | Materialize CSR; `ensure_csc()` for CSC |
| Views | `G.view(vertices=..., edges=..., slices=...)` | Read‑only filtered view |
| Subgraphs | `G.subgraph(vertices)` | Vertex‑induced subgraph |
| Subgraphs | `G.edge_subgraph(edges)` | Edge‑induced subgraph |
| Direction | `Graph(directed=...)` | Default directionality at graph level |
| Direction | `add_edge(..., edge_directed=...)` | Per‑edge direction override |
| Hyperedges | `add_hyperedge(members=[...])` | Undirected hyperedge |
| Hyperedges | `add_hyperedge(head=[...], tail=[...])` | Directed hyperedge |
| Proxies | `G.nx.backend(..., simple=True)` | Get NX graph; collapse multiedges |
