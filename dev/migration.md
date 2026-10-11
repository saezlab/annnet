# Migrating to the current surface

This release gave every attribute address one way to be read and written,
made selections typed and live, and made views resolve correctly. It removed
the names that duplicated those. The package is before its first stable release,
so a removed name carries no alias: it raises `AttributeError` naming its
replacement. This page is the complete map, old to new, with the shape of the
call on both sides. The surface after the change is stated in
`docs/reference/api-contract.md`.

## The graph object

| old | new |
|---|---|
| `G.obs` | `G.attrs.nodes` |
| `G.var` | `G.attrs.edges` |
| `G.nodes()` | `list(G.N)` or `G.N.ids` |
| `G.edges()` | `list(G.E)` or `G.E.ids` |
| `G.nv`, `G.ncount()` | `len(G.N)` |
| `G.ne`, `G.ecount()` | `len(G.E)` |
| `G.ncount(supra=True)` | `G.nv_supra` |
| `G.get_node(node_id)` | `G.N.at(node_id)` |
| `G.get_edge(edge_id)` | `G.E.at(edge_id)` |
| `G.get_edge_ids(source, target)` | `G.has_edge(source, target)[1]` |
| `G.get_edges_by_direction(directed)` | `G.E.select(directed=directed).ids` |
| `G.in_edges(nodes)` | `G.incident_edges(nodes, direction='in')` |
| `G.out_edges(nodes)` | `G.incident_edges(nodes, direction='out')` |
| `G.remove_node(node_id)` | `G.remove_nodes(node_id)` |
| `G.remove_edge(edge_id)` | `G.remove_edges(edge_id)` |
| `G.contextual_table(level)` | `G.attrs.table(address)` |
| `G.views.nodes(...)` | `G.attrs.table('nodes', derived=True)` |
| `G.views.edges(slice=, layer=, in_slice=, ...)` | `G.attrs.table('edges', derived=True, slice=, layer=, in_slice=, ...)` |
| `G.views.hyperedges(...)` | `G.attrs.table('edges', derived=True, include_binary=False, ...)` |
| `G.views.slices()` / `aspects()` / `layers()` / `layers_view()` | `G.attrs.table('slices' / 'aspects' / 'layers', derived=True)` |
| `G.views.entity_kinds()` | `G.entity_kinds()` |

```python
# before
n = G.nv
first = G.nodes()[0]
e = G.get_edge('e1')
frame = G.views.edges(in_slice='fit')

# after
n = len(G.N)
first = G.N[0]
e = G.E.at('e1')
frame = G.attrs.table('edges', derived=True, in_slice='fit')
```

## Attributes: `G.attrs`

The eight addresses are `nodes`, `edges`, `slices`, `aspects`, `layers`,
`edge_slices`, `node_layers` and `elementary_layers`. Every one answers the
same expressions; below, `address` stands for any of them and `key` for the
key that address takes (`docs/reference/api-contract.md`, section 3).

| old | new |
|---|---|
| `G.attrs.get_node_attrs(node_id)` | `G.attrs.row('nodes', node_id)` |
| `G.attrs.get_attr_node(node_id, name, default)` | `G.attrs.row('nodes', node_id).get(name, default)` |
| `G.attrs.get_attr_nodes(ids)` | `G.attrs.rows('nodes', ids)` |
| `G.attrs.set_node_attrs(node_id, **attrs)` | `G.attrs.update('nodes', {node_id: attrs})` |
| `G.attrs.set_node_attrs_bulk(rows)` | `G.attrs.update('nodes', rows)` |
| `G.attrs.get_edge_attrs(edge_id)` | `G.attrs.row('edges', edge_id)` |
| `G.attrs.get_attr_edge(edge_id, name, default)` | `G.attrs.row('edges', edge_id).get(name, default)` |
| `G.attrs.get_attr_edges(ids)` | `G.attrs.rows('edges', ids)` |
| `G.attrs.get_attr_from_edges(name, default)` | `dict(zip(G.E.ids, G.E.column(name, default)))` |
| `G.attrs.get_edges_by_attr(name, value)` | `G.E.select(**{name: value}).ids` |
| `G.attrs.set_edge_attrs(edge_id, **attrs)` | `G.attrs.update('edges', {edge_id: attrs})` |
| `G.attrs.set_edge_attrs_bulk(rows)` | `G.attrs.update('edges', rows)` |
| `G.attrs.get_slice_attr(slice_id, name, default)` | `G.attrs.row('slices', slice_id).get(name, default)` |
| `G.attrs.set_slice_attrs(slice_id, **attrs)` | `G.attrs.update('slices', {slice_id: attrs})` |
| `G.slices.attrs(slice_id)` | `G.attrs.row('slices', slice_id)` |
| `G.attrs.edge_slice(slice_id, edge_id)` | `G.attrs.row('edge_slices', (slice_id, edge_id))` |
| `G.attrs.get_edge_slice_attr(slice_id, edge_id, name, default)` | `G.attrs.row('edge_slices', (slice_id, edge_id)).get(name, default)` |
| `G.attrs.set_edge_slice_attrs(slice_id, edge_id, **attrs)` | `G.attrs.update('edge_slices', {(slice_id, edge_id): attrs})` |
| `G.attrs.set_edge_slice_attrs_bulk(slice_id, items)` | `G.attrs.update('edge_slices', {(slice_id, edge_id): attrs for edge_id, attrs in items})` |
| `G.attrs.set_slice_edge_weight(slice_id, edge_id, w)` | `G.attrs.update('edge_slices', {(slice_id, edge_id): {'weight': w}})` |
| `G.attrs.get_effective_edge_weight(edge_id, slice)` | `G.E.effective_weight(edge_id, slice)` |
| `G.layers.attrs(layer)` | `G.attrs.row('layers', layer)` |
| `G.layers.set_attrs(layer, **attrs)` | `G.attrs.update('layers', {layer: attrs})` |
| `G.layers.node_attrs(node_id, layer)` | `G.attrs.row('node_layers', (node_id, layer))` |
| `G.layers.set_node_attrs(node_id, layer, **attrs)` | `G.attrs.update('node_layers', {(node_id, layer): attrs})` |
| `G.layers.set_node_attrs_bulk(values, layer=, key=)` | `G.attrs.update('node_layers', values, layer=, key=)` |
| `G.layers.aspect_attrs(aspect)` | `G.attrs.row('aspects', aspect)` |
| `G.layers.set_aspect_attrs(aspect, **attrs)` | `G.attrs.update('aspects', {aspect: attrs})` |
| `G.layers.elementary_attrs(aspect, label)` | `G.attrs.row('elementary_layers', (aspect, label))` |
| `G.layers.set_elementary_attrs(aspect, label, **attrs)` | `G.attrs.update('elementary_layers', {(aspect, label): attrs})` |
| `G.attrs.get_graph_attribute(key, default)` | `G.uns.get(key, default)` |
| `G.attrs.get_graph_attributes()` | `dict(G.uns)` |
| `G.attrs.set_graph_attribute(key, value)` | `G.uns[key] = value` |
| `G.attrs.audit_attributes()` | `G.attrs.audit()` |

```python
# before
G.attrs.set_node_attrs('akt', score=0.9)
G.layers.set_node_attrs('akt', ('stim',), expr=0.95)
G.attrs.set_slice_edge_weight('fit', 'e1', 2.0)
w = G.attrs.get_effective_edge_weight('e1', slice='fit')

# after
G.attrs.update('nodes', {'akt': {'score': 0.9}})
G.attrs.update('node_layers', {('akt', ('stim',)): {'expr': 0.95}})
G.attrs.update('edge_slices', {('fit', 'e1'): {'weight': 2.0}})
w = G.E.effective_weight('e1', slice='fit')
```

Two rules that hold at every address and did not before: writing `None` (or
NaN) for a field in `update` deletes it, and a batch that fails partway leaves
the graph as it was, schema included.

### If you already used `G.attrs[address, key]`

An earlier form of this surface indexed `G.attrs`: a live row you could write
through, and a table you could assign. That never shipped in a release, and
the final surface names each action instead. Reading returns a detached copy;
nothing you do to what you read reaches the graph.

| transitional | final |
|---|---|
| `G.attrs['nodes', key]` (read) | `G.attrs.row('nodes', key)`, a `dict` |
| `G.attrs['nodes', key]['score'] = 3` | `G.attrs.update('nodes', {key: {'score': 3}})` |
| `G.attrs['nodes', key] = {...}` (replace the row) | `G.attrs.delete('nodes', keys=[key])`, then `G.attrs.update('nodes', {key: {...}})` |
| `del G.attrs['nodes', key]` | `G.attrs.delete('nodes', keys=[key])` |
| `del G.attrs['nodes', key]['score']` | `G.attrs.delete('nodes', keys=[key], names=['score'])` |
| `G.attrs['nodes']` (read) | `G.attrs.nodes` or `G.attrs.table('nodes')` |
| `G.attrs['nodes'] = table`, `G.attrs.nodes = table` | `G.attrs.replace('nodes', table)` |

Each old form raises `TypeError` naming its replacement. New in the final
surface:

```python
G.attrs.delete('nodes', names=['note'])  # drop a field from every row
G.attrs.delete('nodes', keys=['a', 'b'])  # clear those rows' attributes
G.attrs.delete('nodes', keys=['a'], names=['note'])
G.attrs.replace('nodes', table)  # the whole address, from a table

table = G.attrs.table('nodes')  # any dataframe backend
kept = table[table['score'] > 0.5]  # filter with your own dataframe library
chosen = G.attrs.from_frame('nodes', kept)  # a fixed selection of the graph's keys
```

`G.attrs.select(...)` is the other way to choose rows, and it is a different
thing: a live query that keeps its conditions and answers again after the graph
changes. `from_frame` keeps the identities the frame named and does not follow
the graph.

Elementary layers are keyed `(aspect, label)`. A caller that wrote
`'time_0h'` writes `('time', '0h')`; the legacy string is decoded only where a
file or a legacy frame delivers it, and refused when it could name two rows.

## Matrix caches

| old | new |
|---|---|
| `G.cache.get_adjacency()` | `G.cache.adjacency` |
| `G.cache.get_csr()` | `G.cache.csr` |
| `G.cache.get_csc()` | `G.cache.csc` |

## Views

A view carries the graph's reading vocabulary under the graph's names, and
nothing under another name.

| old | new |
|---|---|
| `V.obs`, `V.var` | `V.attrs.table('nodes')`, `V.attrs.table('edges')` |
| `V.nodes_df(...)`, `V.edges_df(...)` | `V.attrs.table('nodes', derived=True, ...)`, `V.attrs.table('edges', derived=True, ...)` |
| `V.node_count`, `V.edge_count` | `len(V.N)`, `len(V.E)` |
| `V.node_ids`, `V.edge_ids` | `V.N.ids`, `V.E.ids` |
| `V.subview(...)` | `V.view(...)` |
| `V.X` | `V.B` |
| `V.nx`, `V.ig`, `V.gt`, `V.cache` | `V.materialize().nx`, and likewise |

What changed underneath: a view resolves over placement keys and edge ids
with a stated boundary rule (`closed` by default, `open` on request), keeps a
hyperedge whole or drops it, keeps selected isolated nodes, resolves an
edge-entity endpoint through its backing edge, and lets a nested view only
restrict. A view re-resolves when the graph changes. Every write through a
view — a column, `V.attrs.update`, `replace` or `delete`, `uns`, a table
property, a selection made from `V` — raises `ReadOnlyViewError`;
`V.materialize()` gives an independent graph.

## Selections

`G.N.select(...)` and `G.E.select(...)` kept their names and gained an
algebra. A selection is live, ordered, and combines with `&`, `|` and `-`. Code
that turned two selections into Python sets can keep the selections:

```python
# before
hits = set(G.N.select(score__gte=0.5)) | set(G.N.select(group='x'))
hits -= set(G.N.select(flag=True))
V = G.view(nodes=sorted(hits))

# after
hits = (G.N.select(score__gte=0.5) | G.N.select(group='x')) - G.N.select(flag=True)
V = G.view(nodes=hits)  # live: V follows the graph
```

The same operators — `eq`, `ne`, `in`, `not_in`, `lt`, `lte`, `gt`, `gte`,
`isnull` — parse in `G.attrs.select(address, ...)` and in `G.layers.where`. A
row selection on `node_layers` or `edge_slices` names composite keys; it
projects onto an axis explicitly, `.project('nodes')` or `.project('edges')`,
and nothing collapses a placement to a bare id on its own.

## Edge records

`G.E.at(edge_id).kind` is the structural kind — `binary`, `hyper` or
`node_edge` — and `directed` is the direction. A caller that matched
`hyper_directed` or `hyper_undirected` reads `kind == 'hyper'` and `directed`.
The derived edges table and `G.E['kind']` say the same.

## Registered dependents

`dependents.toml` lists, per package, the names it was observed calling before
this release, the spellings they migrate to and one line per changed name.
`tests/test_dependents.py` fails the build when a listed name stops resolving,
and the release gate (`ANNNET_RELEASE_GATE=1`) fails while a package's
migration is still `required_before_release`. `DEPENDENTS.md` says what each
package has to do.
