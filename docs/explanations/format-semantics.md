# What each format keeps

Every exchange format is narrower than an AnnNet graph, and they are narrow in
different places. This page says, for each one, what comes back when you write a
graph and read it again, and by what means. The statements are checked: a round
trip of one graph through every format below is a test
(`tests/test_roundtrip_semantics.py`), and it compares structure, not counts.

The graph in that test has a parallel pair of edges, a self-loop, an undirected
edge, a directed and an undirected hyperedge, an edge entity with attributes, an
edge that joins that entity to a node, edge and node attributes, and a slice.

## Three ways a format holds a graph

**The file holds all of it.** The native format, JSON, NDJSON, Parquet and CX2
carry their own record of what the format's tables have no column for (hyperedges,
slices, contextual attributes, edge entities, aspects) inside the file or the
directory. Reading returns the graph that was written. The dataframes API
(`to_dataframes` / `from_dataframes`) does the same in the dictionary it returns:
the frames, and a plain dict beside them for the parts that are not a frame.
Hand `from_dataframes` the whole dictionary.

**The adapter returns two things.** `to_nx`, `to_igraph` and `to_graphtool`
return a library graph and a manifest. The library graph is what that library can
represent; the manifest holds the rest. Keep the manifest and `from_nx`,
`from_igraph` and `from_graphtool` rebuild the graph. Read the library graph
alone (`from_nx` without a manifest) and you get what that graph says:
hyperedges are whatever reified nodes it holds, and an edge entity, which the
library has no notion of, is a node.

**A sidecar holds the rest.** SIF, CSV, Excel, GraphML and GEXF write the file
the format defines and, when the graph carries something the format cannot hold,
a companion `<file>.annnet-sidecar` beside it. Writing says so with an
`AnnNetLossWarning` naming what went to the sidecar. The sidecar records the
SHA-256 of the file it belongs to, so a file edited by hand is not silently
paired with a stale sidecar; reading refuses it. Read with
`sidecar='ignore'`, or after deleting the sidecar, and you get what the file
alone holds. Write with `sidecar=False` and nothing is kept beside it.

## What comes back

| format | edge direction | parallel edges, self-loops | hyperedges | edge entities | attributes | slices | mechanism |
|---|---|---|---|---|---|---|---|
| native `.annnet` | kept | kept | kept | kept | all eight addresses | kept | the file |
| JSON, NDJSON | kept | kept | kept | kept | edge and node; contextual through the file's extensions | kept | the file |
| Parquet | kept | kept | kept | kept | edge and node; contextual | kept | the directory's manifest |
| CX2 | kept | kept | kept | kept | edge and node; contextual | kept | the embedded manifest |
| dataframes | kept | kept | kept | kept | edge and node | kept | the returned dict |
| NetworkX, igraph | kept | kept | kept | kept | edge and node | kept | the manifest |
| graph-tool | kept | kept | kept | kept | edge and node | kept | the manifest |
| GraphML, GEXF | kept | kept | kept | kept | edge and node | kept | manifest inside the sidecar |
| SIF | kept | kept | kept, when every member is a plain node id | kept | node and edge | kept | the sidecar |
| CSV, Excel | kept | kept | kept, when every member is a plain node id | kept | edge; node from the sidecar | kept | the sidecar |

"Kept" means the round trip in the test returns the same edge ids, the same
endpoints on the same sides, the same weights and directedness, and the same
entity and edge kinds. Where a cell says a condition, that condition is stated
in the last section.

## Edge entities

An edge entity is an edge another edge can name as an endpoint. No exchange format
has such a thing, so each writer records the entities (their identity, the
layer they sit on, the edges that join one to a node, and their attributes) in
whichever of the three places above the format has, and each reader gives them
back. A reader never turns an edge entity into a node on its own: if a record
names an identity it cannot place, the attributes stored under it are dropped
with an `AnnNetLossWarning`, and the read still succeeds.

A manifest written before edge entities were recorded has no such record. Reading
one returns the entity as a node and reports the attributes that had no edge to
go to; it does not fail.

## The direction of an edge

A directed binary edge is written source first in every format. Two formats
used to sort the endpoints and so reverse any edge whose source sorted after its
target; that is fixed, and the test above writes an edge from `B` to `A` through
every format to keep it fixed.

## Limits worth knowing

- **Only binary edges fit an edge list.** SIF, CSV and Excel write a row per
  two-endpoint edge. A hyperedge goes to the sidecar and comes back only when
  every member is a plain node id. On a multilayer graph a member is a placement,
  not an id, and such a hyperedge is left out with an `AnnNetLossWarning`
  naming it.
- **CSV and Excel carry no node table.** A node that no edge names, and every
  node attribute, live in the sidecar.
- **graph-tool exchanges through vertices.** An edge whose endpoint is an edge
  entity has no vertex to sit on, so it is absent from the graph-tool graph and
  comes back from the manifest.
- **A manifest is the adapter's lossless channel.** `hyperedge_mode='skip'`
  leaves the hyperedges out of the library graph; the manifest still holds
  them.
- **SBML is not a general exchange format.** It holds a bipartite reaction
  network and its coefficients, and the rest is not kept. PyG `HeteroData`
  carries tensors for learning and is outside the table above.
- **The coefficient bookkeeping of a hyperedge** is stored under the private
  attribute names `__source_attr` and `__target_attr`, and only when a
  coefficient is not 1. `public_only=True` leaves them out of a file.

See [Storage and IO](io-annnet.md) for the native layout and
[Interoperability](interoperability.md) for the conversions between in-memory
backends.
