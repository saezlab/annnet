# Review public Python object APIs

## Example: edge API of Graph object

current: members, head, tail
new: src, tgt

add_edge(src = {"a": -1, "b": -3"}, tgt = {"c": 1, "d": 3})  # directed, hyperedge with stoichiometry (= with node coefficients)
add_edge(src = {"a": -1, "b": -3", "c": 1, "d": 3})  # directed, hyperedge with stoichiometry (= with node coefficients)
add_edge(src = ["a", "b"], tgt = ["c", "d"], weight = 3)  # directed, weighted hyperedge
add_edge(src = ["a", "b", "c"])  # undirected hyperedge (can be weighted)
add_edge(src = ["a", "b", "c"], weight = 2, parallel = "update")  # undirected, weighted hyperedge added as parallel to existing one
add_edge(src = "a", tgt = "b")  # binary directed edge
add_edge(src = ["a", "b"])  # undirected binary edge
add_edge(...) -> int  # num of added edges

add_edge(src = ["a", "b"], tgt = "__sink__")

  e1 e2
S  1
a -1  1
b  0  1
c  0  1
d  0  1

-- remove add_parallel_edge, add_hyperedge

parallel = False by default:
parallel = Literal['update' | 'error' | 'parallel']
- raise error if edge already exists
- or: update existing edge
- or: add parallel edge
Graph.parallel

vcount, ecount

eids -> Iterable[eid]
E -> Iterable[(etuple)]
edges(directed = True, hyper = False) -> Iterable[(etuple)]

etuple = (src, tgt, key, weight)

__iter__ = edges

edges(directed = True, hyper = False, src, tgt) -> Iterable[(etuple)]
src, tgt: list[vid]
-> implements in and out edges
-> question: how to handle direction? duplicate src, tgt for undirected lookup
replaces: in_edges, out_edges, incident_edges, get_edge_ids,
get_directed_edges, get_undirected_edges,

Same signatures as edges:

ecount(directed = True, hyper = False) -> int
remove_edges(...) -> int  # return number of removed edges

## General points to revise and improve Python APIs

- Do methods have the optimal name? No redundant parts, is it clear and
  intuitive?
- Are there methods which do the same or similar thing and can be merged?
- Arguments: can their number be reduced? Do they have the optimal name? Is
  their order the most intuitive and convenient? Are alternative types, values
  handled intuitive and smart way, e.g. "str" vs "list[str]" vs None?
