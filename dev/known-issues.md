# Known issues

Defects found and not fixed yet, each with a short script that shows it.

The defects found in October 2026 are fixed. `tests/test_known_issues.py` holds
a regression test for each of them.

## Not reproduced

One round trip through SIF lost a hyperedge, on a graph built with
`from_edge_frame`. Round trips of graphs with undirected and directed
hyperedges, with and without `from_edge_frame`, all kept every edge. The
original graph is not known.
