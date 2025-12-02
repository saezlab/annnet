# AnnNet — Annotated Network Data Structures for Science
**(BSD-3 License)** 
AnnNet (**Annotated Network**) is a unified, high-expressivity graph platform designed to bring the convenience of **AnnData-style** annotated containers to **networks, multilayer structures, and hypergraphs**.   
It targets systems biology, network biology, omics integration, computational social science, and any domain needing **fully flexible graph semantics** with modern stable storage and interoperability.

---
### Why AnnNet?
Straight answer: nothing else combines all of this at once:
- **Maximum graph expressiveness**:   
hyperedges, directed/undirected hybrid edges, parallel edges, self-loops, edge-edge edges, vertex-edge edges, custom directionality, graph-level and edge-level semantics.
- **Multilayer networks**:   
full support based on the Kivelä et al. multilayer network framework (layers, aspects, inter-layer edges).
- **Slicing system**:   
fast creation of subnetworks, clusters, and arbitrary “slices” of elements.
- **Annotated tables everywhere**:   
Polars-backed tables for vertices, edges, layers, vertex-layer couples, slices, and graph metadata/attributes.
- **AnnData-style API**:   
.obs, .var, .X, layers,.cache, .idx, etc..   Familiar for users of scanpy/anndata.
- **Interoperability without friction**:   
import/export with NetworkX, igraph, graph-tool, SBML, GraphML, CX2 (Cytoscape), SIF, Excel, CSV, JSON, graphdir, dataframes (Narwhals).
- **Algorithm proxying**:   
seamless lazy calls to algorithms from networkx, igraph, and graph-tool via proxy objects.
- **Disk-backed, lossless storage as .annnet** :  
	- Zarr for matrices.   
	- Parquet for annotated tables.  
	- JSON for metadata.   
	Reopen without loss of structure or metadata.


*AnnNet is a fresh development, available for public testing, we appreciate feedback in GitHub issues <https://github.com/saezlab/annnet>.

---

## 🛠️ Installation
To install annnet, you can use pip:

```bash
pip install annnet
```

Optional dependencies for extended functionality:

```bash
pip install annnet[networkx,igraph]
pip install annnet[graph-tool]
```

---

## Quick Start

```python
G = an.Graph(directed=True)  # default direction; can be overridden per-edge

# Add vertices with attributes
G.add_vertices([
    ('A', {'name': 'a'}),
    ('B', {'name': 'b'}),
    ('C', {'name': 'c'}),
    ('D', {'name': 'd'}),
])

# Create slices and set active
G.add_slice("toy")
G.add_slice("train")
G.add_slice("eval")
G.slices.active = "toy"   # same effect as set_active_slice("toy")

# Add vertices (with attributes) in 'toy'
for v in ["A","B","C","D"]:
    G.add_vertex(v, label=v, kind="gene")

# 1) Binary directed
e_dir = G.add_edge("A", "B", weight=2.0, edge_directed=True, relation="activates")

# 2) Binary undirected
e_undir = G.add_edge("B", "C", weight=1.0, edge_directed=False, relation="binds")

# 3) Self-loop
e_loop = G.add_edge("D", "D", weight=0.5, edge_directed=True, relation="self")

# 4) Parallel edge
e_parallel = G.add_parallel_edge("A", "B", weight=5.0, relation="alternative")

# 5) Vertex–edge (hybrid) edges
G.add_edge_entity("edge_e1", description="signal")
e_vx = G.add_edge("edge_e1", "C", edge_type="vertex_edge", edge_directed=True, channel="edge->vertex")

# 6) Hyperedge (undirected, 3-way membership)
e_hyper_undir = G.add_hyperedge(members=["A","C","D"], weight=1.0, tag="complex")

# 7) Hyperedge (directed head→tail)
e_hyper_dir = G.add_hyperedge(head=["A","B"], tail=["C","D"], weight=1.0, reaction="A+B->C+D")
```

---

## License
annnet is licensed under the BSD-3 License. See the [LICENSE](LICENSE) file for details.

