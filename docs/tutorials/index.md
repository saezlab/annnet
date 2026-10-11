# Tutorials

These notebooks show annnet in a larger analysis or at the boundary with another
tool. For the graph model and everyday API usage, start with the
[Guide](../guide/index.md).

## Case studies

| Notebook | Analysis | Environment |
| --- | --- | --- |
| [Multi-condition causal signaling](multi-condition-signaling.ipynb) | Fit a small signaling network with CORNETO; keep nine synthetic perturbation experiments, selected interactions, and predictions in AnnNet. | [YAML](environments/multi-condition-signaling.yml) |
| [TGF-beta fibrosis response](tgfb-fibrosis.ipynb) | Combine signaling, regulation, complexes, and metabolism across time; join measurements, trace paths, export to Cytoscape, and forecast responses. | [YAML](environments/tgfb-fibrosis.yml) |

## Integrations

| Notebook | What it demonstrates | Environment |
| --- | --- | --- |
| [OmniPath table ingestion](omnipath-table-ingestion.ipynb) | Import a local OmniPath-style table through the knowledge-base client and add analysis slices. | [YAML](environments/omnipath-table-ingestion.yml) |
| [Cytoscape CX2 export](cytoscape-cx2-export.ipynb) | Choose a hyperedge projection and round-trip the embedded AnnNet manifest. | [YAML](environments/cytoscape-cx2-export.yml) |
| [PyG HeteroData export](pyg-heterodata-export.ipynb) | Convert a heterogeneous graph to tensors for PyTorch Geometric. | [YAML](environments/pyg-heterodata-export.yml) |
| [Causal activity bridge](causal-activity-bridge.ipynb) | Store a CORNETO CARNIVAL solution, selected interactions, and activities alongside the prior graph. | [YAML](environments/causal-activity-bridge.yml) |

## Reproduce a tutorial

Use the notebook's linked environment rather than installing every integration
into the guide environment. For example:

```bash
conda env create -f docs/tutorials/environments/cytoscape-cx2-export.yml
```

Activate the environment named in that YAML, then open the notebook in Jupyter.
Case studies may require downloaded data, cached input files, and a solver;
read their setup cells before running them. The website renders saved outputs
for these six tutorials. Its documentation build executes only the five Guide
notebooks, so a successful site build does not validate external services or
rerun the case studies.

The fibrosis case study currently needs prepared inputs that are not bundled
with this repository: `measured.parquet`, `responsive.parquet`, the cached
`signaling.cx2` backbone, `dorothea.tsv`, `omnipath_complexes.tsv`,
`Human-GEM.xml`, and `proteinatlas.tsv`. Its saved outputs can be read here, but
the environment alone is not sufficient to reproduce it.
