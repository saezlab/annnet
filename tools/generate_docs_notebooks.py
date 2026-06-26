"""Generate the runnable documentation notebooks and environment files."""

from __future__ import annotations

import json
import hashlib
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / 'docs' / 'tutorials' / 'notebooks'
TUTORIAL_DIR = NOTEBOOK_DIR / 'tutos'
SCENARIO_DIR = NOTEBOOK_DIR / 'scenarios'
ENV_DIR = NOTEBOOK_DIR / 'envs'


def md(source: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': textwrap.dedent(source).strip() + '\n',
    }


def code(source: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': textwrap.dedent(source).strip() + '\n',
    }


def notebook(cells: list[dict]) -> dict:
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'pygments_lexer': 'ipython3',
            },
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def write_notebook(relative: str, cells: list[dict]) -> None:
    path = NOTEBOOK_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells, start=1):
        digest = hashlib.sha1(
            f'{relative}:{index}:{cell["source"]}'.encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]
        cell['id'] = f'cell-{index:02d}-{digest}'
    path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + '\n')


INTRO_WITH_INFO = code(
    """
    import annnet as an

    an.info()
    """
)

IMPORT_ANNET = code(
    """
    import annnet as an
    """
)


def generate_tutorials() -> None:
    write_notebook(
        'tutos/01_quickstart.ipynb',
        [
            md(
                """
                # Quickstart

                Build a small signaling graph, inspect its edges, draw it, and
                save it in the native AnnNet format.
                """
            ),
            INTRO_WITH_INFO,
            md('## Create a graph'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.add_vertices(['EGF', 'EGFR', 'GRB2', 'SOS1', 'RAS', 'ERK'])
                G.add_edges('EGF', 'EGFR', edge_id='ligand_binding', weight=1.0)
                G.add_edges('EGFR', 'RAS', edge_id='canonical_signal', weight=0.9)
                G.add_edges('EGFR', 'RAS', edge_id='adapter_signal', weight=0.7)
                G.add_edges('GRB2', 'SOS1', edge_id='adapter_complex', directed=False, weight=0.55)
                G.add_edges(
                    src=['EGFR', 'GRB2', 'SOS1'],
                    tgt=['RAS'],
                    edge_id='signalosome',
                    directed=True,
                    weight=1.2,
                )
                G.add_edges('RAS', 'ERK', edge_id='mapk_signal', weight=0.75)

                print('shape:', G.shape)
                print('vertices:', sorted(G.vertices()))
                """
            ),
            md('## Inspect edges'),
            code(
                """
                G.views.edges().select(
                    ['edge_id', 'kind', 'source', 'target', 'head', 'tail', 'directed', 'effective_weight']
                )
                """
            ),
            md(
                """
                ## Draw the topology

                AnnNet stores the graph; Graphviz is just a rendering backend.
                In a notebook, returning the Graphviz object displays the graph
                directly.
                """
            ),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(G, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Save and reload'),
            code(
                """
                import os
                import tempfile

                tmp = tempfile.mkdtemp(prefix='annnet_quickstart_')
                path = os.path.join(tmp, 'signaling.annnet')
                an.write(G, path)
                reloaded = an.read(path)

                print('written:', os.path.basename(path))
                print('reloaded shape:', reloaded.shape)
                """
            ),
            md(
                """
                The native `.annnet` file preserves the parallel edges, the
                undirected adapter complex, and the signalosome hyperedge instead
                of flattening them into a plain edge list.
                """
            ),
        ],
    )

    write_notebook(
        'tutos/02_attributes_and_views.ipynb',
        [
            md(
                """
                # Attributes and views

                Attributes are arbitrary metadata on graph entities. Views
                expose graph state as dataframe-like tables.
                """
            ),
            IMPORT_ANNET,
            md('## Build a graph with metadata'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.add_vertices(['A', 'B', 'C', 'D', 'E'])
                G.add_edges('A', 'B', edge_id='e1', weight=1.0)
                G.add_edges('A', 'B', edge_id='e1_backup', weight=0.65)
                G.add_edges('B', 'C', edge_id='e2', weight=2.0)
                G.add_edges('C', 'D', edge_id='e3', weight=1.5)
                G.add_edges('C', 'E', edge_id='complex_ce', directed=False, weight=0.8)

                G.attrs.set_vertex_attrs('A', label='alpha', tier=1)
                G.attrs.set_vertex_attrs_bulk(
                    {
                        'B': {'label': 'beta', 'tier': 1},
                        'C': {'label': 'gamma', 'tier': 2},
                        'D': {'label': 'delta', 'tier': 2},
                        'E': {'label': 'epsilon', 'tier': 2},
                    }
                )
                G.attrs.set_edge_attrs_bulk(
                    {
                        'e1': {'interaction': 'activation', 'confidence': 0.90},
                        'e1_backup': {'interaction': 'activation', 'confidence': 0.65},
                        'e2': {'interaction': 'inhibition', 'confidence': 0.70},
                        'e3': {'interaction': 'activation', 'confidence': 0.95},
                        'complex_ce': {'interaction': 'binding', 'confidence': 0.85},
                    }
                )
                G.uns['study'] = 'demo'
                """
            ),
            md('## Query attributes'),
            code(
                """
                edge_names = {
                    row['edge_id']: f"{row['source']} -> {row['target']}"
                    if row['directed']
                    else f"{row['source']} -- {row['target']}"
                    for row in G.views.edges().iter_rows(named=True)
                }
                activation_edges = G.attrs.get_edges_by_attr('interaction', 'activation')

                print('A attrs:', G.attrs.get_vertex_attrs('A'))
                print('activation edges:', [edge_names[eid] for eid in sorted(activation_edges)])
                print('graph attrs:', G.attrs.get_graph_attributes())
                """
            ),
            md('## View tables'),
            code('G.views.vertices()'),
            code(
                """
                G.views.edges().select(
                    ['edge_id', 'source', 'target', 'interaction', 'confidence']
                )
                """
            ),
            md(
                """
                ## Visual check

                The drawing uses node IDs for readability and annotates edges
                with compact numeric weights plus the selected edge metadata.
                """
            ),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(
                    G,
                    backend='graphviz',
                    show_edge_labels=True,
                    edge_label_keys=['interaction'],
                )
                """
            ),
            md('## Reserved keys are protected'),
            code(
                """
                try:
                    G.attrs.set_edge_attrs('e1', source='X')
                except ValueError as err:
                    print('caught:', err)
                """
            ),
            md(
                """
                Views are the inspection layer for the same annotated graph
                object; no separate table has to be kept in sync.
                """
            ),
        ],
    )

    write_notebook(
        'tutos/03_tables_and_storage.ipynb',
        [
            md(
                """
                # Tables and storage

                Start from ordinary tables, create an AnnNet graph, and compare
                storage/interchange formats on the same small graph.
                """
            ),
            IMPORT_ANNET,
            md('## Create a graph from a table'),
            code(
                """
                import polars as pl

                edges = pl.DataFrame(
                    {
                        'edge_id': ['e1', 'e1_parallel', 'e2', 'e3'],
                        'source': ['A', 'A', 'B', 'C'],
                        'target': ['B', 'B', 'C', 'D'],
                        'weight': [0.9, 0.45, 0.7, 0.4],
                        'directed': [True, True, True, False],
                        'relation': ['activates', 'phosphorylates', 'inhibits', 'binds'],
                        'slice': ['baseline', 'stimulated', 'stimulated', 'stimulated'],
                    }
                )

                G = an.from_dataframe(edges, schema='edge_list')
                print('shape:', G.shape)
                print('slices:', G.slices.list())
                """
            ),
            md('## Export dataframe tables'),
            code(
                """
                tables = an.to_dataframes(G)
                print('tables:', sorted(tables))
                tables['edges'].select(['edge_id', 'source', 'target', 'relation', 'weight'])
                """
            ),
            md(
                """
                ## Visualize the imported edge table

                This confirms that the table was interpreted as the intended
                directed and undirected interactions.
                """
            ),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(
                    G,
                    backend='graphviz',
                    show_edge_labels=True,
                    edge_label_keys=['relation'],
                )
                """
            ),
            md('## CSV and Excel ingestion'),
            code(
                """
                import os
                import tempfile
                import pandas as pd

                tmp = tempfile.mkdtemp(prefix='annnet_tables_')
                csv_path = os.path.join(tmp, 'edges.csv')
                xlsx_path = os.path.join(tmp, 'edges.xlsx')

                edges.write_csv(csv_path)
                pd.DataFrame(edges.to_dicts()).to_excel(xlsx_path, index=False)

                from_csv = an.from_csv(csv_path)
                from_excel = an.from_excel(xlsx_path)

                print('csv shape:', from_csv.shape)
                print('excel shape:', from_excel.shape)
                print()
                print(open(csv_path, encoding='utf-8').read())
                """
            ),
            md('## Native, Parquet, JSON, and NDJSON outputs'),
            code(
                """
                native_path = os.path.join(tmp, 'graph.annnet')
                parquet_path = os.path.join(tmp, 'graph_parquet')
                json_path = os.path.join(tmp, 'graph.json')
                ndjson_path = os.path.join(tmp, 'graph_ndjson')

                an.write(G, native_path)
                an.to_parquet(G, parquet_path)
                an.to_json(G, json_path, indent=2)
                an.write_ndjson(G, ndjson_path)

                round_trips = {
                    'native': an.read(native_path).shape,
                    'parquet': an.from_parquet(parquet_path).shape,
                    'json': an.from_json(json_path).shape,
                    'ndjson files': sorted(os.listdir(ndjson_path)),
                }
                round_trips
                """
            ),
            md('## Inspect a JSON and NDJSON payload'),
            code(
                """
                import json

                with open(json_path, encoding='utf-8') as handle:
                    json_doc = json.load(handle)
                with open(os.path.join(ndjson_path, 'edges.ndjson'), encoding='utf-8') as handle:
                    edge_lines = [line.strip() for line in handle if line.strip()]

                print('JSON top-level keys:', sorted(json_doc))
                print('JSON edges:')
                print(json.dumps(json_doc['edges'], indent=2))
                print('edges.ndjson:')
                for line in edge_lines:
                    print(json.dumps(json.loads(line), indent=2))
                """
            ),
            md(
                """
                CSV and Excel are useful at human-facing boundaries. Native
                `.annnet` and Parquet are better when AnnNet remains the source
                of record. JSON and NDJSON are convenient when another system
                expects plain structured text.
                """
            ),
        ],
    )

    write_notebook(
        'tutos/04_slices_and_subgraphs.ipynb',
        [
            md(
                """
                # Slices and subgraphs

                Slices keep multiple graph contexts in one AnnNet object.
                Operations materialize smaller graphs from those contexts.
                """
            ),
            IMPORT_ANNET,
            md('## Create condition slices'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.add_vertices(['A', 'B', 'C', 'D', 'E'])
                G.slices.add('control', cohort='wildtype')
                G.slices.add('treated', cohort='drug')

                G.add_edges('A', 'B', edge_id='e1', slice='control', weight=1.0)
                G.add_edges('B', 'C', edge_id='e2', slice='control', weight=2.0)
                G.add_edges('B', 'C', edge_id='e2_treated', slice='treated', weight=5.0)
                G.add_edges('C', 'D', edge_id='e3', slice='treated', weight=3.0)
                G.add_edges('D', 'E', edge_id='e4', slice='treated', weight=1.5)
                G.add_edges(
                    'A',
                    'E',
                    edge_id='context_binding',
                    directed=False,
                    slice='treated',
                    weight=0.6,
                )

                print(G.slices.summary())
                """
            ),
            md('## Slice membership and algebra'),
            code(
                """
                edge_lookup = {}
                for row in G.views.edges().iter_rows(named=True):
                    arrow = '->' if row['directed'] else '--'
                    edge_lookup[row['edge_id']] = f"{row['source']} {arrow} {row['target']}"
                readable = lambda edge_ids: [edge_lookup[eid] for eid in sorted(edge_ids)]

                union = G.slices.union(['control', 'treated'])
                common = G.slices.intersect(['control', 'treated'])
                treated_only = G.slices.difference('treated', 'control')

                print('control:', readable(G.slices.edges('control')))
                print('treated:', readable(G.slices.edges('treated')))
                print('union:', readable(union['edges']))
                print('common:', readable(common['edges']))
                print('treated only:', readable(treated_only['edges']))
                """
            ),
            md(
                """
                ## Draw the full context graph

                The graph contains all context-specific edges. Slices decide
                which of those edges participate in a given context.
                """
            ),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(G, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Per-slice edge weights'),
            code(
                """
                G.slices.add_edge_to_slice('treated', 'e2')
                G.attrs.set_edge_slice_attrs('treated', 'e2', weight=8.0)

                print('base e2 weight:', G.attrs.get_effective_edge_weight('e2'))
                print('treated e2 weight:', G.attrs.get_effective_edge_weight('e2', slice='treated'))
                """
            ),
            md('## Materialize subgraphs'),
            code(
                """
                treated_graph = G.subgraph_from_slice('treated')
                edge_graph = G.ops.edge_subgraph(['e2', 'e3'])
                reversed_graph = G.ops.reverse()

                print('treated shape:', treated_graph.shape)
                print('edge-subgraph shape:', edge_graph.shape)
                print('reversed e1:', reversed_graph.edge_definitions['e1'])
                """
            ),
            md('## Draw the treated subgraph'),
            code(
                """
                plotting.plot(treated_graph, backend='graphviz', show_edge_labels=True)
                """
            ),
            md(
                """
                The treated subgraph materializes only the selected context:
                treated-only edges, the undirected context edge, and the
                slice-specific override on `e2` are included in the derived graph.
                """
            ),
        ],
    )

    write_notebook(
        'tutos/05_hyperedges_and_traversal.ipynb',
        [
            md(
                """
                # Hyperedges and traversal

                Hyperedges model complexes or reactions. Traversal helpers make
                local neighborhoods available without converting to another
                graph library.
                """
            ),
            IMPORT_ANNET,
            md('## Undirected complexes and directed reactions'),
            code(
                """
                H = an.AnnNet(directed=True)
                H.add_vertices(['Glc', 'ATP', 'G6P', 'ADP', 'HK1', 'PFK'])
                H.add_edges(['Glc', 'ATP', 'HK1'], edge_id='enzyme_complex', directed=False)
                H.add_edges(
                    src=['Glc', 'ATP'],
                    tgt=['G6P', 'ADP'],
                    edge_id='hexokinase',
                    directed=True,
                    weight=2.0,
                )
                H.add_edges('G6P', 'PFK', edge_id='activates_pfk')

                H.views.edges().select(['edge_id', 'kind', 'members', 'head', 'tail'])
                """
            ),
            md(
                """
                ## Draw the hypergraph

                Graphviz represents each true hyperedge through a small square
                connector node. Binary edges remain ordinary arrows.
                """
            ),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(H, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Endpoint coefficients'),
            code(
                """
                H.set_edge_coeffs(
                    'hexokinase',
                    {'Glc': -1.0, 'ATP': -1.0, 'G6P': 1.0, 'ADP': 1.0},
                )

                col = H._edges['hexokinase'].col_idx
                for vertex_id in sorted(H.vertices()):
                    row = H._entities[H._resolve_entity_key(vertex_id)].row_idx
                    value = H._matrix[row, col]
                    if value != 0:
                        print(f'{vertex_id:>4}: {value:+.1f}')
                """
            ),
            md('## Traverse local neighborhoods'),
            code(
                """
                print('neighbors(Glc):', sorted(H.neighbors('Glc')))
                print('successors(Glc):', sorted(H.successors('Glc')))
                print('predecessors(G6P):', sorted(H.predecessors('G6P')))
                """
            ),
            md('## Incidence matrix'),
            code(
                """
                import polars as pl

                incidence = H.ops.vertex_incidence_matrix(values=True, sparse=True)
                rows = []
                for edge_id in H.edges():
                    col = H._edges[edge_id].col_idx
                    row = {'edge': edge_id}
                    for vertex_id in sorted(H.vertices()):
                        vertex_row = H._entities[H._resolve_entity_key(vertex_id)].row_idx
                        row[vertex_id] = float(H._matrix[vertex_row, col])
                    rows.append(row)

                print('incidence shape:', incidence.shape)
                print('non-zero entries:', incidence.nnz)
                pl.DataFrame(rows)
                """
            ),
            md(
                """
                The same object supports readable graph views, local traversal,
                and incidence-level inspection. Use the level that matches the
                question you are asking.
                """
            ),
        ],
    )

    write_notebook(
        'tutos/06_multilayer.ipynb',
        [
            md(
                """
                # Multilayer

                Multilayer graphs attach vertices and edges to elementary
                layer coordinates such as condition or time.
                """
            ),
            IMPORT_ANNET,
            md('## Define layers and supra-nodes'),
            code(
                """
                M = an.AnnNet(directed=False, aspects={'condition': ['ctrl', 'stim']})
                M.add_vertices(['A', 'B'], layer=('ctrl',))
                M.add_vertices(['A', 'B'], layer=('stim',))

                print('aspects:', M.layers.list_aspects())
                print('layers:', M.layers.list_layers())
                print('global shape:', M.shape)
                """
            ),
            md('## Intra-layer and coupling edges'),
            code(
                """
                M.add_edges(('A', ('ctrl',)), ('B', ('ctrl',)), edge_id='e_ctrl', weight=1.0)
                M.add_edges(('A', ('stim',)), ('B', ('stim',)), edge_id='e_stim', weight=1.4)
                n_couplings = M.layers.add_layer_coupling_pairs([(('ctrl',), ('stim',))])

                print('coupling edges added:', n_couplings)
                M.views.edges().select(['edge_id', 'kind', 'source', 'target', 'effective_weight'])
                """
            ),
            md(
                """
                ## Draw the supra graph

                Supra-nodes include both the base vertex and its layer
                coordinate, so the drawing makes coupling edges visible.
                """
            ),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(M, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Layer selections and derived slices'),
            code(
                """
                stim_graph = M.layers.subgraph_from_layer_tuple(('stim',))
                M.layers.create_slice_from_layer('stim_only', ('stim',))

                print('stim graph shape:', stim_graph.shape)
                print('stim slice edges:', sorted(M.slices.edges('stim_only')))
                """
            ),
            md('## Draw one layer as its own graph'),
            code(
                """
                plotting.plot(stim_graph, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Supra matrices and a diffusion step'),
            code(
                """
                import numpy as np

                adjacency = M.layers.supra_adjacency()
                start = np.zeros(M.nv_supra)
                start[0] = 1.0
                diffused = M.layers.diffusion_step(start, tau=0.25)

                print('supra-adjacency shape:', adjacency.shape)
                print('supra-adjacency nnz:', adjacency.nnz)
                print('diffused mass:', round(float(diffused.sum()), 3))
                """
            ),
            md(
                """
                Multilayer AnnNet objects keep condition-specific topology,
                coupling edges, derived slices, and supra-level matrices in the
                same container.
                """
            ),
        ],
    )

    write_notebook(
        'tutos/07_history_and_reproducibility.ipynb',
        [
            md(
                """
                # History and reproducibility

                The history accessor records mutating calls, named snapshots,
                diffs, and exportable provenance.
                """
            ),
            IMPORT_ANNET,
            md('## Record construction history'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.history.clear()
                G.add_vertices(['A', 'B'])
                G.add_edges('A', 'B', edge_id='e1', weight=1.0)
                G.history.mark('baseline graph')
                G.history.snapshot('baseline')

                import polars as pl

                print('events:', len(G.history()))
                print('last op:', G.history()[-1]['op'])
                pl.DataFrame(
                    [
                        {
                            'version': event['version'],
                            'operation': event['op'],
                            'result': str(event.get('result', '')),
                        }
                        for event in G.history()
                    ]
                )
                """
            ),
            md('## Compare snapshots'),
            code(
                """
                G.add_vertices('C')
                G.add_edges('B', 'C', edge_id='e2', weight=0.5)
                G.slices.add('candidate')
                G.slices.add_edge_to_slice('candidate', 'e2')
                G.history.snapshot('candidate')

                diff = G.history.diff('baseline', 'candidate')
                import json

                print(diff.summary())
                print(json.dumps(diff.to_dict(), indent=2))
                """
            ),
            md('## Export history'),
            code(
                """
                import os
                import tempfile
                import json

                tmp = tempfile.mkdtemp(prefix='annnet_history_')
                json_path = os.path.join(tmp, 'history.json')
                n_events = G.history.export(json_path)
                with open(json_path, encoding='utf-8') as handle:
                    exported = json.load(handle)

                print('events written:', n_events)
                print('file exists:', os.path.exists(json_path))
                print(json.dumps(exported[:4], indent=2))
                """
            ),
            md(
                """
                History is not a workflow engine; it is a compact provenance
                layer. It helps notebooks explain how a graph changed and makes
                named graph states easy to compare.
                """
            ),
        ],
    )

    write_notebook(
        'tutos/08_backend_accessors.ipynb',
        [
            md(
                """
                # Backend accessors

                AnnNet can remain the source of record while dispatching to
                installed graph, dataframe, and plotting backends.
                """
            ),
            IMPORT_ANNET,
            md('## Inspect installed optional components'),
            code(
                """
                print('graph backends:', an.available_backends())
                print('dataframe backends:', an.available_dataframe_backends())
                print('plot backends:', an.available_plot_backends())
                print('selected dataframe backend:', an.select_dataframe_backend('auto'))
                """
            ),
            md('## NetworkX and igraph access'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.add_vertices(['A', 'B', 'C', 'D'])
                G.add_edges('A', 'B', edge_id='e1', weight=1.0)
                G.add_edges('A', 'B', edge_id='e1_parallel', weight=0.6)
                G.add_edges('B', 'C', edge_id='e2', weight=1.4)
                G.add_edges('C', 'D', edge_id='complex_cd', directed=False, weight=0.8)

                nx_graph = G.nx.backend()
                ig_graph, manifest = an.to_igraph(G)
                restored = an.from_igraph(ig_graph, manifest)

                print(type(nx_graph).__name__, sorted(nx_graph.nodes()))
                print('NetworkX edges:', list(nx_graph.edges(data=True)))
                print('igraph:', ig_graph.vcount(), 'vertices,', ig_graph.ecount(), 'edges')
                print('igraph edge list:', ig_graph.get_edgelist())
                print('round-trip:', restored.shape)
                """
            ),
            md('## Draw the AnnNet source graph'),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(G, backend='graphviz', show_edge_labels=True)
                """
            ),
            md(
                """
                ## Optional backend accessors

                The same accessor pattern applies to other installed graph
                backends. AnnNet keeps the annotated graph state and projects it
                when another library has the algorithm or file format you need.
                """
            ),
            md(
                """
                Backend accessors are execution targets, not replacement graph
                stores. AnnNet keeps the annotated graph state and projects it
                when another library has the algorithm or file format you need.
                """
            ),
        ],
    )


def generate_scenarios() -> None:
    write_notebook(
        'scenarios/scverse_bridge.ipynb',
        [
            md(
                """
                # AnnData/scverse bridge

                This scenario shows the AnnNet/AnnData boundary on a tiny graph.
                """
            ),
            INTRO_WITH_INFO,
            md('## Build an annotated graph'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.add_vertices('cell_A', kind='cell', score=1.0, cluster=0)
                G.add_vertices('cell_B', kind='cell', score=2.0, cluster=1)
                G.add_vertices('cell_C', kind='cell', score=1.6, cluster=1)
                G.add_edges('cell_A', 'cell_B', edge_id='transition_ab', weight=0.6)
                G.add_edges('cell_A', 'cell_B', edge_id='velocity_ab', weight=0.35)
                G.add_edges('cell_B', 'cell_C', edge_id='transition_bc', weight=0.9)
                G.add_edges('cell_B', 'cell_C', edge_id='neighborhood_bc', directed=False, weight=0.5)
                G.attrs.set_edge_attrs('transition_ab', relation='state_transition')
                G.attrs.set_edge_attrs('velocity_ab', relation='rna_velocity')

                G.views.vertices()
                """
            ),
            md('## Draw the graph before conversion'),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(G, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Convert to AnnData and inspect the tables'),
            code(
                """
                from annnet.experimental.scverse import to_anndata, from_anndata

                adata = to_anndata(G)
                restored = from_anndata(adata)

                print('AnnData shape:', adata.shape)
                print('uns keys:', sorted(adata.uns))
                print('round-trip shape:', restored.shape)
                display(adata.obs)
                display(adata.var)
                """
            ),
            md(
                """
                AnnNet maps graph vertices to `obs`, structural edges to `var`,
                and incidence to `X`. AnnNet-specific structure is preserved
                in `uns`, so the round trip can recover parallel and undirected
                graph structure.
                """
            ),
        ],
    )

    write_notebook(
        'scenarios/omnipath_table_ingestion.ipynb',
        [
            md(
                """
                # OmniPath table ingestion

                `from_omnipath` can consume an OmniPath-style interaction table
                directly. This notebook uses a local table so it is deterministic.
                """
            ),
            INTRO_WITH_INFO,
            md('## Build from a local OmniPath-style table'),
            code(
                """
                import polars as pl

                interactions = pl.DataFrame(
                    {
                        'source': ['EGF', 'EGFR', 'EGFR', 'EGFR', 'RAS', 'MEK'],
                        'target': ['EGFR', 'RAS', 'RAS', 'GRB2', 'MEK', 'ERK'],
                        'interaction_id': [
                            'EGF_EGFR',
                            'EGFR_RAS_primary',
                            'EGFR_RAS_secondary',
                            'EGFR_GRB2_complex',
                            'RAS_MEK',
                            'MEK_ERK',
                        ],
                        'is_directed': [True, True, True, False, True, True],
                        'curation_score': [0.95, 0.88, 0.63, 0.76, 0.82, 0.79],
                        'consensus_direction': [1, 1, 1, 0, 1, 1],
                        'source_database': [
                            'omnipath',
                            'omnipath',
                            'literature',
                            'complexportal',
                            'pathwayextra',
                            'kinaseextra',
                        ],
                    }
                )

                G = an.from_omnipath(
                    interactions,
                    source_col='source',
                    target_col='target',
                    edge_id_col='interaction_id',
                    directed_col='is_directed',
                    weight_col='curation_score',
                    edge_attr_cols=['consensus_direction', 'source_database'],
                    load_vertex_annotations=False,
                )

                print('shape:', G.shape)
                G.views.edges().select(
                    ['edge_id', 'source', 'target', 'effective_weight', 'source_database']
                )
                """
            ),
            md('## Add analysis context as slices'),
            code(
                """
                rows = list(G.views.edges().iter_rows(named=True))
                edge_label = {
                    row['edge_id']: f"{row['source']} -> {row['target']}"
                    for row in rows
                }
                high_confidence = [
                    row['edge_id']
                    for row in rows
                    if row['effective_weight'] >= 0.85
                ]
                G.slices.add('high_confidence')
                G.slices.add_edges('high_confidence', high_confidence)

                print(
                    'high-confidence interactions:',
                    [edge_label[eid] for eid in sorted(G.slices.edges('high_confidence'))],
                )
                """
            ),
            md('## Draw the prior network'),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(
                    G,
                    backend='graphviz',
                    show_edge_labels=True,
                    edge_label_keys=['source_database'],
                )
                """
            ),
            md(
                """
                AnnNet can use OmniPath-style tables as prior knowledge while
                keeping confidence, provenance, and downstream contexts in one
                graph object.
                """
            ),
        ],
    )

    write_notebook(
        'scenarios/cytoscape_cx2_export.ipynb',
        [
            md(
                """
                # Cytoscape CX2 export

                CX2 is useful when a graph should be inspected in Cytoscape.
                AnnNet can choose how hyperedges are projected for tools that
                do not support them directly.
                """
            ),
            INTRO_WITH_INFO,
            md('## Build a graph with a hyperedge'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.add_vertices(['A', 'B', 'C', 'D', 'E'])
                G.add_edges('A', 'B', edge_id='binary_edge', weight=1.0)
                G.add_edges('A', 'B', edge_id='parallel_edge', weight=0.65)
                G.add_edges('B', 'C', edge_id='complex_edge', directed=False, weight=0.8)
                G.add_edges(src=['B', 'C'], tgt=['D', 'E'], edge_id='reaction', directed=True, weight=2.0)
                G.attrs.set_vertex_attrs_bulk(
                    {
                        'A': {'label': 'ligand'},
                        'B': {'label': 'enzyme'},
                        'C': {'label': 'cofactor'},
                        'D': {'label': 'product'},
                        'E': {'label': 'byproduct'},
                    }
                )

                G.views.edges().select(['edge_id', 'kind', 'source', 'target', 'head', 'tail'])
                """
            ),
            md(
                """
                ## Preview the network

                The CX2 file is for Cytoscape. For static documentation, a
                Graphviz preview gives a lightweight embedded view of the same
                topology.
                """
            ),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(G, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Compare hyperedge export modes'),
            code(
                """
                for mode in ['skip', 'expand', 'reify']:
                    cx2 = an.to_cx2(G, hyperedges=mode, export_name=f'annnet-{mode}')
                    edge_aspect = next(item['edges'] for item in cx2 if 'edges' in item)
                    node_aspect = next(item['nodes'] for item in cx2 if 'nodes' in item)
                    print(mode, 'nodes:', len(node_aspect), 'edges:', len(edge_aspect))
                """
            ),
            md('## Round-trip through the CX2 manifest'),
            code(
                """
                import json

                cx2 = an.to_cx2(G, hyperedges='reify', export_name='annnet-reified')
                restored = an.from_cx2(cx2)

                print('restored shape:', restored.shape)
                print('restored hyperedges:', sorted(restored.hyperedge_definitions))
                print(json.dumps(cx2[:3], indent=2))
                """
            ),
            md(
                """
                `skip`, `expand`, and `reify` are explicit choices about what
                Cytoscape should see. A true embedded Cytoscape.js widget would
                need an additional browser asset; this notebook keeps the docs
                static by showing the CX2 payload plus a Graphviz preview.
                """
            ),
        ],
    )

    write_notebook(
        'scenarios/pyg_heterodata_export.ipynb',
        [
            md(
                """
                # PyG HeteroData export

                `to_pyg` exports an AnnNet graph to PyTorch Geometric
                `HeteroData` for graph neural network workflows.
                """
            ),
            INTRO_WITH_INFO,
            md('## Build a heterogeneous graph'),
            code(
                """
                G = an.AnnNet(directed=True)
                G.add_vertices('p1', kind='protein', activity=1.2, abundance=4.0)
                G.add_vertices('p2', kind='protein', activity=-0.4, abundance=2.5)
                G.add_vertices('g1', kind='gene', expression=8.0, length=1200.0)
                G.add_vertices('drug_a', kind='drug', dosage=10.0, approved=1.0)

                G.add_edges('p1', 'g1', edge_id='regulates', weight=0.8)
                G.add_edges('p2', 'g1', edge_id='represses', weight=-0.6)
                G.add_edges('drug_a', 'p1', edge_id='targets', weight=0.5)
                G.add_edges('drug_a', 'p2', edge_id='off_target', weight=0.25)
                G.slices.add('train')
                G.slices.add_edges('train', ['regulates', 'represses'])

                G.views.vertices()
                """
            ),
            md('## Draw the source graph'),
            code(
                """
                from annnet.utils import plotting

                plotting.plot(G, backend='graphviz', show_edge_labels=True)
                """
            ),
            md('## Export to PyG and run a tensor operation'),
            code(
                """
                import torch

                data = an.to_pyg(
                    G,
                    node_features={
                        'protein': ['activity', 'abundance'],
                        'gene': ['expression', 'length'],
                        'drug': ['dosage', 'approved'],
                    },
                    slice_id='train',
                    hyperedge_mode='skip',
                )
                data.validate(raise_on_error=True)

                protein_activity = data['protein'].x[:, 0]
                activity_probability = torch.sigmoid(protein_activity)
                homogeneous = data.to_homogeneous()

                print(data)
                print('protein activity:', protein_activity.tolist())
                print('sigmoid(activity):', activity_probability.tolist())
                print('homogeneous edge_index shape:', tuple(homogeneous.edge_index.shape))
                """
            ),
            md(
                """
                AnnNet remains useful before tensorization: it stores names,
                metadata, slices, and graph semantics. `to_pyg` is the boundary
                where selected numeric attributes become tensors.
                """
            ),
        ],
    )

    write_notebook(
        'scenarios/causal_activity_bridge.ipynb',
        [
            md(
                """
                # Causal activity bridge

                CORNETO solves a small CARNIVAL signaling problem. AnnNet then
                stores the selected signal edges, node activities, and a
                solution slice next to the original prior network.
                """
            ),
            INTRO_WITH_INFO,
            md('## Build the CORNETO prior and measurements'),
            code(
                """
                import corneto as cn
                from corneto.graph import Graph

                pkn_tuples = [
                    ('rec1', 1, 'a'),
                    ('rec1', -1, 'b'),
                    ('rec1', 1, 'f'),
                    ('rec1', -1, 'c'),
                    ('rec2', 1, 'b'),
                    ('rec2', 1, 'tf2'),
                    ('b', 1, 'g'),
                    ('g', -1, 'd'),
                    ('rec2', -1, 'd'),
                    ('a', 1, 'c'),
                    ('a', -1, 'd'),
                    ('c', 1, 'd'),
                    ('c', -1, 'e'),
                    ('c', 1, 'tf3'),
                    ('e', 1, 'a'),
                    ('d', -1, 'c'),
                    ('e', 1, 'tf1'),
                    ('a', -1, 'tf1'),
                    ('d', 1, 'tf2'),
                    ('c', -1, 'tf2'),
                    ('tf1', 1, 'tf2'),
                    ('tf1', -1, 'rec2'),
                    ('tf2', 1, 'rec1'),
                    ('tf1', 1, 'f'),
                ]
                corneto_graph = Graph.from_tuples(pkn_tuples)
                samples = {
                    'input_example': {
                        'rec2': {'value': 1, 'mapping': 'vertex', 'role': 'input'},
                        'tf1': {'value': -2, 'mapping': 'vertex', 'role': 'output'},
                        'tf2': {'value': 1, 'mapping': 'vertex', 'role': 'output'},
                    }
                }
                data = cn.Data.from_cdict(samples)

                print(corneto_graph)
                print(data)
                """
            ),
            md('## Solve CARNIVAL and inspect the selected signal'),
            code(
                """
                import numpy as np
                import pandas as pd
                import polars as pl
                from corneto.methods.future.carnival import CarnivalFlow

                model = CarnivalFlow(lambda_reg=1e-3)
                problem = model.build(corneto_graph, data)
                problem.solve(verbosity=0, solver='scipy')

                edge_values = pd.DataFrame(
                    problem.expr.edge_value.value,
                    index=model.processed_graph.E,
                    columns=['edge_activity'],
                ).astype(int)
                vertex_values = pd.DataFrame(
                    problem.expr.vertex_value.value,
                    index=model.processed_graph.V,
                    columns=['node_activity'],
                ).astype(int)
                selected_idx = np.flatnonzero(problem.expr.edge_has_signal.value)

                edge_rows = []
                for edge_obj, row in edge_values.iterrows():
                    source, target = edge_obj
                    activity = int(row['edge_activity'])
                    if activity == 0:
                        continue
                    edge_rows.append(
                        {
                            'edge': f"{', '.join(sorted(source)) or '(input)'} -> "
                            f"{', '.join(sorted(target)) or '(output)'}",
                            'activity': activity,
                        }
                    )

                print('objective values:', [float(obj.value) for obj in problem.objectives])
                display(pl.DataFrame(edge_rows))
                display(
                    pl.DataFrame(
                        [
                            {'node': str(node), 'activity': int(row['node_activity'])}
                            for node, row in vertex_values.iterrows()
                            if int(row['node_activity']) != 0
                        ]
                    )
                )
                """
            ),
            md('## CORNETO selected subgraph'),
            code(
                """
                model.processed_graph.edge_subgraph(selected_idx).plot()
                """
            ),
            md('## Store the CORNETO result in AnnNet'),
            code(
                """
                G = an.AnnNet(directed=True)
                edge_ids_by_pair = {}
                for source, sign, target in pkn_tuples:
                    effect = 'activates' if sign > 0 else 'inhibits'
                    edge_id = f'{source}_{effect}_{target}'
                    G.add_edges(
                        source,
                        target,
                        edge_id=edge_id,
                        weight=abs(sign),
                        interaction=sign,
                        effect=effect,
                    )
                    edge_ids_by_pair[(source, target)] = edge_id

                selected_edges = []
                for edge_obj, row in edge_values.iterrows():
                    source, target = edge_obj
                    activity = int(row['edge_activity'])
                    if activity == 0 or len(source) != 1 or len(target) != 1:
                        continue
                    pair = (next(iter(source)), next(iter(target)))
                    edge_id = edge_ids_by_pair.get(pair)
                    if edge_id is None:
                        continue
                    selected_edges.append(edge_id)
                    G.attrs.set_edge_attrs(edge_id, corneto_activity=activity)

                for node, row in vertex_values.iterrows():
                    if str(node) in G.vertices():
                        G.attrs.set_vertex_attrs(str(node), corneto_activity=int(row['node_activity']))

                G.slices.add('corneto_signal')
                G.slices.add_edges('corneto_signal', selected_edges)

                selected_labels = {
                    row['edge_id']: f"{row['source']} -> {row['target']}"
                    for row in G.views.edges().iter_rows(named=True)
                }
                print('AnnNet shape:', G.shape)
                print('selected edges:', [selected_labels[eid] for eid in sorted(selected_edges)])
                G.views.edges().select(
                    ['edge_id', 'source', 'target', 'effect', 'corneto_activity']
                )
                """
            ),
            md('## AnnNet solution slice'),
            code(
                """
                from annnet.utils import plotting

                solution = G.subgraph_from_slice('corneto_signal')
                plotting.plot(solution, backend='graphviz', show_edge_labels=True)
                """
            ),
            md(
                """
                The CORNETO solve remains reproducible in the notebook, while
                AnnNet keeps the result attached to a graph object that can be
                sliced, annotated, exported, or combined with other analyses.
                """
            ),
        ],
    )


def write_envs() -> None:
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    envs = {
        NOTEBOOK_DIR / 'environment.yml': [
            'name: annnet-docs-howtos',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.10,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
            '      - matplotlib>=3.5',
            '      - mkdocs>=1.6',
            '      - mkdocs-autorefs>=1.0',
            '      - mkdocs-jupyter>=0.25',
            '      - mkdocs-material>=9.5',
            '      - mkdocstrings[python]>=0.25',
            '      - nbconvert>=7',
            '      - networkx>=3.3',
            '      - numcodecs>=0.12',
            '      - openpyxl>=3.1',
            '      - pandas>=2.0',
            '      - polars>=0.20.0',
            '      - pyarrow>=10',
            '      - python-igraph>=0.11',
            '      - zarr>=2.18',
        ],
        ENV_DIR / 'scverse_bridge.yml': [
            'name: annnet-scenario-scverse',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.11,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
            '      - anndata>=0.12',
            '      - mudata>=0.3',
            '      - spatialdata>=0.2',
            '      - pandas>=2.0',
        ],
        ENV_DIR / 'omnipath_table_ingestion.yml': [
            'name: annnet-scenario-omnipath',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.10,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
            '      - omnipath>=1.0',
            '      - polars>=0.20.0',
        ],
        ENV_DIR / 'cytoscape_cx2_export.yml': [
            'name: annnet-scenario-cx2',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.10,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
        ],
        ENV_DIR / 'pyg_heterodata_export.yml': [
            'name: annnet-scenario-pyg',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.10,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
            '      - ipywidgets>=8',
            '      - torch>=2.2',
            '      - torch-geometric>=2.5',
        ],
        ENV_DIR / 'causal_activity_bridge.yml': [
            'name: annnet-scenario-causal-activity',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.11,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
            '      - corneto',
            '      - pandas>=2.0',
            '      - polars>=0.20.0',
            '      - scipy',
        ],
        ENV_DIR / 'uc1_multi_condition_causal_signaling.yml': [
            'name: annnet-use-case-uc1-causal-signaling',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.11,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
            '      - matplotlib>=3.5',
            '      - pandas>=2.0',
            '      - scipy',
            '      - cptac',
            '      - decoupler',
            '      - corneto',
            '      - torch>=2.2',
            '      - torch-geometric>=2.5',
            '      - scikit-learn',
        ],
        ENV_DIR / 'uc2_hek293_heterogeneous_graph.yml': [
            'name: annnet-use-case-uc2-hek293-heterogeneous-graph',
            'channels:',
            '  - conda-forge',
            'dependencies:',
            '  - python>=3.11,<3.14',
            '  - pip',
            '  - pip:',
            '      - annnet @ git+https://github.com/saezlab/annnet.git',
            '      - graphviz',
            '      - ipykernel>=6',
            '      - matplotlib>=3.5',
            '      - numpy>=1.24',
            '      - omnipath>=1.0',
            '      - pandas>=2.0',
            '      - polars>=0.20.0',
            '      - requests',
            '      - scipy',
            '      - lxml>=4.9',
            '      - torch>=2.2',
            '      - torch-geometric>=2.5',
            '      - scikit-learn',
        ],
    }
    for path, lines in envs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(lines) + '\n')


def cleanup_obsolete() -> None:
    obsolete = [
        'Demo.ipynb',
        'SBUC-nopolars.ipynb',
        'SBUC.ipynb',
        'UC1.ipynb',
        'UC2.ipynb',
        'annnet_showcase.ipynb',
        'tutos/00_tutorial_index.ipynb',
        'tutos/03_io_and_interop.ipynb',
        'tutos/04_slices.ipynb',
        'tutos/05_hyperedges.ipynb',
    ]
    for relative in obsolete:
        path = NOTEBOOK_DIR / relative
        if path.exists():
            path.unlink()


def main() -> None:
    TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_obsolete()
    generate_tutorials()
    generate_scenarios()
    write_envs()


if __name__ == '__main__':
    main()
