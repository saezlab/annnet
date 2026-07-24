"""Shared config, paths, and cross-notebook machinery for the UC1 series.

``from uc1_common import *`` gives every notebook the knobs, the output paths,
``load`` for the shared graph snapshot, and the consensus builder — so each
notebook stays about the biology, not the plumbing. State that does not live in
the graph (omics contrasts, activity matrices, CARNIVAL solutions, cohorts) is
handed between notebooks through the artifact paths defined below.
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import annnet as an

SEED = 7

# --- Cohorts -----------------------------------------------------------------
TOP_MUT_QUANTILES = {'top10': 0.90, 'top20': 0.80}
PRIMARY_COHORT_LABEL = 'top10'
SENSITIVITY_COHORT_LABELS = ['top10', 'top20']
INCLUDE_ALL_MATCHED_COHORT = True

# --- Activity inference / CARNIVAL inputs ------------------------------------
KIN_TH = TF_TH = 1.5
TOP_K_KIN, TOP_TF_POS, TOP_TF_NEG = 10, 8, 7
ULM_TMIN_CANDIDATES = [5, 3, 1]

# --- CARNIVAL solve ----------------------------------------------------------
REGULARIZATION_GRID = [0.0, 0.01, 0.1]
PRIMARY_LAMBDA = 0.01
SOLVER_CANDIDATES = ['scip', 'highs', 'cbc', 'glpk']
ACCEPTABLE_SOLVER_STATUSES = {
    'optimal',
    'optimal_inaccurate',
    'feasible',
    'feasible_inaccurate',
}

# --- Consensus / reporting ---------------------------------------------------
CONSENSUS_MIN_FREQ = 0.50
CORNETO_SIG_THRESHOLD = 0.50
NULL_REPEATS = 100
TOP_N_PLOT = 50

PSP_URL = 'https://omnipathdb.org/enz_sub?genesymbols=1'

# --- Paths, relative to a notebook running inside UC1/ ------------------------
DATA, OUT = Path('data'), Path('outputs')
TABLES, FIGS, CACHE = OUT / 'tables', OUT / 'figures', OUT / 'cache'
for _p in (DATA, TABLES, FIGS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

SNAPSHOT = DATA / 'uc1.annnet'  # the shared graph, re-saved per stage
RNA_CONTRAST = DATA / 'rna_contrast.parquet'
TF_ES = DATA / 'tf_activity.parquet'
KIN_ES = DATA / 'kinase_activity.parquet'
MUT_MAT = DATA / 'mutation_matrix.parquet'
CARNIVAL_PKL = DATA / 'carnival_networks.pkl'
CONSENSUS_PKL = DATA / 'consensus_outputs.pkl'
COHORTS_JSON = DATA / 'cohorts.json'
PYG_OUT = OUT / 'UC1_heterodata.pt'
HISTORY_OUT = OUT / 'UC1_history.json'


def load():
    """Reload the shared graph snapshot with history recording live."""
    G = an.AnnNet.read(str(SNAPSHOT))
    G.history.enable(True)
    return G


def pkn_edges(G):
    """Edge table of the loaded PKN, dropping rows with a missing endpoint."""
    df = G.views.edges().to_pandas()
    return df[df['source'].notna() & df['target'].notna()].copy()


def pkn_degree(edges_df):
    """Total (in+out) degree per vertex over an edge table."""
    return (
        pd.concat([edges_df['source'].value_counts(), edges_df['target'].value_counts()], axis=1)
        .fillna(0)
        .sum(axis=1)
        .astype(int)
        .rename('pkn_degree')
        .sort_values(ascending=False)
    )


def bare_vid(name):
    """Reduce a ``(vid, layer_coord)`` supra-node name to the bare vid."""
    if isinstance(name, tuple):
        return name[0]
    return name


def save_carnival(networks):
    CARNIVAL_PKL.write_bytes(pickle.dumps(networks))


def load_carnival():
    return pickle.loads(CARNIVAL_PKL.read_bytes())


def save_cohorts(cohorts):
    COHORTS_JSON.write_text(json.dumps(cohorts))


def load_cohorts():
    return json.loads(COHORTS_JSON.read_text())


# --- Consensus machinery (shared by 05_consensus and 06_validation) ----------


def get_patient_active_nodes(result):
    seen, out = set(), []
    for value in result.get('node_signal', {}):
        value = str(bare_vid(value))
        if value not in seen:
            seen.add(value)
            out.append(value)
    return set(out)


def get_patient_active_edges(result):
    return {(str(s), str(t)) for s, t in result.get('edge_signal', {})}


def build_consensus_layer(patient_results, min_freq):
    """Aggregate per-patient active nodes/edges into consensus tables.

    Counts come from per-patient sets, so every count is in ``[0, n]`` and every
    frequency in ``[0, 1]`` by construction. Returns (node_df, edge_df, summary).
    """
    if not patient_results:
        raise ValueError('patient_results is empty')

    n = len(patient_results)
    node_count, edge_count = Counter(), Counter()
    node_signal, edge_signal = defaultdict(list), defaultdict(list)
    union_nodes, union_edges = set(), set()

    for result in patient_results.values():
        nodes, edges = get_patient_active_nodes(result), get_patient_active_edges(result)
        union_nodes |= nodes
        union_edges |= edges
        for v in nodes:
            node_count[v] += 1
            node_signal[v].append(float(result['node_signal'][v]))
        for e in edges:
            edge_count[e] += 1
            edge_signal[e].append(float(result['edge_signal'][e]))

    node_df = pd.DataFrame(
        {
            'vertex_id': v,
            'active_count': c,
            'patient_frequency': c / n,
            'mean_signal': float(np.mean(node_signal[v])),
            'median_signal': float(np.median(node_signal[v])),
            'selected_for_consensus': bool(c / n >= min_freq),
        }
        for v, c in sorted(node_count.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    edge_df = pd.DataFrame(
        {
            'source': s,
            'target': t,
            'active_count': c,
            'patient_frequency': c / n,
            'mean_signal': float(np.mean(edge_signal[(s, t)])),
            'median_signal': float(np.median(edge_signal[(s, t)])),
            'selected_for_consensus': bool(c / n >= min_freq),
        }
        for (s, t), c in sorted(edge_count.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    summary = {
        'n_patients': n,
        'n_union_nodes': len(union_nodes),
        'n_union_edges': len(union_edges),
        'n_consensus_nodes': int(node_df['selected_for_consensus'].sum())
        if not node_df.empty
        else 0,
        'n_consensus_edges': int(edge_df['selected_for_consensus'].sum())
        if not edge_df.empty
        else 0,
    }
    return node_df, edge_df, summary


def add_consensus_layer_to_graph(G, layer_label, node_df, edge_df):
    """Materialize a consensus layer as a new elementary layer on ``G``."""
    if layer_label not in G.layers.elem_layers['patient']:
        G.layers.add_elementary_layer('patient', layer_label)
    aa = (layer_label,)

    nodes = node_df[node_df['selected_for_consensus']]
    edges = edge_df[edge_df['selected_for_consensus']]
    if not nodes.empty:
        G.add_vertices(nodes['vertex_id'].tolist(), layer=aa)
        for row in nodes.itertuples(index=False):
            G.layers.set_vertex_layer_attrs(
                row.vertex_id,
                aa,
                corneto_signal=float(row.mean_signal),
                patient_frequency=float(row.patient_frequency),
                active_count=int(row.active_count),
            )
    if not edges.empty:
        keep = set(nodes['vertex_id'])
        specs = [
            {
                'source': (row.source, aa),
                'target': (row.target, aa),
                'weight': float(row.mean_signal),
            }
            for row in edges.itertuples(index=False)
            if row.source in keep and row.target in keep
        ]
        if specs:
            G.add_edges(specs)
    return aa
