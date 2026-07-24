"""Shared constants, helpers and snapshot IO for the UC2b notebook series.

UC2b is the time-resolved sibling of UC2. It keeps UC2's heterogeneous multi-source
build (OmniPath signaling, DoRothEA regulation, OmniPath complexes as hyperedges,
Human-GEM metabolism) but swaps the second Kivela aspect from *structure*
(``complex`` = member/monomer) to *time*. The time course comes from the Saez-lab
kidney-fibrosis multi-omics study (human kidney PDGFRb+ mesenchymal cells,
TGF-beta stimulation, seven timepoints).

Both aspects now carry biological meaning:

- ``mechanism`` in {signaling, regulatory, metabolic}: the process an edge belongs to.
- ``time`` in {0h, 1h, 12h, 24h, 48h, 72h, 96h}: when an entity is *responsive*.

A node-layer is therefore e.g. ``(prot:COL1A1, signaling, 24h)``. Each time layer
holds only the entities that respond at that timepoint (pure responsive
restriction, adj.P < 0.05, gene-symbol union across all modalities); the layers
genuinely differ and grow as the response spreads. ``0h`` is the baseline that
carries the time-invariant scaffold (metabolic reactions, complexes).

``from uc2b_common import *`` gives every notebook the aspect model, the small
helpers, the output paths, and ``load`` / ``edges_frame`` so the notebooks stay
about the biology, not the plumbing.
"""

from pathlib import Path
from itertools import combinations  # noqa: F401 (re-exported for notebooks)

import annnet as an

SEED = 7

# --- The two Kivela aspects: functional mechanism x response time -------------
ASPECTS = ['mechanism', 'time']
MECHANISMS = ('signaling', 'regulatory', 'metabolic')
MECH_SIGNALING, MECH_REGULATORY, MECH_METABOLIC = MECHANISMS

BASELINE = '0h'  # time-invariant scaffold layer
TIMES = ('1h', '12h', '24h', '48h', '72h', '96h')  # responsive dynamics
ALL_TIMES = (BASELINE, *TIMES)
ELEM_LAYERS = {'mechanism': list(MECHANISMS), 'time': list(ALL_TIMES)}

# Coordinates used repeatedly in the build.
GENE_COORD = lambda t: (MECH_REGULATORY, t)
PROT_COORD = lambda t: (MECH_SIGNALING, t)
METAB_COORD = (MECH_METABOLIC, BASELINE)
KIND_PROTEIN, KIND_GENE, KIND_METABOLITE = 'protein', 'gene', 'metabolite'

# Significance gate for "responsive" (matches the sizing analysis; no logFC floor
# so the early layers stay populated). ``strong`` marks the |logFC|>1 subset.
ADJP_THRESHOLD = 0.05
STRONG_LOGFC = 1.0

# HPA subcellular location -> (GEM compartment code, organelle slice name).
COMPARTMENTS = {
    'Mitochondria': ('m', 'mitochondria'),
    'Nucleus': ('n', 'nucleus'),
    'Nucleoplasm': ('n', 'nucleus'),
    'Nucleoli': ('n', 'nucleus'),
    'Nuclear membrane': ('n', 'nucleus'),
    'Endoplasmic reticulum': ('r', 'er'),
    'Golgi apparatus': ('g', 'golgi'),
    'Lysosome': ('l', 'lysosome'),
    'Cytosol': ('c', 'cytosol'),
    'Cytoplasm': ('c', 'cytosol'),
    'Plasma membrane': ('p', 'plasma_membrane'),
    'Cell Junctions': ('p', 'plasma_membrane'),
    'Peroxisome': ('x', 'peroxisome'),
    'Vesicles': ('v', 'vesicles'),
    'Extracellular': ('e', 'extracellular'),
}
CODE_TO_NAME = dict(COMPARTMENTS.values())

# Paths, relative to a notebook running inside UC2b/.
DATA, OUT = Path('data'), Path('outputs')
TABLES, FIGS = OUT / 'tables', OUT / 'figures'
for _p in (DATA, TABLES, FIGS):
    _p.mkdir(parents=True, exist_ok=True)
SNAPSHOT = DATA / 'uc2b.annnet'
HUMANGEM = DATA / 'Human-GEM.xml'

# Kidney-fibrosis source (RData) and the derived, R-free caches 01 writes.
DIFF_RDATA = DATA / '2024-08-15_diff_results.RData'
NETWORK_RDATA = DATA / '2024-08-16_res_network.RData'  # CARNIVAL early/late ground truth
RESPONSIVE = DATA / 'responsive.parquet'  # long: symbol, time, best_logFC, min_adjp, strong
MEASURED = DATA / 'measured.parquet'  # symbol, modality  (the context universe)
CARNIVAL = DATA / 'carnival_network.parquet'  # network(early/late), source, target, sign_num

# Coarse phase mapping for the CARNIVAL early/late comparison in 05.
EARLY_TIMES, LATE_TIMES = ('1h', '12h', '24h'), ('48h', '72h', '96h')


def symbol_of(feature_id):
    """Gene symbol from a diff_results feature_id.

    rna / proteomics / secretomics ids are bare symbols; phospho ids look like
    ``SYMBOL_PEPTIDE___n_SITE``, so the symbol is the first underscore field.
    """
    return feature_id.split('_', 1)[0] if isinstance(feature_id, str) else feature_id


def compartment_code(loc):
    """Map an HPA location string to a single-letter GEM compartment code."""
    for key, (c, _) in COMPARTMENTS.items():
        if loc and key.lower() in loc.lower():
            return c
    return 'c'


def sign_of(stim, inhib):
    """Signed edge weight from stimulation / inhibition flags."""
    return 1.0 if stim and not inhib else (-1.0 if inhib and not stim else 0.0)


def bare_vid(e):
    """Reduce a ``(vid, coord)`` supra-node key (or its repr) to the bare vid."""
    if isinstance(e, tuple):
        return e[0]
    if isinstance(e, str) and e.startswith("('"):
        j = e.find("',", 2)
        return e[2:j] if j > 0 else e
    return e


def load():
    """Reload the graph saved by 02_build.ipynb."""
    return an.AnnNet.read(str(SNAPSHOT))


def edges_frame(G):
    """Edge table with both endpoints reduced to bare vertex ids."""
    df = G.views.edges().to_pandas()
    df['source'] = df['source'].map(bare_vid)
    df['target'] = df['target'].map(bare_vid)
    return df


def responsive_by_time(resp_df):
    """{time: set(symbol)} from the responsive parquet, for the layer gate."""
    return {t: set(g['symbol']) for t, g in resp_df.groupby('time', observed=True)}


def complex_subunits(G, min_size=1):
    """{complex_id: set of subunit gene symbols} from the complex hyperedges."""
    out = {}
    for eid, spec in G.hyperedge_definitions.items():
        if not str(eid).startswith('cpx:'):
            continue
        genes = {
            bare_vid(m)[len('prot:') :]
            for m in (spec.get('members') or [])
            if str(bare_vid(m)).startswith('prot:')
        }
        if len(genes) >= min_size:
            out[eid] = genes
    return out
