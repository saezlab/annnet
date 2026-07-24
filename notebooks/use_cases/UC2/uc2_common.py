"""Shared constants, helpers and snapshot IO for the UC2 notebook series.

Importing ``from uc2_common import *`` gives every notebook the aspect model,
the small unwrap/sign helpers, the output paths, and ``load`` / ``edges_frame``
so the notebooks themselves stay about the biology, not the plumbing.
"""

from pathlib import Path
from itertools import combinations  # noqa: F401 (re-exported for notebooks)

import annnet as an

SEED = 7

# Two Kivela aspects: functional mechanism x structural assembly state.
ASPECTS = ['mechanism', 'complex']
MECHANISMS = ('signaling', 'metabolic', 'regulatory')
MECH_SIGNALING, MECH_METABOLIC, MECH_REGULATORY = MECHANISMS
CPLX_MEMBER, CPLX_MONOMER = 'member', 'monomer'
ELEM_LAYERS = {'mechanism': list(MECHANISMS), 'complex': [CPLX_MEMBER, CPLX_MONOMER]}
GENE_COORD = (MECH_REGULATORY, CPLX_MONOMER)
METAB_COORD = (MECH_METABOLIC, CPLX_MONOMER)
KIND_PROTEIN, KIND_GENE, KIND_METABOLITE = 'protein', 'gene', 'metabolite'

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

# Paths, relative to a notebook running inside UC2/.
DATA, OUT = Path('data'), Path('outputs')
TABLES, FIGS = OUT / 'tables', OUT / 'figures'
for _p in (DATA, TABLES, FIGS):
    _p.mkdir(parents=True, exist_ok=True)
SNAPSHOT = DATA / 'uc2.annnet'
HEK293 = DATA / 'hek293.parquet'
HUMANGEM = DATA / 'Human-GEM.xml'


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
