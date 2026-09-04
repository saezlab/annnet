"""The core holds no biology.

Constitution Principle IV and FR-033: the core is a general network data
structure. Biology belongs in the documentation, where it makes an example
concrete, and in the client packages that reach a knowledge base. A core that
names a gene has a shape that fits one field, and every other field then reads
its own vocabulary as a foreign one.

The gate reads the source of `annnet/core` and fails on a word from the list
below. A word that a general reader would use anyway — "cell" of a matrix, or a
"tissue" of a mesh — is not in the list, because a gate that cries wolf is one
that gets suppressed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import annnet

CORE = Path(annnet.__file__).parent / 'core'
PROJECT = Path(annnet.__file__).parent.parent

# One word per line, and each one names a thing that exists in biology and
# nowhere else in the vocabulary a network data structure needs.
BIOLOGY_WORDS = (
    'gene',
    'genesymbol',
    'protein',
    'proteins',
    'uniprot',
    'ensembl',
    'entrez',
    'kinase',
    'phosphatase',
    'ligand',
    'receptor',
    'metabolite',
    'metabolites',
    'metabolic',
    'transcription',
    'transcriptomic',
    'regulon',
    'pathway',
    'pathways',
    'enzyme',
    'reaction',
    'reactions',
    'stoichiometry',
    'organism',
    'taxon',
    'mirna',
    'lncrna',
    'chebi',
    'omics',
    'proteomic',
    'signaling',
    'signalling',
    'biological',
    'biology',
    'phospho',
)

PATTERN = re.compile(r'(?<![A-Za-z])(' + '|'.join(BIOLOGY_WORDS) + r')(?![A-Za-z])', re.IGNORECASE)


#: The parts of the package a general network data structure is made of. `io`
#: and `_support` are here for the same reason `core` is: a reader that names a
#: gene has a shape that fits one field, and a format is not a domain.
MECHANISM = ('core', 'io', '_support')


#: Readers for formats that are themselves domain formats. SBML is a systems
#: biology exchange format and CX2 carries a domain schema; a reader for one
#: names what the format names, and refusing that would be the gate crying wolf.
#: The exemption is per *file* and deliberately short — a new name here is a
#: decision, not a convenience.
EXEMPT = frozenset({'sbml.py', 'sbml_cobra.py', 'cx2.py'})


def core_files() -> list[Path]:
    root = Path(annnet.__file__).parent
    return sorted(
        path
        for part in MECHANISM
        for path in (root / part).rglob('*.py')
        if path.name not in EXEMPT
    )


def test_every_exemption_names_a_file_that_exists():
    """An exemption for a file that moved would silently stop covering it."""
    root = Path(annnet.__file__).parent
    present = {path.name for part in MECHANISM for path in (root / part).rglob('*.py')}
    assert EXEMPT <= present, f'these exemptions name nothing: {sorted(EXEMPT - present)}'


def test_the_exemption_list_stays_short():
    """Three format readers is a boundary; thirty would be a suppressed gate."""
    assert len(EXEMPT) <= 5


def test_the_word_list_is_the_one_the_gate_reads():
    """A gate over an empty list passes for the wrong reason."""
    assert len(BIOLOGY_WORDS) >= 20
    assert PATTERN.search('one gene here')
    assert not PATTERN.search('one generic cell of a matrix')


@pytest.mark.parametrize('path', core_files(), ids=lambda path: path.name)
def test_no_module_of_the_mechanism_names_a_biological_concept(path):
    found = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = PATTERN.search(line)
        if match:
            found.append(f'{path.relative_to(PROJECT)}:{number} says {match.group(0)!r}')
    assert not found, (
        'the mechanism half of the package is a general network data structure and\n'
        'names no biology; the vocabulary lives in annnet.experimental:\n' + '\n'.join(found)
    )
