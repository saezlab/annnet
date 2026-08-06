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


def core_files() -> list[Path]:
    return sorted(CORE.rglob('*.py'))


def test_the_word_list_is_the_one_the_gate_reads():
    """A gate over an empty list passes for the wrong reason."""
    assert len(BIOLOGY_WORDS) >= 20
    assert PATTERN.search('one gene here')
    assert not PATTERN.search('one generic cell of a matrix')


@pytest.mark.parametrize('path', core_files(), ids=lambda path: path.name)
def test_no_module_of_the_core_names_a_biological_concept(path):
    found = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = PATTERN.search(line)
        if match:
            found.append(f'{path.relative_to(PROJECT)}:{number} says {match.group(0)!r}')
    assert not found, (
        'the core is a general network data structure and names no biology:\n' + '\n'.join(found)
    )
