"""An archive written before the rename still reads.

The package said "vertex" where it now says "node". The word reached four member
names of the native format, two of its columns, and the entity kind it writes on
every node. A file already on disk holds the old word, so the reader takes both
spellings. The writer emits the new one alone, which is the only direction that
has to hold.

The archive under test is one this version wrote, renamed back to the old
vocabulary. That is what a file from before the rename looks like, and building
it this way keeps it in step with the format rather than frozen against one
version of it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from annnet.core.graph import AnnNet
from annnet.io import annnet_format

# What the members and the columns were called.
MEMBER_NAMES = {
    '_node_table.parquet': '_vertex_table.parquet',
    'node_presence.parquet': 'vertex_presence.parquet',
    'node_memberships.parquet': 'vertex_memberships.parquet',
    'node_layer_attributes.parquet': 'vertex_layer_attributes.parquet',
}
COLUMN_NAMES = {'node_id': 'vertex_id', 'node_layer_id': 'vertex_layer_id'}


def a_graph() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes(['A', 'B', 'C'], kind='gene')
    graph.add_edges('A', 'B', edge_id='e0', weight=1.5)
    graph.add_edges('B', 'C', edge_id='e1', weight=2.0)
    graph.attrs.set_node_attrs('A', symbol='TP53')
    graph.slices.add('left')
    graph.slices.add_edges('left', ['e0'])
    return graph


def written_in_the_old_words(graph: AnnNet, tmp_path: Path) -> Path:
    """Write an archive and rename every member and column back."""
    fresh = tmp_path / 'fresh.annnet'
    annnet_format.write(graph, fresh)

    unpacked = tmp_path / 'unpacked'
    with zipfile.ZipFile(fresh) as archive:
        archive.extractall(unpacked)
    root = next(path for path in unpacked.iterdir() if path.is_dir())

    for member in sorted(root.rglob('*.parquet')):
        table = pl.read_parquet(member)
        mapping = {new: old for new, old in COLUMN_NAMES.items() if new in table.columns}
        if mapping:
            table.rename(mapping).write_parquet(member)
        older = MEMBER_NAMES.get(member.name)
        if older is not None:
            member.rename(member.with_name(older))

    # The entity kind is written on every node, and it was the same word.
    entities = root / 'structure' / 'entity_index.parquet'
    if entities.exists():
        table = pl.read_parquet(entities)
        if 'type' in table.columns:
            table.with_columns(
                pl.col('type').replace({'node': 'vertex'}),
            ).write_parquet(entities)

    old = tmp_path / 'old.annnet'
    with zipfile.ZipFile(old, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in sorted(root.rglob('*')):
            if entry.is_file():
                archive.write(entry, arcname=f'{root.name}/{entry.relative_to(root).as_posix()}')
    return old


@pytest.fixture
def legacy_archive(tmp_path):
    return written_in_the_old_words(a_graph(), tmp_path)


def test_the_structure_comes_back(legacy_archive):
    graph = annnet_format.read(legacy_archive)
    assert set(graph.N) == {'A', 'B', 'C'}
    assert set(graph.E) == {'e0', 'e1'}
    assert graph.get_edge('e0').weight == 1.5


def test_every_entity_is_a_node_again(legacy_archive):
    graph = annnet_format.read(legacy_archive)
    assert set(graph.views.entity_kinds().values()) == {'node'}


def test_the_node_attributes_come_back(legacy_archive):
    graph = annnet_format.read(legacy_archive)
    assert graph.attrs.get_attr_node('A', 'symbol') == 'TP53'
    assert graph.attrs.get_attr_node('B', 'kind') == 'gene'


def test_the_slices_come_back(legacy_archive):
    graph = annnet_format.read(legacy_archive)
    assert graph.slices.edges('left') == {'e0'}


def test_a_multilayer_archive_comes_back(tmp_path):
    graph = AnnNet(directed=True, aspects={'cond': ['ctrl', 'treat']})
    graph.add_nodes('A', layer=('ctrl',))
    graph.add_nodes('A', layer=('treat',))
    graph.add_nodes('B', layer=('ctrl',))
    graph.add_edges(('A', ('ctrl',)), ('B', ('ctrl',)), edge_id='e0')
    graph.layers.set_node_attrs('A', ('ctrl',), state='baseline')

    read_back = annnet_format.read(written_in_the_old_words(graph, tmp_path))
    assert set(read_back.N) == {'A', 'B'}
    assert read_back.nv_supra == 3
    assert read_back.layers.node_attrs('A', ('ctrl',)) == {'state': 'baseline'}


def test_the_writer_emits_the_new_words_only(tmp_path):
    path = tmp_path / 'fresh.annnet'
    annnet_format.write(a_graph(), path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    assert not [name for name in names if 'vertex' in name], names
