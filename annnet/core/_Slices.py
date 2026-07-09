"""Slice operations, exposed as ``G.slices``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TypedDict
from collections.abc import Iterable

from ._records import SliceRecord, _df_filter_not_equal
from .._support.dataframe_backend import dataframe_columns, dataframe_to_rows, dataframe_filter_eq

if TYPE_CHECKING:
    from .graph import AnnNet


class SliceMembership(TypedDict):
    vertices: set[str]
    edges: set[str]


class SliceInfo(TypedDict):
    vertices: set[str]
    edges: set[str]
    attributes: dict[str, Any]


class SliceStats(TypedDict):
    vertices: int
    edges: int
    attributes: dict[str, Any]


class TemporalChange(TypedDict):
    added: int
    removed: int
    net_change: int


def _bare(v):
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], tuple):
        return v[0]
    return v


class SliceManager:
    """Namespace for all slice operations, exposed as ``G.slices``."""

    __slots__ = ('_G',)

    def __init__(self, graph: AnnNet) -> None:
        self._G = graph

    def _empty_slice_record(self) -> SliceRecord:
        return SliceRecord()

    def _slice_attrs(self, slice_id: str) -> dict[str, Any]:
        df = getattr(self._G, 'slice_attributes', None)
        if df is None or 'slice_id' not in dataframe_columns(df):
            return {}
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'slice_id', slice_id))
        if not rows:
            return {}
        return {
            k: v
            for k, v in rows[0].items()
            if k != 'slice_id' and v is not None and not (isinstance(v, float) and math.isnan(v))
        }

    def _ensure_slice(self, slice_id: str, **attributes: Any) -> SliceRecord:
        G = self._G
        if slice_id not in G._slices:
            G._slices[slice_id] = self._empty_slice_record()
        if attributes:
            G.attrs.set_slice_attrs(slice_id, **attributes)
        return G._slices[slice_id]

    # ── core mutations ────────────────────────────────────────────────────────

    def add(self, slice_id: str, **attributes: Any) -> str:
        """Create a new empty slice.

        Parameters
        ----------
        slice_id : str
        **attributes
            Slice attributes.

        Returns
        -------
        str
        """
        G = self._G
        if slice_id in G._slices and slice_id != 'default':
            raise ValueError(f'slice {slice_id} already exists')
        self._ensure_slice(slice_id, **attributes)
        return slice_id

    def remove(self, slice_id: str) -> None:
        """Remove a non-default slice and its per-slice attributes.

        Parameters
        ----------
        slice_id : str

        Raises
        ------
        ValueError
            If attempting to remove the internal default slice.
        KeyError
            If the slice does not exist.
        """
        G = self._G
        if slice_id == G._default_slice:
            raise ValueError('Cannot remove default slice')
        if slice_id not in G._slices:
            raise KeyError(f'slice {slice_id} not found')

        ela = getattr(G, 'edge_slice_attributes', None)
        if ela is not None and hasattr(ela, 'columns'):
            is_empty = (getattr(ela, 'height', None) == 0) or (
                hasattr(ela, '__len__') and len(ela) == 0
            )
            if (not is_empty) and ('slice_id' in list(ela.columns)):
                G.edge_slice_attributes = _df_filter_not_equal(ela, 'slice_id', slice_id)

        if isinstance(G.slice_edge_weights, dict):
            G.slice_edge_weights.pop(slice_id, None)

        del G._slices[slice_id]
        G._rebuild_slice_edge_weights_cache()
        if G._current_slice == slice_id:
            G._current_slice = G._default_slice

    def add_edge_to_slice(self, lid: str, eid: str) -> None:
        """Attach an existing edge to a slice (no weight changes).

        Parameters
        ----------
        lid : str
        eid : str

        Raises
        ------
        KeyError
            If the slice or edge does not exist.
        """
        G = self._G
        if lid not in G._slices:
            raise KeyError(f'slice {lid!r} does not exist')
        if eid not in G._edges:
            raise KeyError(f'edge {eid!r} does not exist')
        G._slices[lid]['edges'].add(eid)

    def add_edges(self, slice_id: str | None, edge_ids: Iterable[str]) -> None:
        """Attach many existing edges to a slice and include their incident vertices."""
        G = self._G
        sid = slice_id if slice_id is not None else G._current_slice
        data = self._ensure_slice(sid)

        add_edges = {eid for eid in edge_ids if eid in G._edges and G._edges[eid].col_idx >= 0}
        if not add_edges:
            return
        data['edges'].update(add_edges)

        verts: set[str] = set()
        for eid in add_edges:
            rec = G._edges[eid]
            if rec.etype == 'hyper':
                if rec.src is not None:
                    verts.update(_bare(m) for m in rec.src)
                if rec.tgt is not None:
                    verts.update(_bare(m) for m in rec.tgt)
            else:
                if rec.src is not None:
                    verts.add(_bare(rec.src))
                if rec.tgt is not None:
                    verts.add(_bare(rec.tgt))
        data['vertices'].update(verts)

    # ── active slice ──────────────────────────────────────────────────────────

    @property
    def active(self) -> str:
        """Currently active slice identifier."""
        return self._G._current_slice

    @active.setter
    def active(self, slice_id: str) -> None:
        """Set the active slice used by default mutation operations."""
        if slice_id not in self._G._slices:
            raise KeyError(f'slice {slice_id} not found')
        self._G._current_slice = slice_id

    # ── queries ───────────────────────────────────────────────────────────────

    def get_slices_dict(self, include_default: bool = True) -> dict[str, SliceRecord]:
        """Return the raw slice_id → SliceRecord mapping (distinct from ``list``)."""
        G = self._G
        if include_default:
            return G._slices
        return {k: v for k, v in G._slices.items() if k != G._default_slice}

    def list(self, include_default: bool = True) -> list[str]:
        """Slice IDs as a list."""
        return list(self.get_slices_dict(include_default=include_default).keys())

    def exists(self, slice_id: str) -> bool:
        """Return ``True`` if a slice exists."""
        return slice_id in self._G._slices

    def count(self) -> int:
        """Return the number of registered slices."""
        return len(self._G._slices)

    def info(self, slice_id: str) -> SliceInfo:
        """Vertices, edges, and attributes of a slice."""
        G = self._G
        if slice_id not in G._slices:
            raise KeyError(f'slice {slice_id} not found')
        data = G._slices[slice_id]
        return {
            'vertices': data['vertices'].copy(),
            'edges': data['edges'].copy(),
            'attributes': self._slice_attrs(slice_id),
        }

    def vertices(self, slice_id: str) -> set[str]:
        """Return a copy of the vertex IDs in a slice."""
        return self._G._slices[slice_id]['vertices'].copy()

    def edges(self, slice_id: str) -> set[str]:
        """Return a copy of the edge IDs in a slice."""
        return self._G._slices[slice_id]['edges'].copy()

    # ── set operations ────────────────────────────────────────────────────────

    def union(self, slice_ids: Iterable[str]) -> SliceMembership:
        """Return the union of vertices and edges across multiple slices."""
        G = self._G
        uv: set[str] = set()
        ue: set[str] = set()
        for sid in slice_ids:
            if sid in G._slices:
                uv.update(G._slices[sid]['vertices'])
                ue.update(G._slices[sid]['edges'])
        return {'vertices': uv, 'edges': ue}

    def intersect(self, slice_ids: list[str]) -> SliceMembership:
        """Return the intersection of vertices and edges across multiple slices."""
        G = self._G
        if not slice_ids:
            return {'vertices': set(), 'edges': set()}
        if len(slice_ids) == 1:
            data = G._slices.get(slice_ids[0], SliceRecord())
            return {'vertices': data['vertices'].copy(), 'edges': data['edges'].copy()}
        common_v = G._slices[slice_ids[0]]['vertices'].copy()
        common_e = G._slices[slice_ids[0]]['edges'].copy()
        for sid in slice_ids[1:]:
            if sid in G._slices:
                common_v &= G._slices[sid]['vertices']
                common_e &= G._slices[sid]['edges']
            else:
                return {'vertices': set(), 'edges': set()}
        return {'vertices': common_v, 'edges': common_e}

    def difference(self, slice_a: str, slice_b: str) -> SliceMembership:
        """Return the vertices and edges present in one slice but not another."""
        G = self._G
        if slice_a not in G._slices or slice_b not in G._slices:
            raise KeyError('One or both slices not found')
        s1, s2 = G._slices[slice_a], G._slices[slice_b]
        return {
            'vertices': s1['vertices'] - s2['vertices'],
            'edges': s1['edges'] - s2['edges'],
        }

    def create_slice_from_operation(
        self, result_slice_id: str, operation_result: SliceMembership, **attributes: Any
    ) -> str:
        """Create a new slice from a precomputed membership result."""
        G = self._G
        if result_slice_id in G._slices:
            raise ValueError(f'slice {result_slice_id} already exists')
        data = self._ensure_slice(result_slice_id, **attributes)
        data['vertices'] = operation_result['vertices'].copy()
        data['edges'] = operation_result['edges'].copy()
        return result_slice_id

    def add_vertex_to_slice(self, lid: str, vid: str) -> None:
        """Attach an existing vertex to a slice.

        Raises
        ------
        KeyError
            If the slice or vertex does not exist.
        """
        G = self._G
        if lid not in G._slices:
            raise KeyError(f'slice {lid!r} does not exist')
        # Existence check works on flat graphs too, where _vid_to_ekeys is unused
        # (a vid exists iff its placeholder ekey is registered).
        if G._aspects == ('_',):
            exists = (vid, ('_',)) in G._entities
        else:
            exists = vid in G._vid_to_ekeys
        if not exists:
            raise KeyError(f'vertex {vid!r} does not exist')
        G._slices[lid]['vertices'].add(vid)

    # ── set-op creation helpers ───────────────────────────────────────────────

    def union_create(self, slice_ids: Iterable[str], name: str, **attributes: Any) -> str:
        """Create a slice from the union of existing slices."""
        return self.create_slice_from_operation(name, self.union(slice_ids), **attributes)

    def intersect_create(self, slice_ids: list[str], name: str, **attributes: Any) -> str:
        """Create a slice from the intersection of existing slices."""
        return self.create_slice_from_operation(name, self.intersect(slice_ids), **attributes)

    def difference_create(self, slice_a: str, slice_b: str, name: str, **attributes: Any) -> str:
        """Create a slice from the difference of two slices."""
        return self.create_slice_from_operation(
            name, self.difference(slice_a, slice_b), **attributes
        )

    def aggregate(
        self,
        source_slice_ids: list[str],
        target_slice_id: str,
        method: str = 'union',
        weight_func: Any = None,
        **attributes: Any,
    ) -> str:
        """Build a target slice from sources via ``'union'`` or ``'intersection'``."""
        if not source_slice_ids:
            raise ValueError('Must specify at least one source slice')
        G = self._G
        if target_slice_id in G._slices:
            raise ValueError(f'Target slice {target_slice_id} already exists')
        data = self._ensure_slice(target_slice_id, **attributes)

        if method == 'union':
            result = self.union(source_slice_ids)
        elif method == 'intersection':
            result = self.intersect(source_slice_ids)
        else:
            raise ValueError(f'Unknown aggregation method: {method}')
        data['vertices'] = result['vertices']
        data['edges'] = result['edges']
        return target_slice_id

    # ── analytics ─────────────────────────────────────────────────────────────

    def stats(self, include_default: bool = True) -> dict[str, SliceStats]:
        """Return per-slice counts and attributes."""
        return {
            sid: {
                'vertices': len(data['vertices']),
                'edges': len(data['edges']),
                'attributes': self._slice_attrs(sid),
            }
            for sid, data in self.get_slices_dict(include_default=include_default).items()
        }

    def vertex_presence(self, vertex_id: str, include_default: bool = False) -> list[str]:
        """List slices that contain a given vertex."""
        return [
            sid
            for sid, data in self.get_slices_dict(include_default=include_default).items()
            if vertex_id in data['vertices']
        ]

    def edge_presence(
        self,
        edge_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        *,
        include_default: bool = False,
        undirected_match: bool | None = None,
    ) -> list[str] | dict[str, list[str]]:
        """Slices containing an edge by id, or by (source, target) endpoint pair."""
        G = self._G
        has_id = edge_id is not None
        has_pair = (source is not None) and (target is not None)
        if has_id == has_pair:
            raise ValueError('Provide either edge_id OR (source and target), but not both.')
        slices_view = self.get_slices_dict(include_default=include_default)
        if has_id:
            return [lid for lid, ldata in slices_view.items() if edge_id in ldata['edges']]
        if undirected_match is None:
            undirected_match = False
        out: dict[str, list[str]] = {}
        default_dir = True if G.directed is None else G.directed
        for lid, ldata in slices_view.items():
            matches: list[str] = []
            for eid in ldata['edges']:
                rec = G._edges.get(eid)
                if rec is None or rec.col_idx < 0 or rec.etype == 'hyper':
                    continue
                s, t = rec.src, rec.tgt
                edge_is_directed = rec.directed if rec.directed is not None else default_dir
                if s == source and t == target:
                    matches.append(eid)
                elif undirected_match and not edge_is_directed and s == target and t == source:
                    matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def hyperedge_presence(
        self,
        *,
        members: Iterable[str] | None = None,
        head: Iterable[str] | None = None,
        tail: Iterable[str] | None = None,
        include_default: bool = False,
    ) -> dict[str, list[str]]:
        """Slices containing a hyperedge by undirected members or directed head+tail."""
        G = self._G
        undirected = members is not None
        if undirected and (head is not None or tail is not None):
            raise ValueError('Use either members OR head+tail, not both.')
        if not undirected and (head is None or tail is None):
            raise ValueError('Directed hyperedge query requires both head and tail.')
        if undirected:
            members_set = {_bare(v) for v in members} if members is not None else set()
            if not members_set:
                raise ValueError('members must be non-empty.')
        else:
            head_set = {_bare(v) for v in head} if head is not None else set()
            tail_set = {_bare(v) for v in tail} if tail is not None else set()
            if not head_set or not tail_set:
                raise ValueError('head and tail must be non-empty.')
            if head_set & tail_set:
                raise ValueError('head and tail must be disjoint.')

        out: dict[str, list[str]] = {}
        for lid, ldata in self.get_slices_dict(include_default=include_default).items():
            matches: list[str] = []
            for eid in ldata['edges']:
                rec = G._edges.get(eid)
                if rec is None or rec.col_idx < 0 or rec.etype != 'hyper':
                    continue
                if undirected and rec.tgt is None:
                    if {_bare(v) for v in rec.src} == members_set:
                        matches.append(eid)
                elif (not undirected) and rec.tgt is not None:
                    if {_bare(v) for v in rec.src} == head_set and {
                        _bare(v) for v in rec.tgt
                    } == tail_set:
                        matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def conserved_edges(self, min_slices: int = 2, include_default: bool = False) -> dict[str, int]:
        """Count edges that appear in at least ``min_slices`` slices."""
        G = self._G
        edge_counts: dict[str, int] = {}
        for sid, data in G._slices.items():
            if not include_default and sid == G._default_slice:
                continue
            for eid in data['edges']:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
        return {eid: c for eid, c in edge_counts.items() if c >= min_slices}

    def specific_edges(self, slice_id: str) -> set[str]:
        """Return edges that appear only in the given slice."""
        G = self._G
        if slice_id not in G._slices:
            raise KeyError(f'slice {slice_id} not found')
        return {
            eid
            for eid in G._slices[slice_id]['edges']
            if sum(1 for data in G._slices.values() if eid in data['edges']) == 1
        }

    def temporal_dynamics(
        self, ordered_slices: list[str], metric: str = 'edge_change'
    ) -> list[TemporalChange]:
        """Summarize added and removed members across an ordered slice sequence."""
        G = self._G
        if len(ordered_slices) < 2:
            raise ValueError('Need at least 2 slices for temporal analysis')
        changes: list[TemporalChange] = []
        for i in range(len(ordered_slices) - 1):
            cur, nxt = ordered_slices[i], ordered_slices[i + 1]
            if cur not in G._slices or nxt not in G._slices:
                raise KeyError('One or more slices not found')
            cd, nd = G._slices[cur], G._slices[nxt]
            key = 'edges' if metric == 'edge_change' else 'vertices'
            added = len(nd[key] - cd[key])
            removed = len(cd[key] - nd[key])
            changes.append({'added': added, 'removed': removed, 'net_change': added - removed})
        return changes

    # ── convenience ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a compact human-readable summary of all slices."""
        stats = self.stats(include_default=True)
        lines = [f'slices: {len(stats)}']
        for i, (sid, info) in enumerate(stats.items()):
            prefix = '├─' if i < len(stats) - 1 else '└─'
            lines.append(f'{prefix} {sid}: {info["vertices"]} vertices, {info["edges"]} edges')
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f'SliceManager({self.count()} slices)'
