"""One sidecar, one schema, for every format that cannot hold a whole graph.

Most exchange formats are narrower than AnnNet. A SIF file is an edge list; a
GraphML document has no idea what a slice is; even JSON and Parquet, as this
package writes them, carry no contextual attributes. What each of them dropped
used to be its own business, and the result was five conventions:

===============  ==========================================================
GraphML / GEXF   ``<path>.manifest.json``, written and read automatically
SIF              ``manifest_path=``, opt-in, different shape
Parquet          ``manifest_path=``, different shape again
PyG              a ``.manifest`` attribute on the returned object
nx / ig / gt     returned as the second half of a tuple, caller must keep it
CX2 / JSON       nothing at all
===============  ==========================================================

None of them declared *what* was lost, so the only way to find out was to
compare the graph you got back with the one you sent.

This module replaces all of that with one file, one schema, and one rule: a
writer records what its format cannot represent, and a reader puts it back.

**The sidecar is bound to its primary.** It stores the SHA-256 of the file it
was written beside. Edit the SIF by hand and the sidecar refuses to apply rather
than quietly restoring state that no longer matches — a stale sidecar is worse
than none, because it looks like it worked.
"""

from __future__ import annotations

import json
from typing import Any
import hashlib
from pathlib import Path
import warnings

from ...core import _structure as S
from .contextual import contextual_payload, restore_contextual

SIDECAR_SUFFIX = '.annnet-sidecar'
SIDECAR_SCHEMA = 'annnet-sidecar/1'

# Everything a graph can carry that a format may or may not be able to hold.
# One vocabulary, so that a capability declaration and a loss report use the
# same words.
CAPABILITIES = (
    'nodes',
    'binary_edges',
    'directed',
    'weights',
    'parallel_edges',
    'hyperedges',
    'edge_entities',
    'coefficients',
    'node_attributes',
    'edge_attributes',
    'slices',
    'slice_membership',
    'contextual_attributes',
    'multilayer',
    'graph_attributes',
)

# What each format preserves. Anything a format is not listed for goes to the
# sidecar. These were measured by round-tripping a graph that carries every
# capability and comparing, not assumed from the format's reputation.
FORMAT_CAPABILITIES: dict[str, frozenset[str]] = {
    'annnet': frozenset(CAPABILITIES),
    'cx2': frozenset(CAPABILITIES),
    'json': frozenset(CAPABILITIES),
    'parquet': frozenset(CAPABILITIES),
    'graphml': frozenset(
        {
            'nodes',
            'binary_edges',
            'directed',
            'weights',
            'parallel_edges',
            'hyperedges',
            'node_attributes',
            'edge_attributes',
            'multilayer',
        }
    ),
    # SIF is an edge list: a node that no edge names is not in the file at all,
    # so node identity is one of the things the sidecar has to carry.
    'sif': frozenset({'binary_edges', 'directed'}),
    # An edge list names no node on its own and carries no node attribute, so
    # both go to the sidecar; the columns beside the endpoints carry the rest.
    'csv': frozenset(
        {
            'binary_edges',
            'directed',
            'weights',
            'parallel_edges',
            'edge_attributes',
            'slices',
            'slice_membership',
        }
    ),
}
FORMAT_CAPABILITIES['gexf'] = FORMAT_CAPABILITIES['graphml']
FORMAT_CAPABILITIES['excel'] = FORMAT_CAPABILITIES['csv']
# SBML is reactions over species: topology and stoichiometry survive, the rest does not.
_ALL_CAPABILITIES = frozenset(CAPABILITIES)

FORMAT_CAPABILITIES['sbml'] = frozenset(
    {'nodes', 'binary_edges', 'hyperedges', 'directed', 'coefficients'}
)


class SidecarIntegrityError(ValueError):
    """A sidecar does not match the file it was written beside."""


class AnnNetLossWarning(UserWarning):
    """A format could not hold part of a graph."""


def sidecar_path(primary: str | Path) -> Path:
    """The companion path for one primary file. Deterministic, never guessed."""
    primary = Path(primary)
    return primary.with_name(primary.name + SIDECAR_SUFFIX)


def digest(primary: str | Path) -> str:
    """Hash one file, or a directory tree with its member names included."""
    path = Path(primary)
    sha = hashlib.sha256()
    if path.is_file():
        sha.update(b'file\0')
        _hash_file(sha, path)
        return sha.hexdigest()
    sha.update(b'dir\0')
    for member in sorted(item for item in path.rglob('*') if item.is_file()):
        relative = member.relative_to(path).as_posix().encode()
        sha.update(len(relative).to_bytes(8, 'big'))
        sha.update(relative)
        _hash_file(sha, member)
    return sha.hexdigest()


def _hash_file(sha, path: Path) -> None:
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)


def losses_for(graph, format_name: str) -> tuple[str, ...]:
    """Which capabilities this graph carries that this format cannot hold."""
    supported = FORMAT_CAPABILITIES.get(format_name, frozenset())
    if _ALL_CAPABILITIES <= supported:
        return ()
    return tuple(name for name in present(graph) if name not in supported)


def present(graph) -> tuple[str, ...]:
    """Which capabilities a graph actually carries.

    A format is only reported as lossy for what the graph has. Writing a plain
    directed graph to SIF loses nothing, and should say so. The edge questions
    are answered from the store's own arrays: asking the facade once per edge
    made deciding whether to write a sidecar cost more than writing one.
    """
    found = ['nodes', 'binary_edges']
    store = S.store_of(graph)
    slots = store.live_edge_slots()
    if S.has_hyperedges(graph):
        found.append('hyperedges')
    if slots.size:
        if bool((store.edge_weight[slots] != 1.0).any()):
            found.append('weights')
        if bool(store.edge_explicit[slots].any()):
            found.append('coefficients')
    if store.edge_entity_count:
        found.append('edge_entities')
    if graph._attr_store.node_columns:
        found.append('node_attributes')
    if graph._attr_store.edge_columns:
        found.append('edge_attributes')
    non_default = [s for s in graph.slices.list() if s != graph._default_slice]
    if non_default:
        found.extend(('slices', 'slice_membership'))
    if not graph._contextual.is_empty():
        found.append('contextual_attributes')
    if graph.is_multilayer:
        found.append('multilayer')
    if graph.graph_attributes:
        found.append('graph_attributes')
    return tuple(found)


def write_sidecar(
    graph,
    primary: str | Path,
    *,
    format_name: str,
    warn: bool = True,
    format_payload=None,
):
    """Write what ``format_name`` could not hold, beside the file it wrote.

    ``format_payload`` is whatever that one format needs to rebuild what it
    wrote — for GraphML and GEXF it is the mapping that keeps edge identity,
    which the format itself does not preserve. It lives here rather than in a
    file of the format's own invention, which is what made three conventions
    out of one job.

    Returns the sidecar path, or None when there was nothing to keep.
    """
    lost = losses_for(graph, format_name)
    path = sidecar_path(primary)
    if not lost and format_payload is None:
        if path.exists():
            path.unlink()
        return None
    document = {
        'schema': SIDECAR_SCHEMA,
        'primary': {
            'name': Path(primary).name,
            'format': format_name,
            'sha256': digest(primary),
        },
        'holds': list(lost),
        'payload': _payload(graph, lost),
        'format_payload': format_payload,
    }
    path.write_text(json.dumps(document, indent=1, sort_keys=True), encoding='utf-8')
    if warn and lost:
        warnings.warn(
            f'{format_name} cannot hold {", ".join(lost)}; preserved them in {path.name}',
            AnnNetLossWarning,
            stacklevel=3,
        )
    return path


def read_sidecar(primary: str | Path, *, policy: str = 'auto') -> dict[str, Any] | None:
    """Load and verify the sidecar beside one file.

    ``policy`` is ``'auto'`` (use it when present), ``'require'`` (fail when
    absent) or ``'ignore'``.
    """
    if policy == 'ignore':
        return None
    path = sidecar_path(primary)
    if not path.exists():
        if policy == 'require':
            raise FileNotFoundError(f'required sidecar {path} does not exist')
        return None
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema') != SIDECAR_SCHEMA:
        raise SidecarIntegrityError(
            f'{path} has schema {document.get("schema")!r}, expected {SIDECAR_SCHEMA!r}'
        )
    recorded = (document.get('primary') or {}).get('sha256')
    actual = digest(primary)
    if recorded != actual:
        raise SidecarIntegrityError(
            f'{path} was written for a different version of {Path(primary).name}; '
            f'the file has changed since. Delete the sidecar to read without it.'
        )
    return document


def apply_sidecar(graph, document: dict[str, Any] | None):
    """Put back everything a sidecar holds. Returns the graph."""
    if not document:
        return graph
    _restore(graph, document.get('payload') or {}, set(document.get('holds') or ()))
    return graph


def _payload(graph, lost) -> dict[str, Any]:
    """The state itself, only for the capabilities the format could not hold."""
    lost = set(lost)
    payload: dict[str, Any] = {}
    if 'contextual_attributes' in lost:
        payload['contextual'] = contextual_payload(graph)
    if 'graph_attributes' in lost:
        payload['graph_attributes'] = dict(graph.graph_attributes)
    if 'slices' in lost or 'slice_membership' in lost:
        payload['slices'] = {
            sid: {
                'nodes': sorted(graph.slices.nodes(sid)),
                'edges': sorted(graph.slices.edges(sid)),
            }
            for sid in graph.slices.list()
            if sid != graph._default_slice
        }
    if 'hyperedges' in lost:
        payload['hyperedges'] = [
            {
                'id': definition.id,
                'head': sorted(str(v) for v in definition.source),
                'tail': sorted(str(v) for v in definition.target),
                'weight': definition.weight,
            }
            for definition in S.definitions_of(graph)[1]
            if definition.kind == S.HYPER
        ]
    if 'nodes' in lost:
        payload['nodes'] = [
            {'id': ref.id, 'layer': list(ref.layer)}
            for ref in S.iter_entities(graph)
            if ref.kind == S.NODE
        ]
    if 'node_attributes' in lost:
        payload['node_attributes'] = graph._attr_store.node_attr_rows()
    if 'edge_attributes' in lost:
        payload['edge_attributes'] = graph._attr_store.edge_attr_rows()
    if 'multilayer' in lost:
        payload['multilayer'] = {
            'aspects': list(S.aspects(graph)) if hasattr(S, 'aspects') else list(graph.aspects),
            'entities': {ref.id: list(ref.layer) for ref in S.iter_entities(graph)},
        }
    return payload


def _restore(graph, payload: dict[str, Any], holds: set[str]) -> None:
    # Nodes first: everything below is keyed by an id that has to exist.
    for record in payload.get('nodes') or ():
        if not S.has_entity_id(graph, record['id']):
            layer = tuple(record['layer'])
            graph.add_nodes([record['id']], layer=None if layer == ('_',) else layer)
    if 'contextual' in payload:
        restore_contextual(graph, payload['contextual'])
    if 'graph_attributes' in payload:
        graph.graph_attributes.update(payload['graph_attributes'])
    for sid, members in (payload.get('slices') or {}).items():
        if not graph.slices.exists(sid):
            graph.slices.add(sid)
        graph.slices.add_nodes(sid, [n for n in members['nodes'] if S.has_entity_id(graph, n)])
        graph.slices.attach_edges(sid, [e for e in members['edges'] if S.has_edge(graph, e)])
    for node_id, attrs in (payload.get('node_attributes') or {}).items():
        clean = {k: v for k, v in attrs.items() if v is not None and k != 'node_id'}
        if clean and S.has_entity_id(graph, node_id):
            graph.attrs.set_node_attrs(node_id, **clean)
    for edge_id, attrs in (payload.get('edge_attributes') or {}).items():
        clean = {k: v for k, v in attrs.items() if v is not None and k != 'edge_id'}
        if clean and S.has_edge(graph, edge_id):
            graph.attrs.set_edge_attrs(edge_id, **clean)


# ---------------------------------------------------------------------------
# The decorators every adapter wears
# ---------------------------------------------------------------------------


def preserves(format_name: str):
    """Wrap a writer so it records what its format could not hold.

    The writer keeps its own signature and stays unaware of the sidecar. Callers
    can turn it off per call with ``sidecar=False``, which is the only way to get
    a lossy file with nothing beside it.
    """
    import inspect
    import functools

    def decorate(writer):
        signature = inspect.signature(writer)

        @functools.wraps(writer)
        def wrapper(graph, path=None, *args, **kwargs):
            wanted = kwargs.pop('sidecar', True)
            result = writer(graph, path, *args, **kwargs)
            if wanted and path is not None:
                target = Path(path)
                if target.exists():
                    write_sidecar(graph, target, format_name=format_name)
            return result

        extra = inspect.Parameter(
            'sidecar', inspect.Parameter.KEYWORD_ONLY, default=True, annotation='bool'
        )
        parameters = list(signature.parameters.values())
        variadic = [p for p in parameters if p.kind is inspect.Parameter.VAR_KEYWORD]
        fixed = [p for p in parameters if p.kind is not inspect.Parameter.VAR_KEYWORD]
        if 'sidecar' not in signature.parameters:
            wrapper.__signature__ = signature.replace(parameters=fixed + [extra] + variadic)
        return wrapper

    return decorate


def restores(reader):
    """Wrap a reader so a sidecar beside its source is put back automatically."""
    import inspect
    import functools

    signature = inspect.signature(reader)

    @functools.wraps(reader)
    def wrapper(source, *args, **kwargs):
        policy = kwargs.pop('sidecar', 'auto')
        graph = reader(source, *args, **kwargs)
        try:
            document = read_sidecar(source, policy=policy)
        except (TypeError, OSError):
            document = None
        return apply_sidecar(graph, document)

    extra = inspect.Parameter(
        'sidecar', inspect.Parameter.KEYWORD_ONLY, default='auto', annotation='str'
    )
    parameters = list(signature.parameters.values())
    variadic = [p for p in parameters if p.kind is inspect.Parameter.VAR_KEYWORD]
    fixed = [p for p in parameters if p.kind is not inspect.Parameter.VAR_KEYWORD]
    if 'sidecar' not in signature.parameters:
        wrapper.__signature__ = signature.replace(parameters=fixed + [extra] + variadic)
    return wrapper
