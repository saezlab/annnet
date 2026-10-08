"""Sets of keys and bindings to an external feature axis.

``Set``, ``Space`` and ``G.spaces`` are generic mechanisms: a set of keys, and a
name for the axis of an external array those keys index. They know nothing about
what the keys mean. They are kept in AnnNet while the integrations built on them
(``annnet.experimental`` and the optional AnnData accessor) are still being
designed, and their shape may change with those.
"""

from collections.abc import Set as AbstractSet, Iterable


def _keys(value):
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return (value,)
    return value


class Set(AbstractSet):
    """An immutable set of arbitrary keys with scalar input support.

    Strings and non-iterable values are single keys. Wrap a tuple key in a
    list to distinguish it from a collection. Algebra returns a generic Set.
    """

    def __init__(self, keys=()):
        self._keys = frozenset(_keys(keys))

    def __contains__(self, key):
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    @classmethod
    def _from_iterable(cls, values):
        return Set(values)

    def __and__(self, other):
        return Set(self._keys.intersection(_keys(other)))

    __rand__ = __and__

    def __or__(self, other):
        return Set(self._keys.union(_keys(other)))

    __ror__ = __or__

    def __sub__(self, other):
        return Set(self._keys.difference(_keys(other)))

    def __rsub__(self, other):
        return Set(frozenset(_keys(other)).difference(self._keys))

    def __repr__(self):
        return f'{type(self).__name__}({set(self)!r})'


class Spaces:
    """Persistent node membership, accessed through ``graph.spaces``."""

    def __init__(self, graph):
        self._graph = graph

    def of(self, node):
        """Return the space names containing an existing node."""
        if not self._graph.has_node(node):
            return Set()
        return Set(
            name for name, nodes in self._graph.uns.get('_spaces', {}).items() if node in nodes
        )

    def nodes(self, name):
        """Return existing nodes recorded in one space."""
        return Set(
            node
            for node in self._graph.uns.get('_spaces', {}).get(name, ())
            if self._graph.has_node(node)
        )


class Space(Set):
    """Bind graph nodes to an external axis without populating membership.

    ``axis`` accepts feature keys or a container with ``var_names``.
    ``id_map`` maps node keys to one or several feature keys. Without a map,
    identical keys bind. Translation returns sets so multiplicity is explicit.
    Algebra returns node keys and does not mutate the binding.
    """

    def __init__(self, graph, axis=(), *, name='default', id_map=None):
        super().__init__()
        self.graph = graph
        self.axis = axis if hasattr(axis, 'var_names') else tuple(_keys(axis))
        if not isinstance(name, str) or not name:
            raise ValueError('Space name must be a nonempty string')
        self.name = name
        self._id_map = (
            None
            if id_map is None
            else {node: frozenset(_keys(features)) for node, features in id_map.items()}
        )
        if hasattr(axis, 'var_names'):
            from .backend_accessors.axis import bind_axis

            bind_axis(axis, self)

    def _axis_keys(self):
        return frozenset(_keys(getattr(self.axis, 'var_names', self.axis)))

    def features(self, nodes):
        """Translate node keys to current feature keys, preserving all matches."""
        axis = self._axis_keys()
        return Set(
            feature
            for node in _keys(nodes)
            for feature in ({node} if self._id_map is None else self._id_map.get(node, ()))
            if feature in axis
        )

    def nodes(self, features):
        """Translate feature keys to existing graph nodes."""
        wanted = frozenset(_keys(features)) & self._axis_keys()
        candidates = list(self.graph.N) if self._id_map is None else self._id_map
        return Set(
            node
            for node in candidates
            if self.graph.has_node(node) and wanted.intersection(self.features(node))
        )

    def populate(self, nodes):
        """Add node membership after validating the complete input."""
        wanted = Set(nodes)
        missing = [node for node in wanted if not self.graph.has_node(node)]
        if missing:
            raise KeyError(f'Unknown nodes: {missing!r}')
        memberships = self.graph.uns.setdefault('_spaces', {})
        held = memberships.get(self.name, [])
        memberships[self.name] = list(dict.fromkeys([*held, *wanted]))
        return self

    def __iter__(self):
        return iter(self.graph.spaces.nodes(self.name))

    def __len__(self):
        return len(self.graph.spaces.nodes(self.name))

    def __contains__(self, key):
        held = self.graph.spaces.nodes(self.name)
        normalized = self._normalized(key)
        return bool(normalized & held)

    def _normalized(self, other):
        values = Set(other)
        translated = self.nodes(values)
        for key in values:
            if self.graph.has_node(key) and self.nodes(key) - {key}:
                raise ValueError(
                    'ambiguous node and feature key spaces; use nodes() or features() explicitly'
                )
        return Set(values | translated)

    def __and__(self, other):
        return Set(self).__and__(self._normalized(other))

    __rand__ = __and__

    def __or__(self, other):
        values = self._normalized(other)
        return Set(self) | Set(key for key in values if self.graph.has_node(key))

    __ror__ = __or__

    def __sub__(self, other):
        return Set(self) - self._normalized(other)

    def __rsub__(self, other):
        return self._normalized(other) - Set(self)
