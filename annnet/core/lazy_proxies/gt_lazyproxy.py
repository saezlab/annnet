## Lazy graph-tool proxy

from ._base import _LazyProxyBase


class _LazyGTProxy(_LazyProxyBase):
    """Lazy graph-tool proxy attached to an AnnNet instance.

    This proxy lets you call graph-tool algorithms via namespaces such as:
    `G.gt.topology.shortest_distance(...)`, `G.gt.centrality.betweenness(...)`,
    and `G.gt.flow.push_relabel_max_flow(...)`.

    On first use (or after a graph mutation), AnnNet is converted to a graph-tool
    graph and cached; subsequent calls reuse the cached backend until the AnnNet
    version changes. Conversion produces a **manifest** dictionary that preserves
    information graph-tool cannot represent (e.g., hyperedges, per-edge directedness,
    slices, multilayer metadata, stable edge IDs). The manifest is JSON-serializable
    and can be persisted by the adapter.

    Notes
    -----
    - Requires the optional `graph-tool` dependency (not on PyPI).
    - The typical usage pattern is `G.gt.<namespace>.<algo>(...)`, which lazily
      converts, runs the graph-tool algorithm, and returns its output.
    """

    # lazy module loader
    @staticmethod
    def _load_gt_module(name):
        from importlib import import_module

        return import_module(f"graph_tool.{name}")

    VERTEX_KEYS = {"source", "target", "vertex", "root", "u", "v"}

    def __init__(self, owner):
        self._G = owner
        self._cache = {}
        self.cache_enabled = True

        # lazy module map (values are callables)
        self._GT_MODULES = {
            name: (lambda n=name: self._load_gt_module(n))
            for name in [
                "topology",
                "centrality",
                "clustering",
                "flow",
                "inference",
                "generation",
                "search",
                "util",
            ]
        }

        # Initialize namespace objects (lazy modules)
        self._namespaces = {name: _GTNamespaceProxy(self, name) for name in self._GT_MODULES.keys()}

    def clear(self):
        self._cache.clear()

    def backend(self):
        return self._get_or_make_gt()

    def __getattr__(self, name):
        if name in self._namespaces:
            return self._namespaces[name]
        matches = []
        for namespace in self._namespaces.values():
            try:
                matches.append(getattr(namespace, name))
            except AttributeError:
                continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AttributeError(
                f"G.gt attribute '{name}' is ambiguous across namespaces; call it via G.gt.<namespace>.{name}(...)"
            )
        raise AttributeError(f"G.gt has no attribute '{name}'. Only namespaces: {list(self._namespaces.keys())}")

    def __dir__(self):
        names = set(super().__dir__()) | set(self._namespaces.keys())
        for namespace in self._namespaces.values():
            try:
                names.update(dir(namespace))
            except Exception:
                continue
        return sorted(names)

    # Conversion

    def _get_or_make_gt(self):
        key = ("gt",)
        version = getattr(self._G, "_version", None)
        entry = self._cache.get(key)

        if not self.cache_enabled or entry is None or entry["version"] != version:
            from ...adapters.graphtool_adapter import to_graphtool

            gtG, manifest = to_graphtool(self._G)
            self._warn_on_loss(manifest)
            self._cache[key] = {"gtG": gtG, "version": version}
        return self._cache[key]["gtG"]

    # Vertex coercion

    def _coerce_vertices(self, bound, kwargs, gtG):
        label_field = self._infer_label_field()
        id_map = self._build_id_map(gtG)
        vertex_ids = {
            ekey[0]
            for ekey, rec in self._G._entities.items()
            if rec.kind == "vertex"
        }

        def map_one(x):
            # AnnNet internal id?
            if hasattr(x, "__class__") and x.__class__.__module__.startswith("graph_tool."):
                return x
            if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], tuple):
                vid = x[0]
            elif x in vertex_ids:
                vid = x
            else:
                vid = self._lookup_vertex_id(label_field, x)
                if vid is None:
                    raise ValueError(f"Unknown vertex label '{x}' (label field '{label_field}')")
            idx = id_map[vid]
            return gtG.vertex(idx)

        def map_obj(obj):
            if isinstance(obj, (list, tuple, set)):
                items = [map_one(o) for o in obj]
                if isinstance(obj, tuple):
                    return tuple(items)
                if isinstance(obj, set):
                    return set(items)
                return items
            return map_one(obj)

        # Bound arguments
        if bound:
            for k in list(bound.arguments):
                if k in self.VERTEX_KEYS:
                    bound.arguments[k] = map_obj(bound.arguments[k])
        # Raw kwargs
        else:
            for k in list(kwargs):
                if k in self.VERTEX_KEYS:
                    kwargs[k] = map_obj(kwargs[k])

    def _build_id_map(self, gtG):
        if "id" not in gtG.vp:
            raise RuntimeError("graph-tool backend missing vertex ID property 'id'")
        vp = gtG.vp["id"]
        out = {}
        for v in gtG.vertices():
            out[str(vp[v])] = int(v)
        return out

    # Label inference

    def _infer_label_field(self):
        return super()._infer_label_field()

    def _lookup_vertex_id(self, field, val):
        if field is None:
            return None
        return self._lookup_vertex_id_by_label(field, val)

    # Lossy conversion warnings

    def _warn_on_loss(self, manifest):
        msgs = []
        if manifest:
            if manifest["edges"]["hyperedges"]:
                msgs.append("hyperedges dropped")
            if len(manifest["slices"]["data"]) > 1:
                msgs.append("multiple slices collapsed")
        if msgs:
            import warnings

            warnings.warn("AnnNet → graph-tool conversion is lossy: " + "; ".join(msgs))


class _GTNamespaceProxy:
    def __init__(self, parent, module_name):
        self._parent = parent
        self._module_name = module_name
        self._module = None
        self._module_loader = parent._GT_MODULES[module_name]

    def _load(self):
        if self._module is None:
            self._module = self._module_loader()  # load real module
        return self._module

    def __getattr__(self, name):
        mod = self._load()
        func = getattr(mod, name, None)
        if not callable(func):
            raise AttributeError(f"graph_tool.{self._module_name} has no function '{name}'")
        return self._wrap(func)

    def __dir__(self):
        mod = self._load()
        names = set(super().__dir__())
        for name in dir(mod):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(mod, name)
            except Exception:
                continue
            if callable(attr):
                names.add(name)
        return sorted(names)

    def _wrap(self, func):
        def call(*args, **kwargs):
            parent = self._parent

            # detect if AnnNet AnnNet passed
            has_owner = any(a is parent._G for a in args) or any(
                v is parent._G for v in kwargs.values()
            )
            gtG = None

            try:
                import inspect

                sig = inspect.signature(func)
            except Exception:
                sig = None

            if has_owner:
                gtG = parent._get_or_make_gt()

                # replace G with gtG
                args = tuple(gtG if a is parent._G else a for a in args)
                for k, v in list(kwargs.items()):
                    if v is parent._G:
                        kwargs[k] = gtG
            elif sig is not None:
                params = list(sig.parameters.values())
                if params:
                    first = params[0]
                    needs_graph = (
                        first.kind
                        in (
                            first.POSITIONAL_ONLY,
                            first.POSITIONAL_OR_KEYWORD,
                        )
                        and first.default is inspect._empty
                        and first.name in {"g", "graph", "G"}
                        and len(args) == 0
                        and first.name not in kwargs
                    )
                    if needs_graph:
                        gtG = parent._get_or_make_gt()
                        args = (gtG,)

            # bind signature
            try:
                bound = sig.bind_partial(*args, **kwargs) if sig is not None else None
            except Exception:
                bound = None

            # vertex coercion
            if gtG is not None:
                parent._coerce_vertices(bound, kwargs, gtG)

            # weight property mapping
            if bound:
                if "weights" in bound.arguments and isinstance(bound.arguments["weights"], str):
                    pname = bound.arguments["weights"]
                    bound.arguments["weights"] = gtG.ep[pname]
                return func(*bound.args, **bound.kwargs)

            return func(*args, **kwargs)

        return call
