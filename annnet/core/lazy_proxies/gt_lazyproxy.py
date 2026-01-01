## Lazy graph-tool proxy


class _LazyGTProxy:
    """
    graph-tool lazy proxy.
    Supports:
        G.gt.topology.shortest_distance(...)
        G.gt.centrality.betweenness(...)
        G.gt.flow.push_relabel_max_flow(...)
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
        raise AttributeError(
            f"G.gt has no attribute '{name}'. Only namespaces: {list(self._namespaces.keys())}"
        )

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

        def map_one(x):
            # AnnNet internal id?
            if hasattr(x, "__class__") and x.__class__.__module__.startswith("graph_tool."):
                return x
            if x in self._G.entity_types:
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
        G = self._G
        if getattr(G, "default_label_field", None):
            return G.default_label_field
        va = getattr(G, "vertex_attributes", None)
        if va is None or not hasattr(va, "columns"):
            return None
        for c in ("name", "label", "title", "slug", "external_id", "string_id"):
            if c in va.columns:
                return c
        return None

    def _lookup_vertex_id(self, field, val):
        if field is None:
            return None
        va = self._G.vertex_attributes
        matches = va.filter(va[field] == val)
        if matches.height == 0:
            return None
        id_col = "vertex_id" if "vertex_id" in va.columns else "id" if "id" in va.columns else "vid"
        return matches.select(id_col).item(0, 0)

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

    def _wrap(self, func):
        def call(*args, **kwargs):
            parent = self._parent

            # detect if AnnNet AnnNet passed
            has_owner = any(a is parent._G for a in args) or any(
                v is parent._G for v in kwargs.values()
            )

            gtG = None
            if has_owner:
                gtG = parent._get_or_make_gt()

                # replace G with gtG
                args = tuple(gtG if a is parent._G else a for a in args)
                for k, v in list(kwargs.items()):
                    if v is parent._G:
                        kwargs[k] = gtG

            # bind signature
            try:
                import inspect

                sig = inspect.signature(func)
                bound = sig.bind_partial(*args, **kwargs)
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
