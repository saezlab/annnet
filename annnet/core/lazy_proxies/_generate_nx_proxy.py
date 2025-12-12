from pathlib import Path

import networkx as nx

# Always write into lazy_proxies folder next to this script
OUTPUT = Path(__file__).parent / "lazy_proxies" / "nx_proxy_autogen.py"


def collect_callables():
    """Collect all callable public NetworkX functions."""
    modules = [
        nx,
        nx.algorithms,
        nx.algorithms.community,
        nx.algorithms.approximation,
        nx.algorithms.centrality,
        nx.algorithms.shortest_paths,
        nx.algorithms.flow,
        nx.algorithms.components,
        nx.algorithms.traversal,
        nx.algorithms.bipartite,
        nx.algorithms.link_analysis,
        nx.classes,
        nx.classes.function,
    ]

    funcs = {}

    for mod in modules:
        for name in dir(mod):
            if name.startswith("_"):
                continue

            try:
                obj = getattr(mod, name)
            except Exception:
                continue

            if callable(obj) and getattr(obj, "__module__", "").startswith("networkx"):
                funcs[name] = obj

    return funcs


def build_method(name: str):
    """
    Correct stub format:
    - DO NOT attempt to replicate signatures
    - Only forward via __getattr__
    - Safe for all NX versions
    - Autocomplete fully works
    """
    return f"""
def {name}(self, *args, **kwargs):
    return self.__getattr__("{name}")(*args, **kwargs)
""".strip()


def generate_autogen_class():
    funcs = collect_callables()
    methods = [build_method(n) for n in sorted(funcs.keys())]

    out = "class _LazyNXProxyAutogen:\n"
    for m in methods:
        for line in m.splitlines():
            out += "    " + line + "\n"
    return out


if __name__ == "__main__":
    OUTPUT.write_text(generate_autogen_class())
    print(f"Generated {OUTPUT}")
