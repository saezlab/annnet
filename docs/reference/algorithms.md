# Algorithms

## Traversal

::: annnet.algorithms.traversal.Traversal
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## Lazy Proxies (NetworkX, igraph, graph-tool)

These proxies provide a thin, lazy bridge to optional backends. They are
exposed through `AnnNet` for convenience but require the corresponding
dependency to be installed.

::: annnet.core.lazy_proxies.nx_lazyproxy._LazyNXProxy
    options:
      members: false
      show_root_heading: true

::: annnet.core.lazy_proxies.ig_lazyproxy._LazyIGProxy
    options:
      members: false
      show_root_heading: true

::: annnet.core.lazy_proxies.gt_lazyproxy._LazyGTProxy
    options:
      members: false
      show_root_heading: true
