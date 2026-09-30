"""Hand a Space to the adapters layer when its axis is an external container.

``annnet.core`` does not import ``annnet.adapters``, and this package is the one
exemption ``tests/test_import_boundaries.py`` grants: its modules are the glue
between a core object and an optional library, imported by the call that reaches
for them and never by ``import annnet``. ``G.nx`` is the precedent. The AnnData
accessor is the same kind of glue in the other direction — an ``annnet``
namespace on an ``anndata`` object — so its registration is imported here, and
only when a Space is bound to something that carries ``var_names``.
"""

from __future__ import annotations

from typing import Any


def bind_axis(axis: Any, space: Any) -> None:
    """Register ``space`` on ``axis`` through the accessor for its container."""
    from ...adapters.anndata_accessor import bind

    bind(axis, space)
