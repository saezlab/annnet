"""Named feature bindings on AnnData through its public extension API.

Importing this optional adapter registers ``adata.annnet``. Bindings are runtime
objects and are not copied or persisted by AnnData.

This is the registration half of an integration that is still being designed. It
depends on AnnData only through its public extension API, is not imported unless
asked for, and is not a finished AnnData or scverse integration.
"""

from __future__ import annotations

from types import MappingProxyType

from anndata import AnnData, register_anndata_namespace


@register_anndata_namespace('annnet')
class AxisNamespace:
    """Named Space bindings for one AnnData instance."""

    def __init__(self, adata: AnnData):
        self._adata = adata
        self._spaces: dict = {}

    @property
    def spaces(self):
        """Read-only mapping of binding names to Space objects."""
        return MappingProxyType(self._spaces)


def bind(adata, space) -> None:
    """Add a named binding without replacing another graph's binding."""
    if not isinstance(adata, AnnData):
        raise TypeError('Bind each modality separately through its feature axis')
    held = adata.annnet._spaces
    if space.name in held and held[space.name] is not space:
        raise ValueError(f'Space {space.name!r} is already bound; choose another name')
    held[space.name] = space
