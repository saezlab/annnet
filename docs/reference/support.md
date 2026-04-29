# Support Helpers

These helpers expose package metadata plus backend-selection utilities for
dataframe and plotting support. Use them through the documented top-level
`annnet` namespace.

Their implementations live in private `annnet._support.*` modules. Direct
imports from those underscore-prefixed modules follow the
[internal API policy](api-boundary.md).

## Package Metadata

```python
annnet.get_metadata()
```

Return normalized package metadata.

```python
annnet.info()
```

Return a package component summary. The returned object renders
as plain text in terminals and as HTML in compatible notebook frontends.

```python
annnet.get_latest_version()
```

Fetch the latest version declared on the default branch.

## Dataframe Backends

```python
annnet.available_dataframe_backends()
```

Return installed dataframe backends in AnnNet preference order.

```python
annnet.select_dataframe_backend(preferred="auto")
```

Resolve a dataframe backend name. `"auto"` selects the first installed backend
in AnnNet's preference order.

```python
annnet.get_default_dataframe_backend()
```

Return the configured default dataframe backend.

```python
annnet.set_default_dataframe_backend(backend="auto") 
```

Set the default dataframe backend for new AnnNet annotation tables.

## Plotting Backends

```python
annnet.available_plot_backends()
```

Return installed plotting backends in AnnNet preference order.

```python
annnet.select_plot_backend(preferred="auto")
```

Resolve a plotting backend name. `"auto"` selects the first installed backend
in AnnNet's preference order.

```python
annnet.get_default_plot_backend()
```

Return the configured default plotting backend.

```python
annnet.set_default_plot_backend(backend="auto")
```

Set the default backend used by `plot(..., backend=None)`.
