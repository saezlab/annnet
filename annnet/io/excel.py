from __future__ import annotations

import pathlib

from .._dataframe_backend import _dataframe_read_excel


def from_excel(
    path: str | pathlib.Path,
    graph=None,
    schema: str = 'auto',
    sheet: str | None = None,
    default_slice=None,
    default_directed=None,
    default_weight: float = 1.0,
    **kwargs,
):
    """Load an Excel (.xlsx/.xls) file by converting it internally to CSV, then building a graph.

    Parameters
    ----------
    path : str or Path
        Path to the Excel file.
    graph : AnnNet, optional
        Existing graph instance. If None, a new one is created.
    schema : {'auto', 'edge_list', 'hyperedge', 'incidence', 'adjacency', 'lil'}, default 'auto'
        AnnNet schema to assume or infer.
    sheet : str, optional
        Sheet name to load. Defaults to the first sheet.
    default_slice : str, optional
        Default slice name if not present.
    default_directed : bool, optional
        Default directedness if not present or inferrable.
    default_weight : float, default 1.0
        Default weight if no weight column exists.
    **kwargs
        Extra keyword arguments passed to the graph constructor.

    Returns
    -------
    AnnNet
        The created or augmented graph.

    Notes
    -----
    - Excel reading is centralized through AnnNet's dataframe backend helpers.
    - Supported formats and schemas are identical to `from_csv`.

    """
    path = pathlib.Path(path)

    from .csv_format import from_dataframe

    return from_dataframe(
        _dataframe_read_excel(path, sheet_name=sheet),
        graph=graph,
        schema=schema,
        default_slice=default_slice,
        default_directed=default_directed,
        default_weight=default_weight,
        **kwargs,
    )
