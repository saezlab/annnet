"""
OmniPath ingestion helper for AnnNet.

Provides:
    from_omnipath(...) -> AnnNet

This module loads OmniPath-style interaction tables and converts them into an
AnnNet graph. Table handling is routed through AnnNet's dataframe backend so
the implementation does not depend directly on a specific dataframe library.
"""

import time

import narwhals as nw
import numpy as np

from ..core import AnnNet
from ._common import (
    dataframe_height,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_read_tsv,
)


def from_omnipath(
    df=None,
    *,
    dataset: str = 'omnipath',
    include=None,
    exclude=None,
    query: dict | None = None,
    source_col: str | None = None,
    target_col: str | None = None,
    directed_col: str | None = None,
    weight_col: str | None = None,
    edge_id_col: str | None = None,
    slice_col: str | None = None,
    slice: str | None = None,
    default_directed: bool = True,
    edge_attr_cols: list[str] | None = None,
    dropna: bool = True,
    annotations_backend: str | None = None,
    vertex_annotations_df=None,
    vertex_annotations_path: str | None = None,
    vertex_annotation_sources: list[str] | None = None,
    load_vertex_annotations: bool = True,
    **graph_kwargs,
):
    r"""Build an AnnNet from OmniPath interaction data, with edge and vertex annotations.

    Fetches a signaling interaction dataset from the OmniPath web service (or accepts
    a pre-loaded DataFrame), builds the graph structure via bulk operations, and
    optionally enriches vertices with annotations from the OmniPath annotation archive.

    The annotation archive (~114MB) is downloaded once and cached at
    ``~/.cache/annnet/omnipath_annotations.tsv.gz`` for fast subsequent loads.

    Parameters
    ----------
    df : DataFrame-like, optional
        If provided, skip the OmniPath network request and build from this table.
        Must contain at least source and target columns. Accepts any dataframe-like
        object supported by AnnNet's dataframe backend.
    dataset : str, optional
        OmniPath interaction dataset to fetch. One of:
        ``"omnipath"`` (default, curated core), ``"all"``, ``"posttranslational"``,
        ``"pathwayextra"``, ``"kinaseextra"``, ``"ligrecextra"``, ``"dorothea"``,
        ``"tftarget"``, ``"transcriptional"``, ``"tfmirna"``, ``"mirna"``,
        ``"lncrnamrna"``, ``"collectri"``.
    include, exclude : optional
        Dataset include/exclude filters. Only used when ``dataset="all"``
        (include/exclude) or ``dataset="posttranslational"`` (exclude only).
    query : dict, optional
        Extra query parameters forwarded to the OmniPath web service.
        Example: ``{"organism": "human", "genesymbols": True}``.
        Use ``omnipath.interactions.<Dataset>.params()`` to inspect available keys.
    source_col : str, optional
        Column name for source node identifiers. Auto-detected from common
        OmniPath field names if omitted (``source``, ``source_genesymbol``, etc.).
    target_col : str, optional
        Column name for target node identifiers. Auto-detected if omitted.
    directed_col : str, optional
        Column holding per-edge directedness flags (bool-like). If omitted,
        ``default_directed`` is used for all edges.
    weight_col : str, optional
        Column holding edge weights. If omitted, weight defaults to 1.0.
    edge_id_col : str, optional
        Column holding stable edge identifiers. If omitted, AnnNet assigns
        sequential IDs (``edge_0``, ``edge_1``, ...).
    slice_col : str, optional
        Column holding per-edge slice identifiers.
    slice : str, optional
        Slice to place all edges into. Ignored if ``slice_col`` is provided.
    default_directed : bool, optional
        Fallback directedness when ``directed_col`` is missing or null.
        Defaults to ``True``.
    edge_attr_cols : list[str], optional
        Columns to store as edge attributes. If omitted, all non-structural
        columns are used. Pass ``[]`` to skip edge attribute loading entirely.
    dropna : bool, optional
        If ``True`` (default), silently drop rows with null source or target.
        If ``False``, raise ``ValueError`` on first null endpoint.
    annotations_backend : str, optional
        Backend for AnnNet attribute tables. ``None`` uses AnnNet's configured
        dataframe backend default.
    vertex_annotations_df : DataFrame-like, optional
        Pre-loaded annotation table in OmniPath long format
        ``(genesymbol, source, label, value)``. Skips all file I/O.
        Fastest option when rebuilding the graph multiple times in one session —
        load the archive once and pass it here.
    vertex_annotations_path : str, optional
        Path to a local OmniPath annotation file (``.tsv`` or ``.tsv.gz``).
        Skips the cache-check and download.
    vertex_annotation_sources : list[str], optional
        OmniPath annotation resource names to include as vertex attributes.
        If omitted, all resources in the annotation table are loaded (254 columns).
        Recommended subset for signaling graphs::

            [
                'HGNC',
                'CancerGeneCensus',
                'SignaLink_function',
                'SignaLink_pathway',
                'UniProt_location',
                'HPA_subcellular',
                'PROGENy',
                'IntOGen',
                'Phosphatome',
                'kinase.com',
            ]

    load_vertex_annotations : bool, optional
        Whether to load vertex annotations at all. Set to ``False`` to skip
        annotation loading entirely and get a structure-only graph. Default ``True``.
    **graph_kwargs
        Additional keyword arguments forwarded to the ``AnnNet`` constructor.

    Returns
    -------
    AnnNet
        Fully constructed graph with:
        - Vertices: one per unique gene symbol encountered as source or target.
        - Edges: one per interaction row, with incidence matrix weights encoding
          direction (+w source, −w target for directed; +w both for undirected).
        - ``edge_attributes``: dataframe-backed table with one row per edge and one column
          per entry in ``edge_attr_cols``.
        - ``vertex_attributes``: dataframe-backed table with one row per vertex and one column
          per ``(source:label)`` annotation pair from the requested resources.

    Notes
    -----
    - Edge and vertex attribute tables are populated via bulk operations —
      no per-row DataFrame allocations occur during loading.
    - History tracking is disabled during construction and re-enabled on return.
    - The annotation archive is ~114MB compressed. First call downloads and caches
      it; subsequent calls load from ``~/.cache/annnet/omnipath_annotations.tsv.gz``
      in ~2–3s.
    - ``source`` and ``target`` columns from the interaction table are redundant
      as edge attributes (the graph structure already encodes them). Exclude them
      via ``edge_attr_cols`` if not needed.

    See Also
    --------
    AnnNet, AnnNet.add_edges_bulk, AnnNet.add_vertices_bulk

    Examples
    --------
    Minimal load (structure only, no annotations)::

        G = from_omnipath(load_vertex_annotations=False)

    Full load with curated vertex annotation sources::

        G = from_omnipath(
            dataset='omnipath',
            query={'organism': 'human', 'genesymbols': True},
            source_col='source_genesymbol',
            target_col='target_genesymbol',
            edge_attr_cols=[
                'is_stimulation',
                'is_inhibition',
                'consensus_direction',
                'n_sources',
                'n_references',
                'sources',
                'references_stripped',
            ],
            vertex_annotation_sources=[
                'HGNC',
                'CancerGeneCensus',
                'SignaLink_function',
                'UniProt_location',
                'HPA_subcellular',
                'IntOGen',
            ],
        )

    Pass a pre-loaded annotation table to avoid repeated downloads::

        ann = pl.read_csv('~/.cache/annnet/omnipath_annotations.tsv.gz', separator='\\t')
        G = from_omnipath(vertex_annotations_df=ann)

    Build from a custom DataFrame instead of fetching from OmniPath::

        import pandas as pd

        df = pd.DataFrame(
            {
                'source': ['EGFR', 'TP53'],
                'target': ['STAT3', 'MDM2'],
                'is_directed': [True, True],
            }
        )
        G = from_omnipath(df=df, load_vertex_annotations=False)

    """

    def _pick_col(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    def _is_null(x):
        if x is None:
            return True
        try:
            if isinstance(x, (float, np.floating)):
                return bool(np.isnan(x))
        except Exception:  # noqa: BLE001
            pass
        return False

    def _stringify_edge_id(x):
        if _is_null(x):
            return None
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            if np.isfinite(x) and float(x).is_integer():
                return str(int(x))
            return str(x)
        return str(x)

    def _coerce_bool(x, default):
        if x is None:
            return default
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, np.integer)):
            return bool(x)
        if isinstance(x, str):
            v = x.strip().lower()
            if v in {'1', 'true', 't', 'yes', 'y', 'directed', 'dir'}:
                return True
            if v in {'0', 'false', 'f', 'no', 'n', 'undirected', 'undir', 'u'}:
                return False
        return default

    # fetch
    t_fetch0 = time.perf_counter()

    if df is None:
        try:
            from omnipath import interactions as opi
        except Exception as exc:
            raise ImportError(
                'omnipath package is required to fetch from the OmniPath web service. '
                'Install with `pip install omnipath` or pass a dataframe as `df=`.'
            ) from exc

        def _norm(s):
            return s.lower().replace('-', '').replace('_', '')

        dataset_key = _norm(dataset)
        classes = {
            'omnipath': opi.OmniPath,
            'all': opi.AllInteractions,
            'posttranslational': opi.PostTranslational,
            'pathwayextra': opi.PathwayExtra,
            'kinaseextra': opi.KinaseExtra,
            'ligrecextra': opi.LigRecExtra,
            'dorothea': opi.Dorothea,
            'tftarget': opi.TFtarget,
            'transcriptional': opi.Transcriptional,
            'tfmirna': opi.TFmiRNA,
            'mirna': opi.miRNA,
            'lncrnamrna': opi.lncRNAmRNA,
            'collectri': opi.CollecTRI,
        }

        if dataset_key not in classes:
            raise ValueError(f'Unknown dataset {dataset!r}. Try one of: {sorted(classes.keys())}')

        query = query or {}
        cls = classes[dataset_key]
        if dataset_key == 'all':
            df = cls.get(include=include, exclude=exclude, **query)
        elif dataset_key == 'posttranslational':
            df = cls.get(exclude=exclude, **query)
        else:
            df = cls.get(**query)

    print(f'[timing] fetch/receive df:     {time.perf_counter() - t_fetch0:.3f}s')

    # column resolution
    t_cols0 = time.perf_counter()

    cols = dataframe_columns(df)

    if source_col is None:
        source_col = _pick_col(
            cols,
            [
                'source',
                'source_genesymbol',
                'source_gene_symbol',
                'source_gene',
                'source_uniprot',
                'source_id',
            ],
        )
    if target_col is None:
        target_col = _pick_col(
            cols,
            [
                'target',
                'target_genesymbol',
                'target_gene_symbol',
                'target_gene',
                'target_uniprot',
                'target_id',
            ],
        )

    if source_col is None or target_col is None:
        raise ValueError(
            'Could not infer source/target columns. Pass source_col and target_col explicitly.'
        )

    if directed_col is None:
        directed_col = _pick_col(cols, ['is_directed', 'directed', 'consensus_direction'])
    if weight_col is None:
        weight_col = _pick_col(cols, ['weight', 'consensus_weight', 'score'])
    if edge_id_col is None:
        edge_id_col = _pick_col(cols, ['edge_id', 'interaction_id', 'id'])
    if slice_col is None:
        slice_col = _pick_col(cols, ['slice', 'slice_id'])

    if edge_attr_cols is None:
        structural = {source_col, target_col, directed_col, weight_col, edge_id_col, slice_col}
        edge_attr_cols = [c for c in cols if c not in structural]

    print(f'[timing] column resolution:    {time.perf_counter() - t_cols0:.4f}s')
    print(f'         source={source_col!r}  target={target_col!r}  directed={directed_col!r}')
    print(f'         edge_attr_cols ({len(edge_attr_cols)}): {edge_attr_cols}')

    # AnnNet init
    t_init0 = time.perf_counter()

    n_rows = dataframe_height(df)
    G = AnnNet(
        directed=default_directed,
        annotations_backend=annotations_backend,
        n=n_rows,
        e=n_rows,
        **graph_kwargs,
    )
    G._history_enabled = False
    print(
        f'[timing] AnnNet init:          {time.perf_counter() - t_init0:.3f}s  (pre-sized n={n_rows} e={n_rows})'
    )

    t_dicts0 = time.perf_counter()
    # Try to iterate the dataframe lazily via narwhals. dataframe_to_rows() in _common
    # materialises the entire dataframe as list[dict] -- fine for 100k rows but blows up
    # for dataset='all' / 'posttranslational' (millions of rows) when combined with the
    # bulk list being built below. Streaming via iter_rows keeps peak memory ~constant.
    edge_attr_set = set(edge_attr_cols)
    try:
        df_nw = nw.from_native(df, eager_only=True, pass_through=False)
        row_iter = df_nw.iter_rows(named=True)
        # narwhals iter_rows is a generator; I can't get a row count up front without
        # walking it, so use the dataframe height instead.
        rows_height = df_nw.shape[0]
        streaming = True
    except Exception:  # noqa: BLE001
        # Backend doesn't speak narwhals -> fall back to the eager path.
        rows = dataframe_to_rows(df)
        row_iter = iter(rows)
        rows_height = len(rows)
        streaming = False
    print(
        f'[timing] to_rows setup:        {time.perf_counter() - t_dicts0:.3f}s  '
        f'({rows_height} rows, streaming={streaming})'
    )

    # bulk list build
    t_bulk0 = time.perf_counter()
    bulk = []
    bulk_append = bulk.append  # local-name lookup, tighter inner loop
    for row in row_iter:
        s = row.get(source_col)
        t = row.get(target_col)
        if dropna and (_is_null(s) or _is_null(t)):
            continue
        if _is_null(s) or _is_null(t):
            raise ValueError('Found null source/target with dropna=False.')

        edge_dir = (
            _coerce_bool(row.get(directed_col), default_directed)
            if directed_col
            else default_directed
        )
        w_raw = row.get(weight_col) if weight_col else None
        w = 1.0 if _is_null(w_raw) else float(w_raw)
        eid = None
        if edge_id_col:
            eid = _stringify_edge_id(row.get(edge_id_col))

        edge_slice = slice
        if slice_col:
            s_raw = row.get(slice_col)
            if not _is_null(s_raw):
                edge_slice = str(s_raw)

        # Only build the attributes dict if there's anything to put in it.
        # Empty edge_attr_cols is a common "structure-only" code path and
        # the comprehension cost adds up over millions of edges.
        if edge_attr_set:
            attributes = {c: row.get(c) for c in edge_attr_cols if c in row}
        else:
            attributes = {}

        bulk_append(
            {
                'source': str(s),
                'target': str(t),
                'weight': w,
                'edge_id': eid,
                'edge_directed': edge_dir,
                'slice': edge_slice,
                'attributes': attributes,
            }
        )

    print(
        f'[timing] bulk list build:      {time.perf_counter() - t_bulk0:.3f}s  ({len(bulk)} edges)'
    )

    # add_edges_bulk
    t_aeb0 = time.perf_counter()
    G.add_edges_bulk(bulk)
    print(f'[timing] add_edges_bulk:       {time.perf_counter() - t_aeb0:.3f}s')

    G._history_enabled = True

    print(f'         vertices={G._num_entities}  edges={G._num_edges}')

    # vertex table: register all vertices
    all_vids = [vid for vid, t in G.entity_types.items() if t == 'vertex']
    G.add_vertices_bulk(all_vids)

    # vertex annotations
    if load_vertex_annotations:
        ann_raw = None
        # True when ann_raw is already the (gene, attr_key, joined) aggregated frame
        # produced by the lazy scan -- the downstream block then skips straight to the pivot.
        ann_pre_aggregated = False

        # Set-dedup the vids and source filter once so both the lazy and eager paths use the same lists.
        _vids_set = set(all_vids)
        _source_filter_set = (
            set(vertex_annotation_sources) if vertex_annotation_sources is not None else None
        )

        def _scan_and_aggregate(path):
            """Lazy Polars scan + pushed-down filter + streaming group_by.

            Used for the cache-hit and download-then-write-to-disk paths so the 217 MB
            gzip never gets fully decompressed into RAM -- only filtered rows do.
            Returns the aggregated Polars DataFrame: (genesymbol, attr_key, joined).
            """
            import polars as pl

            lf = pl.scan_csv(path, separator='\t', has_header=True)

            f_expr = (
                pl.col('genesymbol').is_in(list(_vids_set))
                & pl.col('source').is_not_null()
                & pl.col('label').is_not_null()
                & pl.col('value').is_not_null()
            )
            if _source_filter_set is not None:
                f_expr = f_expr & pl.col('source').is_in(list(_source_filter_set))

            return (
                lf.filter(f_expr)
                .select(
                    [
                        pl.col('genesymbol'),
                        (pl.col('source') + pl.lit(':') + pl.col('label')).alias('attr_key'),
                        pl.col('value').cast(pl.Utf8),
                    ]
                )
                .group_by(['genesymbol', 'attr_key'])
                .agg(pl.col('value').unique().sort().str.join(';').alias('joined'))
                .collect(streaming=True)
            )

        # 1) caller passed a pre-loaded DF directly -- can't push down, hand to eager path
        if vertex_annotations_df is not None:
            ann_raw = vertex_annotations_df

        # 2) caller passed a local file path -- try lazy scan, fall back to eager read
        elif vertex_annotations_path is not None:
            try:
                ann_raw = _scan_and_aggregate(vertex_annotations_path)
                ann_pre_aggregated = True
                print(
                    f'[vertex annotations] streamed from path: {vertex_annotations_path}  '
                    f'rows={ann_raw.height}'
                )
            except Exception as e:  # noqa: BLE001
                print(f'[warning] lazy scan of {vertex_annotations_path} failed ({e}); falling back to eager read')
                try:
                    ann_raw = dataframe_read_tsv(vertex_annotations_path, backend=annotations_backend)
                except Exception as e2:  # noqa: BLE001
                    print(f'[warning] vertex_annotations_path failed: {e2}')

        # 3) check cache first, then download from OmniPath archive
        else:
            try:
                import os
                import requests as _requests

                _ANN_URL = (
                    'https://archive.omnipathdb.org/omnipath_webservice_annotations__latest.tsv.gz'
                )
                _cache_path = os.path.join(
                    os.path.expanduser('~'), '.cache', 'annnet', 'omnipath_annotations.tsv.gz'
                )

                if os.path.exists(_cache_path):
                    print(f'[vertex annotations] loading from cache (lazy scan + pushdown): {_cache_path}')
                    t_ann = time.perf_counter()
                    ann_raw = _scan_and_aggregate(_cache_path)
                    ann_pre_aggregated = True
                    print(
                        f'[vertex annotations] streamed + aggregated in '
                        f'{time.perf_counter() - t_ann:.1f}s  rows={ann_raw.height}'
                    )
                else:
                    print(
                        '[vertex annotations] downloading from OmniPath archive (~114MB, one-time)...'
                    )
                    t_ann = time.perf_counter()
                    # Stream the download straight to disk -- the previous code read
                    # resp.content into memory AND then re-fed it via BytesIO, doubling RAM.
                    resp = _requests.get(_ANN_URL, stream=True, timeout=(5, 60))
                    resp.raise_for_status()
                    os.makedirs(os.path.dirname(_cache_path), exist_ok=True)
                    with open(_cache_path, 'wb') as _f:
                        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                            if chunk:
                                _f.write(chunk)
                    print(
                        f'[vertex annotations] downloaded + cached in '
                        f'{time.perf_counter() - t_ann:.1f}s  → {_cache_path}'
                    )
                    # Now run the same lazy aggregate against the file on disk.
                    t_ann2 = time.perf_counter()
                    ann_raw = _scan_and_aggregate(_cache_path)
                    ann_pre_aggregated = True
                    print(
                        f'[vertex annotations] streamed + aggregated in '
                        f'{time.perf_counter() - t_ann2:.1f}s  rows={ann_raw.height}'
                    )
            except Exception as e:  # noqa: BLE001
                print(f'[warning] vertex annotations download failed: {e}')

        if ann_raw is not None:
            try:
                vids_set = _vids_set  # reuse the set built above

                if ann_pre_aggregated:
                    # Already (genesymbol, attr_key, joined) from the lazy scan path.
                    # Skip the filter+group_by; go straight to the pivot.
                    ann_agg = nw.from_native(ann_raw, eager_only=True, pass_through=False)
                    print(
                        f'[vertex annotations] using pre-aggregated frame: '
                        f'{ann_agg.shape[0]:,} (gene, attr) pairs'
                    )
                else:
                    # Eager path: caller passed an already-loaded DataFrame (or the lazy scan
                    # fell back). Do the filter + aggregate via narwhals here.
                    source_filter = (
                        list(_source_filter_set) if _source_filter_set is not None else None
                    )

                    t_filter = time.perf_counter()
                    ann_nw = nw.from_native(ann_raw, eager_only=True, pass_through=False)

                    expr = (
                        nw.col('genesymbol').is_in(list(vids_set))
                        & ~nw.col('source').is_null()
                        & ~nw.col('label').is_null()
                        & ~nw.col('value').is_null()
                    )
                    if source_filter is not None:
                        expr = expr & nw.col('source').is_in(source_filter)

                    ann_agg = (
                        ann_nw.filter(expr)
                        .select(
                            [
                                nw.col('genesymbol'),
                                (nw.col('source') + nw.lit(':') + nw.col('label')).alias('attr_key'),
                                nw.col('value').cast(nw.String),
                            ]
                        )
                        .group_by(['genesymbol', 'attr_key'])
                        .agg(nw.col('value').unique().sort().str.join(';').alias('joined'))
                    )
                    print(
                        f'[vertex annotations] aggregated to {ann_agg.shape[0]:,} (gene, attr) pairs '
                        f'in {time.perf_counter() - t_filter:.1f}s'
                    )

                # Pivot the small aggregated frame -> (gene, {attr_key: joined}).
                # n_genes * n_attr_keys, bounded by graph size, not by raw annotation rows.
                t_pivot = time.perf_counter()
                grouped: dict[str, dict[str, str]] = {}
                for row in ann_agg.iter_rows(named=True):
                    gene = row['genesymbol']
                    if gene not in vids_set:
                        continue
                    grouped.setdefault(gene, {})[row['attr_key']] = row['joined']
                print(
                    f'[vertex annotations] pivoted in {time.perf_counter() - t_pivot:.1f}s'
                )

                t_load = time.perf_counter()
                payload = [
                    (gene, attrs)
                    for gene, attrs in grouped.items()
                    if G._resolve_entity_key(gene) in G._entities
                ]
                G.add_vertices_bulk(payload)
                print(
                    f'[vertex annotations] loaded {len(payload)} vertices '
                    f'in {time.perf_counter() - t_load:.1f}s'
                )
            except Exception as e:  # noqa: BLE001
                print(f'[warning] vertex annotation pivot/load failed: {e}')

    return G