import time
from typing import Optional

import narwhals as nw
import numpy as np
import os
from ..core.graph import AnnNet

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None

def read_omnipath(
    df=None,
    *,
    dataset: str = "omnipath",
    include=None,
    exclude=None,
    query: Optional[dict] = None,
    source_col: Optional[str] = None,
    target_col: Optional[str] = None,
    directed_col: Optional[str] = None,
    weight_col: Optional[str] = None,
    edge_id_col: Optional[str] = None,
    slice_col: Optional[str] = None,
    slice: Optional[str] = None,
    default_directed: bool = True,
    edge_attr_cols: Optional[list[str]] = None,
    dropna: bool = True,
    annotations_backend: str = "polars",
    vertex_annotations_df=None,
    vertex_annotations_path: Optional[str] = None,
    vertex_annotation_sources: Optional[list[str]] = None,
    load_vertex_annotations: bool = True,
    **graph_kwargs,
):
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
        except Exception:
            pass
        return False

    def _coerce_bool(x, default):
        if x is None:
            return default
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, np.integer)):
            return bool(x)
        if isinstance(x, str):
            v = x.strip().lower()
            if v in {"1", "true", "t", "yes", "y", "directed", "dir"}:
                return True
            if v in {"0", "false", "f", "no", "n", "undirected", "undir", "u"}:
                return False
        return default

    def _to_dicts(native, ndf):
        if pl is not None and isinstance(native, pl.DataFrame):
            return native.to_dicts()
        if hasattr(native, "to_dict"):
            try:
                return native.to_dict(orient="records")
            except Exception:
                pass
        if hasattr(ndf, "to_dicts"):
            return ndf.to_dicts()
        raise TypeError("Unsupported dataframe type for OmniPath import")

    # fetch
    t_fetch0 = time.perf_counter()

    if df is None:
        try:
            from omnipath import interactions as opi
        except Exception as exc:
            raise ImportError(
                "omnipath package is required to fetch from the OmniPath web service. "
                "Install with `pip install omnipath` or pass a dataframe as `df=`."
            ) from exc

        def _norm(s):
            return s.lower().replace("-", "").replace("_", "")

        dataset_key = _norm(dataset)
        classes = {
            "omnipath": opi.OmniPath,
            "all": opi.AllInteractions,
            "posttranslational": opi.PostTranslational,
            "pathwayextra": opi.PathwayExtra,
            "kinaseextra": opi.KinaseExtra,
            "ligrecextra": opi.LigRecExtra,
            "dorothea": opi.Dorothea,
            "tftarget": opi.TFtarget,
            "transcriptional": opi.Transcriptional,
            "tfmirna": opi.TFmiRNA,
            "mirna": opi.miRNA,
            "lncrnamrna": opi.lncRNAmRNA,
            "collectri": opi.CollecTRI,
        }

        if dataset_key not in classes:
            raise ValueError(
                f"Unknown dataset {dataset!r}. Try one of: {sorted(classes.keys())}"
            )

        query = query or {}
        cls = classes[dataset_key]
        if dataset_key == "all":
            df = cls.get(include=include, exclude=exclude, **query)
        elif dataset_key == "posttranslational":
            df = cls.get(exclude=exclude, **query)
        else:
            df = cls.get(**query)

    print(f"[timing] fetch/receive df:     {time.perf_counter()-t_fetch0:.3f}s")

    # column resolution
    t_cols0 = time.perf_counter()

    ndf = nw.from_native(df, eager_only=True)
    native = nw.to_native(ndf)
    cols = list(getattr(native, "columns", ndf.columns))

    if source_col is None:
        source_col = _pick_col(cols, ["source", "source_genesymbol", "source_gene_symbol", "source_gene", "source_uniprot", "source_id"])
    if target_col is None:
        target_col = _pick_col(cols, ["target", "target_genesymbol", "target_gene_symbol", "target_gene", "target_uniprot", "target_id"])

    if source_col is None or target_col is None:
        raise ValueError("Could not infer source/target columns. Pass source_col and target_col explicitly.")

    if directed_col is None:
        directed_col = _pick_col(cols, ["is_directed", "directed", "consensus_direction"])
    if weight_col is None:
        weight_col = _pick_col(cols, ["weight", "consensus_weight", "score"])
    if edge_id_col is None:
        edge_id_col = _pick_col(cols, ["edge_id", "interaction_id", "id"])
    if slice_col is None:
        slice_col = _pick_col(cols, ["slice", "slice_id"])

    if edge_attr_cols is None:
        structural = {source_col, target_col, directed_col, weight_col, edge_id_col, slice_col}
        edge_attr_cols = [c for c in cols if c not in structural]

    print(f"[timing] column resolution:    {time.perf_counter()-t_cols0:.4f}s")
    print(f"         source={source_col!r}  target={target_col!r}  directed={directed_col!r}")
    print(f"         edge_attr_cols ({len(edge_attr_cols)}): {edge_attr_cols}")

    # AnnNet init
    t_init0 = time.perf_counter()
    G = AnnNet(directed=default_directed, annotations_backend=annotations_backend, n=len(ndf), e=len(ndf), **graph_kwargs)
    G._history_enabled = False
    print(f"[timing] AnnNet init:          {time.perf_counter()-t_init0:.3f}s  (pre-sized n={len(ndf)} e={len(ndf)})")

    # _to_dicts
    t_dicts0 = time.perf_counter()
    rows = _to_dicts(native, ndf)
    print(f"[timing] _to_dicts:            {time.perf_counter()-t_dicts0:.3f}s  ({len(rows)} rows)")

    # bulk list build
    t_bulk0 = time.perf_counter()
    bulk = []
    for row in rows:
        s = row.get(source_col)
        t = row.get(target_col)
        if dropna and (_is_null(s) or _is_null(t)):
            continue
        if _is_null(s) or _is_null(t):
            raise ValueError("Found null source/target with dropna=False.")

        edge_dir = _coerce_bool(row.get(directed_col), default_directed) if directed_col else default_directed
        w_raw = row.get(weight_col) if weight_col else None
        w = 1.0 if _is_null(w_raw) else float(w_raw)
        eid = None
        if edge_id_col:
            eid_raw = row.get(edge_id_col)
            if not _is_null(eid_raw):
                eid = str(eid_raw)
        edge_slice = slice
        if slice_col:
            s_raw = row.get(slice_col)
            if not _is_null(s_raw):
                edge_slice = str(s_raw)

        bulk.append({
            "source": str(s),
            "target": str(t),
            "weight": w,
            "edge_id": eid,
            "edge_directed": edge_dir,
            "slice": edge_slice,
            "attributes": {c: row.get(c) for c in edge_attr_cols if c in row},
        })

    print(f"[timing] bulk list build:      {time.perf_counter()-t_bulk0:.3f}s  ({len(bulk)} edges)")

    # add_edges_bulk
    t_aeb0 = time.perf_counter()
    G.add_edges_bulk(bulk)
    print(f"[timing] add_edges_bulk:       {time.perf_counter()-t_aeb0:.3f}s")

    G._history_enabled = True

    print(f"         vertices={G._num_entities}  edges={G._num_edges}")

    # vertex table: register all vertices
    all_vids = [vid for vid, t in G.entity_types.items() if t == "vertex"]
    G.add_vertices_bulk(all_vids)

    # vertex annotations
    if load_vertex_annotations:
        ann_raw = None

        # 1) caller passed a pre-loaded DF directly
        if vertex_annotations_df is not None:
            try:
                ann_raw = nw.from_native(vertex_annotations_df, eager_only=True)
                ann_raw = nw.to_native(ann_raw)  # keep as native (polars or pandas)
            except Exception as e:
                print(f"[warning] vertex_annotations_df could not be read: {e}")

        # 2) caller passed a local file path
        elif vertex_annotations_path is not None:
            try:
                ann_raw = pl.read_csv(vertex_annotations_path, separator="\t")
            except Exception as e:
                print(f"[warning] vertex_annotations_path failed: {e}")
    
        # 3) check cache first, then download from OmniPath archive
        else:
            try:
                import requests as _requests
                import io
                import os

                _ANN_URL = "https://archive.omnipathdb.org/omnipath_webservice_annotations__latest.tsv.gz"
                _cache_path = os.path.join(os.path.expanduser("~"), ".cache", "annnet", "omnipath_annotations.tsv.gz")

                if os.path.exists(_cache_path):
                    print(f"[vertex annotations] loading from cache: {_cache_path}")
                    t_ann = time.perf_counter()
                    ann_raw = pl.read_csv(_cache_path, separator="\t")
                    print(f"[vertex annotations] loaded in {time.perf_counter()-t_ann:.1f}s  shape={ann_raw.shape}")
                else:
                    print(f"[vertex annotations] downloading from OmniPath archive (~114MB, one-time)...")
                    t_ann = time.perf_counter()
                    resp = _requests.get(_ANN_URL, stream=True)
                    resp.raise_for_status()
                    os.makedirs(os.path.dirname(_cache_path), exist_ok=True)
                    with open(_cache_path, "wb") as _f:
                        _f.write(resp.content)
                    ann_raw = pl.read_csv(io.BytesIO(resp.content), separator="\t")
                    print(f"[vertex annotations] downloaded + cached in {time.perf_counter()-t_ann:.1f}s  → {_cache_path}")
            except Exception as e:
                print(f"[warning] vertex annotations download failed: {e}")

        if ann_raw is not None:
            try:
                if not isinstance(ann_raw, pl.DataFrame):
                    ann_raw = pl.from_pandas(ann_raw)

                vids_set = set(all_vids)

                # filter to graph vertices + requested sources
                mask = pl.col("genesymbol").is_in(vids_set)
                if vertex_annotation_sources is not None:
                    mask = mask & pl.col("source").is_in(vertex_annotation_sources)
                ann = ann_raw.filter(mask)

                # pivot: one row per protein, one col per source:label
                flat = (
                    ann
                    .with_columns(
                        pl.concat_str([pl.col("source"), pl.col("label")], separator=":").alias("attr_key")
                    )
                    .group_by(["genesymbol", "attr_key"])
                    .agg(pl.col("value").cast(pl.Utf8).drop_nulls().unique().sort().str.join(";"))
                    .pivot(on="attr_key", index="genesymbol", values="value")
                    .rename({"genesymbol": "vertex_id"})
                )

                G.add_vertices_bulk([
                    (row["vertex_id"], {k: v for k, v in row.items() if k != "vertex_id" and v is not None})
                    for row in flat.to_dicts()
                    if row["vertex_id"] in G.entity_to_idx
                ])
                print(f"[vertex annotations] loaded  shape={G.vertex_attributes.shape}")
            except Exception as e:
                print(f"[warning] vertex annotation pivot/load failed: {e}")

    return G
