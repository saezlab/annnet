import narwhals as nw

try:
    import polars as pl
except Exception:
    pl = None


class IndexManager:
    """Namespace for index operations.
    Provides clean API over existing dicts.
    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Entity (vertex) Indexes ====================

    def entity_to_row(self, entity_id):
        """Map entity ID to matrix row index."""
        if entity_id not in self._G.entity_to_idx:
            raise KeyError(f"Entity '{entity_id}' not found")
        return self._G.entity_to_idx[entity_id]

    def row_to_entity(self, row):
        """Map matrix row index to entity ID."""
        if row not in self._G.idx_to_entity:
            raise KeyError(f"Row {row} not found")
        return self._G.idx_to_entity[row]

    def entities_to_rows(self, entity_ids):
        """Batch convert entity IDs to row indices."""
        return [self._G.entity_to_idx[eid] for eid in entity_ids]

    def rows_to_entities(self, rows):
        """Batch convert row indices to entity IDs."""
        return [self._G.idx_to_entity[r] for r in rows]

    # ==================== Edge Indexes ====================

    def edge_to_col(self, edge_id):
        """Map edge ID to matrix column index."""
        if edge_id not in self._G.edge_to_idx:
            raise KeyError(f"Edge '{edge_id}' not found")
        return self._G.edge_to_idx[edge_id]

    def col_to_edge(self, col):
        """Map matrix column index to edge ID."""
        if col not in self._G.idx_to_edge:
            raise KeyError(f"Column {col} not found")
        return self._G.idx_to_edge[col]

    def edges_to_cols(self, edge_ids):
        """Batch convert edge IDs to column indices."""
        return [self._G.edge_to_idx[eid] for eid in edge_ids]

    def cols_to_edges(self, cols):
        """Batch convert column indices to edge IDs."""
        return [self._G.idx_to_edge[c] for c in cols]

    # ==================== Utilities ====================

    def entity_type(self, entity_id):
        """Get entity type ('vertex' or 'edge')."""
        if entity_id not in self._G.entity_types:
            raise KeyError(f"Entity '{entity_id}' not found")
        return self._G.entity_types[entity_id]

    def is_vertex(self, entity_id):
        """Check if entity is a vertex."""
        return self.entity_type(entity_id) == "vertex"

    def is_edge_entity(self, entity_id):
        """Check if entity is an edge-entity (vertex-edge hybrid)."""
        return self.entity_type(entity_id) == "edge"

    def has_entity(self, entity_id: str) -> bool:
        """True if the ID exists as any entity (vertex or edge-entity)."""
        return entity_id in self._G.entity_to_idx

    def has_vertex(self, vertex_id: str) -> bool:
        """True if the ID exists and is a vertex (not an edge-entity)."""
        return self._G.entity_types.get(vertex_id) == "vertex"

    def has_edge_id(self, edge_id: str) -> bool:
        """True if an edge with this ID exists."""
        return edge_id in self._G.edge_to_idx

    def edge_count(self) -> int:
        """Number of edges (columns in incidence)."""
        return len(self._G.edge_to_idx)

    def entity_count(self) -> int:
        """Number of entities (vertices + edge-entities)."""
        return len(self._G.entity_to_idx)

    def vertex_count(self) -> int:
        """Number of true vertices (excludes edge-entities)."""
        return sum(1 for t in self._G.entity_types.values() if t == "vertex")

    def stats(self):
        """Get index statistics."""
        return {
            "n_entities": len(self._G.entity_to_idx),
            "n_vertices": sum(1 for t in self._G.entity_types.values() if t == "vertex"),
            "n_edge_entities": sum(1 for t in self._G.entity_types.values() if t == "edge"),
            "n_edges": len(self._G.edge_to_idx),
            "max_row": max(self._G.idx_to_entity.keys()) if self._G.idx_to_entity else -1,
            "max_col": max(self._G.idx_to_edge.keys()) if self._G.idx_to_edge else -1,
        }


class IndexMapping:
    # ID + entity ensure helpers

    def _get_next_edge_id(self) -> str:
        """INTERNAL: Generate a unique edge ID for parallel edges.

        Returns
        ---
        str
            Fresh ``edge_<n>`` identifier (monotonic counter).

        """
        edge_id = f"edge_{self._next_edge_id}"
        self._next_edge_id += 1
        return edge_id

    def _ensure_vertex_table(self) -> None:
        """INTERNAL: Ensure the vertex attribute table exists with a canonical schema.

        Notes
        -
        - Creates an empty Polars DF [DataFrame] with a single ``Utf8`` ``vertex_id`` column
        if missing or malformed.

        """
        df = getattr(self, "vertex_attributes", None)

        needs_init = df is None or not hasattr(df, "columns") or "vertex_id" not in df.columns

        if needs_init:
            try:
                import polars as pl

                self.vertex_attributes = pl.DataFrame({"vertex_id": pl.Series([], dtype=pl.Utf8)})
            except Exception:
                try:
                    import pandas as pd

                    self.vertex_attributes = pd.DataFrame({"vertex_id": pd.Series(dtype="string")})
                except Exception:
                    raise RuntimeError(
                        "Cannot initialize vertex_attributes: install polars (recommended) or pandas."
                    )

    def _ensure_vertex_row(self, vertex_id: str) -> None:
        """INTERNAL: Ensure a row for ``vertex_id`` exists in the vertex attribute DF.

        Notes
        -
        - Appends a new row with ``vertex_id`` and ``None`` for other columns if absent.
        - Preserves existing schema and columns.

        """
        # Intern for cheaper dict ops
        try:
            import sys as _sys

            if isinstance(vertex_id, str):
                vertex_id = _sys.intern(vertex_id)
        except Exception:
            pass

        df = self.vertex_attributes

        # Build/refresh a cached id-set if needed (auto-invalidates on DF object change)
        try:
            cached_ids = getattr(self, "_vertex_attr_ids", None)
            cached_df_id = getattr(self, "_vertex_attr_df_id", None)
            if cached_ids is None or cached_df_id != id(df):
                ids = set()
                try:
                    import polars as pl  # optional
                except Exception:
                    pl = None

                if df is not None and hasattr(df, "columns") and "vertex_id" in df.columns:
                    # One-time scan to seed cache
                    if pl is not None and isinstance(df, pl.DataFrame):
                        if df.height > 0:
                            try:
                                ids = set(df.get_column("vertex_id").to_list())
                            except Exception:
                                ids = set(df.select("vertex_id").to_series().to_list())
                    else:
                        import narwhals as nw

                        ndf = nw.from_native(df)
                        try:
                            ids = set(nw.to_native(ndf.select("vertex_id")).to_series().to_list())
                        except Exception:
                            # fallback: convert to native and pull column
                            native = nw.to_native(ndf)
                            col = native["vertex_id"]
                            ids = set(col.to_list() if hasattr(col, "to_list") else list(col))
                self._vertex_attr_ids = ids
                self._vertex_attr_df_id = id(df)
        except Exception:
            # If anything about caching fails, proceed without it
            self._vertex_attr_ids = None
            self._vertex_attr_df_id = None

        # membership check via cache when available
        ids = getattr(self, "_vertex_attr_ids", None)
        if ids is not None and vertex_id in ids:
            return

        # If DF is empty, create the first row with the canonical schema
        is_empty = False
        try:
            is_empty = df.is_empty()  # polars-like
        except Exception:
            try:
                is_empty = len(df) == 0  # pandas-like
            except Exception:
                is_empty = False

        if is_empty:
            try:
                import polars as pl

                self.vertex_attributes = pl.DataFrame(
                    {"vertex_id": [vertex_id]}, schema={"vertex_id": pl.Utf8}
                )
            except Exception:
                try:
                    import pandas as pd

                    self.vertex_attributes = pd.DataFrame({"vertex_id": [vertex_id]})
                except Exception:
                    raise RuntimeError(
                        "Cannot initialize vertex_attributes row: install polars (recommended) or pandas."
                    )
            # keep cache in sync
            try:
                if isinstance(self._vertex_attr_ids, set):
                    self._vertex_attr_ids.add(vertex_id)
                else:
                    self._vertex_attr_ids = {vertex_id}
                self._vertex_attr_df_id = id(self.vertex_attributes)
            except Exception:
                pass
            return

        # Align columns: create a single dict with all columns present
        row = dict.fromkeys(df.columns)
        row["vertex_id"] = vertex_id

        # Append one row efficiently
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            try:
                new_df = df.vstack(pl.DataFrame([row]))
            except Exception:
                new_df = pl.concat([df, pl.DataFrame([row])], how="vertical")
            self.vertex_attributes = new_df
        else:
            # generic fallback (narwhals -> native)
            try:
                import pandas as pd

                self.vertex_attributes = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            except Exception:
                import narwhals as nw

                ndf = nw.from_native(df)
                nrow = nw.from_native(pd.DataFrame([row]))  # requires pandas
                self.vertex_attributes = nw.to_native(nw.concat([ndf, nrow], how="vertical"))

        # Update cache after mutation
        try:
            if isinstance(self._vertex_attr_ids, set):
                self._vertex_attr_ids.add(vertex_id)
            else:
                self._vertex_attr_ids = {vertex_id}
            self._vertex_attr_df_id = id(self.vertex_attributes)
        except Exception:
            pass

    def _vertex_key_enabled(self) -> bool:
        return bool(self._vertex_key_fields)

    def _build_key_from_attrs(self, attrs: dict) -> tuple | None:
        """Return tuple of field values in declared order, or None if any missing."""
        if not self._vertex_key_fields:
            return None
        vals = []
        for f in self._vertex_key_fields:
            if f not in attrs or attrs[f] is None:
                return None  # incomplete — not indexable
            vals.append(attrs[f])
        return tuple(vals)

    def _current_key_of_vertex(self, vertex_id) -> tuple | None:
        """Read the current key tuple of a vertex from vertex_attributes (None if incomplete)."""
        if not self._vertex_key_fields:
            return None
        cur = {f: self.get_attr_vertex(vertex_id, f, None) for f in self._vertex_key_fields}
        return self._build_key_from_attrs(cur)

    def _gen_vertex_id_from_key(self, key_tuple: tuple) -> str:
        """Deterministic, human-readable vertex_id from a composite key."""
        parts = [f"{f}={repr(v)}" for f, v in zip(self._vertex_key_fields, key_tuple)]
        return "cid:" + "|".join(parts)
