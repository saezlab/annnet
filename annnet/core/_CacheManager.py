class CacheManager:
    """Cache manager for materialized views (CSR/CSC)."""

    def __init__(self, graph):
        self._G = graph
        self._csr = None
        self._csc = None
        self._adjacency = None
        self._csr_version = None
        self._csc_version = None
        self._adjacency_version = None

    # ==================== CSR/CSC Properties ====================

    @property
    def csr(self):
        """Get CSR (Compressed Sparse Row) format.
        Builds and caches on first access.
        """
        if self._csr is None or self._csr_version != self._G._version:
            self._csr = self._G._matrix.tocsr()
            self._csr_version = self._G._version
        return self._csr

    @property
    def csc(self):
        """Get CSC (Compressed Sparse Column) format.
        Builds and caches on first access.
        """
        if self._csc is None or self._csc_version != self._G._version:
            self._csc = self._G._matrix.tocsc()
            self._csc_version = self._G._version
        return self._csc

    @property
    def adjacency(self):
        """Get adjacency matrix (computed from incidence).
        For incidence B: adjacency A = B @ B.T
        """
        if self._adjacency is None or self._adjacency_version != self._G._version:
            csr = self.csr
            # Adjacency from incidence: A = B @ B.T
            self._adjacency = csr @ csr.T
            self._adjacency_version = self._G._version
        return self._adjacency

    def has_csr(self) -> bool:
        """True if CSR cache exists and matches current graph version."""
        return self._csr is not None and self._csr_version == self._G._version

    def has_csc(self) -> bool:
        """True if CSC cache exists and matches current graph version."""
        return self._csc is not None and self._csc_version == self._G._version

    def has_adjacency(self) -> bool:
        """True if adjacency cache exists and matches current graph version."""
        return self._adjacency is not None and self._adjacency_version == self._G._version

    def get_csr(self):
        return self.csr

    def get_csc(self):
        return self.csc

    def get_adjacency(self):
        return self.adjacency

    # ==================== Cache Management ====================

    def invalidate(self, formats=None):
        """Invalidate cached formats.

        Parameters
        --
        formats : list[str], optional
            Formats to invalidate ('csr', 'csc', 'adjacency').
            If None, invalidate all.

        """
        if formats is None:
            formats = ["csr", "csc", "adjacency"]

        for fmt in formats:
            if fmt == "csr":
                self._csr = None
                self._csr_version = None
            elif fmt == "csc":
                self._csc = None
                self._csc_version = None
            elif fmt == "adjacency":
                self._adjacency = None
                self._adjacency_version = None

    def build(self, formats=None):
        """Pre-build specified formats (eager caching).

        Parameters
        --
        formats : list[str], optional
            Formats to build ('csr', 'csc', 'adjacency').
            If None, build all.

        """
        if formats is None:
            formats = ["csr", "csc", "adjacency"]

        for fmt in formats:
            if fmt == "csr":
                _ = self.csr
            elif fmt == "csc":
                _ = self.csc
            elif fmt == "adjacency":
                _ = self.adjacency

    def clear(self):
        """Clear all caches."""
        self.invalidate()

    def info(self):
        """Get cache status and memory usage.

        Returns
        ---
        dict
            Status of each cached format

        """

        def _format_info(matrix, version):
            if matrix is None:
                return {"cached": False}

            # Calculate size
            size_bytes = 0
            if hasattr(matrix, "data"):
                size_bytes += matrix.data.nbytes
            if hasattr(matrix, "indices"):
                size_bytes += matrix.indices.nbytes
            if hasattr(matrix, "indptr"):
                size_bytes += matrix.indptr.nbytes

            return {
                "cached": True,
                "version": version,
                "size_mb": size_bytes / (1024**2),
                "nnz": matrix.nnz if hasattr(matrix, "nnz") else 0,
                "shape": matrix.shape,
            }

        return {
            "csr": _format_info(self._csr, self._csr_version),
            "csc": _format_info(self._csc, self._csc_version),
            "adjacency": _format_info(self._adjacency, self._adjacency_version),
        }

