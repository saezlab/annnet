import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

import time

import pytest

from annnet.io.dataframe_io import from_dataframes, to_dataframes  # DF (DataFrame)
from annnet.io.json_io import from_json, to_json  # JSON (JavaScript Object Notation)
from annnet.io.Parquet_io import (
    from_parquet,
    to_parquet,
)  # Parquet (columnar storage)

# Generous soft thresholds (seconds) — only fail on clear regression
_THRESHOLD_COMPLEX = 5.0  # complex_graph (6 vertices, 6 edges) per adapter
_THRESHOLD_LARGE = 30.0  # 5 000 vertices / 10 000 edges round-trip


class TestPerformance:
    """Adapter timing tests: each test enforces a soft ceiling so regressions
    produce a clear failure rather than a silent slowdown."""

    @pytest.mark.slow
    def test_json_round_trip_speed(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        start = time.perf_counter()
        to_json(G, tmpdir_fixture / "perf.json")
        G2 = from_json(tmpdir_fixture / "perf.json")
        elapsed = time.perf_counter() - start

        assert G2.ne == G.ne
        assert elapsed < _THRESHOLD_COMPLEX, (
            f"JSON round-trip took {elapsed:.3f}s — exceeds {_THRESHOLD_COMPLEX}s threshold"
        )

    @pytest.mark.slow
    def test_parquet_round_trip_speed(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        start = time.perf_counter()
        to_parquet(G, tmpdir_fixture / "perf_dir")
        G2 = from_parquet(tmpdir_fixture / "perf_dir")
        elapsed = time.perf_counter() - start

        assert G2.ne == G.ne
        assert elapsed < _THRESHOLD_COMPLEX, (
            f"Parquet round-trip took {elapsed:.3f}s — exceeds {_THRESHOLD_COMPLEX}s threshold"
        )

    @pytest.mark.slow
    def test_dataframe_round_trip_speed(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        start = time.perf_counter()
        dfs = to_dataframes(G)
        G2 = from_dataframes(**dfs)
        elapsed = time.perf_counter() - start

        assert G2.ne == G.ne
        assert elapsed < _THRESHOLD_COMPLEX, (
            f"DataFrame round-trip took {elapsed:.3f}s — exceeds {_THRESHOLD_COMPLEX}s threshold"
        )

    @pytest.mark.slow
    def test_adapter_speed_comparison(self, complex_graph, tmpdir_fixture):
        """Print relative timings so humans can spot regressions in CI logs."""
        G = complex_graph
        results = {}

        start = time.perf_counter()
        to_json(G, tmpdir_fixture / "cmp.json")
        from_json(tmpdir_fixture / "cmp.json")
        results["JSON"] = time.perf_counter() - start

        start = time.perf_counter()
        to_parquet(G, tmpdir_fixture / "cmp_dir")
        from_parquet(tmpdir_fixture / "cmp_dir")
        results["Parquet"] = time.perf_counter() - start

        start = time.perf_counter()
        dfs = to_dataframes(G)
        from_dataframes(**dfs)
        results["DataFrame"] = time.perf_counter() - start

        print("\nAdapter Performance (seconds):")
        for adapter, elapsed in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {adapter}: {elapsed:.4f}s")

        # All adapters must complete within threshold
        for adapter, elapsed in results.items():
            assert elapsed < _THRESHOLD_COMPLEX, (
                f"{adapter} took {elapsed:.3f}s — exceeds {_THRESHOLD_COMPLEX}s threshold"
            )

    @pytest.mark.slow
    def test_large_graph_parquet_speed(self, tmpdir_fixture):
        """Parquet round-trip on a large graph must stay within threshold."""
        import random

        from annnet.core.graph import AnnNet

        G = AnnNet()
        n_v, n_e = 5_000, 10_000
        for i in range(n_v):
            G.add_vertices(f"v{i}")
        random.seed(0)
        for i in range(n_e):
            u = f"v{random.randint(0, n_v - 1)}"  # nosec B311
            v = f"v{random.randint(0, n_v - 1)}"  # nosec B311
            G.add_edges(u, v, edge_id=f"e{i}")

        start = time.perf_counter()
        to_parquet(G, tmpdir_fixture / "large_dir")
        G2 = from_parquet(tmpdir_fixture / "large_dir")
        elapsed = time.perf_counter() - start

        assert G2.nv == n_v
        assert G2.ne == n_e
        assert elapsed < _THRESHOLD_LARGE, (
            f"Large Parquet round-trip took {elapsed:.3f}s — exceeds {_THRESHOLD_LARGE}s threshold"
        )
