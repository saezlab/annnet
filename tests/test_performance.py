import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

import time

import pytest

from annnet.io.dataframe_io import from_dataframes, to_dataframes  # DF (DataFrame)
from annnet.io.GraphDir_Parquet_io import (
    from_parquet_graphdir,
    to_parquet_graphdir,
)  # Parquet (columnar storage)
from annnet.io.json_io import from_json, to_json  # JSON (JavaScript Object Notation)


class TestPerformance:
    """Optional performance comparison tests."""

    @pytest.mark.slow
    def test_adapter_speed_comparison(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        results = {}
        start = time.time()
        to_json(G, tmpdir_fixture / "perf.json")
        from_json(tmpdir_fixture / "perf.json")
        results["JSON"] = time.time() - start

        start = time.time()
        to_parquet_graphdir(G, tmpdir_fixture / "perf_dir")
        H = from_parquet_graphdir(tmpdir_fixture / "perf_dir")
        # optional sanity:
        assert H.number_of_edges() == G.number_of_edges()
        results["Parquet"] = time.time() - start

        start = time.time()
        dfs = to_dataframes(G)
        from_dataframes(**dfs)
        results["DataFrame"] = time.time() - start

        print("\nAdapter Performance (seconds):")
        for adapter, elapsed in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {adapter}: {elapsed:.4f}s")
