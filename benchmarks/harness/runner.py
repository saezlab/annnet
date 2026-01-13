import importlib
import json
from pathlib import Path

from .scales import SCALES

BENCH_ROOT = Path(__file__).resolve().parents[1]


def discover_benchmarks():
    benches = []
    for pkg in ("core", "io", "adapters"):
        pkg_dir = BENCH_ROOT / pkg
        for py in pkg_dir.glob("*.py"):
            if py.name.startswith("_"):
                continue
            benches.append(f"benchmarks.{pkg}.{py.stem}")
    return benches


def run(scale_name="small"):
    scale = SCALES[scale_name]
    results = {"scale": scale_name, "benchmarks": {}}

    for modname in discover_benchmarks():
        mod = importlib.import_module(modname)
        if hasattr(mod, "run"):
            try:
                results["benchmarks"][modname] = mod.run(scale)
            except Exception as e:
                results["benchmarks"][modname] = {
                    "error": str(e),
                    "skipped": True,
                }
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="small", choices=SCALES.keys())
    p.add_argument("--out", default="benchmark_results.json")
    args = p.parse_args()

    res = run(args.scale)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"Wrote {args.out}")
