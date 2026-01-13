import json
from pathlib import Path


def render(json_path: str):
    data = json.loads(Path(json_path).read_text())
    print(f"# annnet benchmark ({data['scale']})\n")

    for name, res in data["benchmarks"].items():
        print(f"## {name}")
        if "skipped" in res:
            print(f"- SKIPPED: {res['error']}\n")
            continue
        for k, v in res.items():
            print(f"- {k}: {v}")
        print()
