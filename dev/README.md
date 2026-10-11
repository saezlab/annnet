# Developer notes

Notes for people who change the annnet implementation. They are not part of
the public documentation at <https://annnet.sysbioverse.org/>.

| File | Content |
| --- | --- |
| [`internals.md`](internals.md) | The in-memory architecture: canonical state, stores, and module responsibilities |
| [`migration.md`](migration.md) | The map from removed names to the current API, for code written before 0.4 |
| [`dependents.md`](dependents.md) | The packages that call annnet, and how `tests/test_dependents.py` protects them from renames |
| [`known-issues.md`](known-issues.md) | Defects found and not fixed yet, each with a script that shows it |
| [`api-contract.md`](api-contract.md) | The detailed contract of the public surface; `tests/test_public_surface.py` checks the code against it |
