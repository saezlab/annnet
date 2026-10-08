# Guide data

These three small CSV files are shared by the quickstart and the Guide
notebooks. Download them together to run a notebook outside this repository.

| File | Contents |
| --- | --- |
| [interactions.csv](interactions.csv) | 11 directed interactions among 10 nodes, including two separately identified EGFR–GRB2 edges. |
| [edge_results.csv](edge_results.csv) | Illustrative edge activities for two conditions: eight selected edges per condition. |
| [node_measurements.csv](node_measurements.csv) | One value per node in each of the two conditions. |

## Provenance and meaning

This is a **synthetic teaching dataset**, assembled for the AnnNet documentation
from the signaling examples previously built inside its notebooks. The names
evoke a signaling network, but the topology is simplified and the confidence,
activity, and measurement values are invented. No biological conclusion should
be drawn from them. The files are distributed under the repository's BSD-3-Clause
license.

In `interactions.csv`, `effect` uses +1 for activation and −1 for inhibition.
`confidence` is an illustrative evidence score; `evidence` identifies a teaching
source. Neither becomes a structural edge weight in these notebooks. Explicit
`edge_id` values keep the two EGFR–GRB2 records distinct and let results refer back
to the same interactions.

A missing row in `edge_results.csv` means the interaction was not selected for
that condition. It is different from a selected interaction with activity zero.
The node measurements are a separate table: importing edges does not attach
these values automatically.

To use your own network, replace the input files and specify your source,
target, identifier, and annotation columns at import. Start with the
[quickstart](../../quickstart.md) or
[Annotations, selections, and views](../annotations-and-views.ipynb).
