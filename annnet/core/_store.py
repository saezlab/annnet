"""The slot-addressed canonical store.

The store holds the canonical state of a graph. An element is addressed by a
stable identity, which is a string id, or by a stable slot, which is an integer
that the store assigns on insert and frees on delete. The store never renumbers
a slot, so a delete touches only the deleted element.

Topology lives in the member lists. The member lists are an incidence matrix in
compressed sparse column form, addressed by slot. One member list holds every
edge kind.

Only the mutation gateway writes the store. The derive layer and the query
facade read it. One structural clock rises on every write, and every derived
structure rebuilds when its recorded clock value differs.
"""

from __future__ import annotations
