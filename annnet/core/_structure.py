"""The read-only structural query facade.

This module is the one boundary between the canonical store and the rest of the
package. Input-output code, adapters, and bridges read topology through the
functions here. They never read a private store attribute of the graph.

The facade answers questions about structure only. It reports which entities and
edges exist, which entities an edge holds, and which edges touch an entity. It
does not report attributes, and it never writes.

The facade hides which store backs the graph. The current store keeps entity and
edge records. A later store keeps slot-addressed member lists. The signatures
here stay the same across that change, so the callers stay the same too.
"""

from __future__ import annotations
