"""What a number in a graph means — the half the core may not hold.

Becomes its own package. Everything here is vocabulary rather than mechanism:
reserved names with declared domains, the identifiers two names may share, the
kinds an entity may be, and the checks over all of it.
"""

from __future__ import annotations

from ._names import (
    SIGN,
    KINDS,
    WEIGHT,
    CONFIDENCE,
    GENE_SYMBOL,
    SIGN_DOMAIN,
    EDGE_RESERVED,
    NODE_RESERVED,
    STOICHIOMETRY,
    CONFIDENCE_RANGE,
    ContractViolation,
)
from ._method import HYPEREDGE_POLICIES, MethodSpec
from ._contract import (
    CAPABILITIES,
    DEFAULT_CONTRACT,
    check,
    rules,
    requires,
    contracts,
    capabilities,
    method_problems,
)
from ._identifiers import (
    Mapper,
    Identifier,
    NullMapper,
    SymbolMapper,
    parse,
    render,
    names_of,
    as_mapper,
)

__all__ = [
    'CAPABILITIES',
    'CONFIDENCE',
    'CONFIDENCE_RANGE',
    'DEFAULT_CONTRACT',
    'EDGE_RESERVED',
    'GENE_SYMBOL',
    'HYPEREDGE_POLICIES',
    'KINDS',
    'NODE_RESERVED',
    'SIGN',
    'SIGN_DOMAIN',
    'STOICHIOMETRY',
    'WEIGHT',
    'ContractViolation',
    'Identifier',
    'Mapper',
    'MethodSpec',
    'NullMapper',
    'SymbolMapper',
    'as_mapper',
    'capabilities',
    'check',
    'contracts',
    'method_problems',
    'names_of',
    'parse',
    'render',
    'requires',
    'rules',
]
