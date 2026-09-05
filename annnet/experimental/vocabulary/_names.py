"""The names a graph's numbers go under, and what each one means.

The core holds mechanism: incidence, layers, slices, coefficients. It may not
say what a number *means*, because a general network data structure that names a
gene has a shape that fits one field and reads every other field's vocabulary as
foreign. So the meanings live here.

Reserved rather than suggested
------------------------------

A name is reserved when a method will read it without asking. ``sign`` is the
clearest case: a method that scores activation reads ``sign`` and assumes ±1. If
one resource wrote a continuous confidence there, the method does not fail — it
produces a plausible number, which is worse.

So each reserved name has a **declared domain**, and
:func:`annnet.experimental.vocabulary.check` reads a graph and says where the
data does not honour it.

The three-way split that used to be one column
----------------------------------------------

``weight`` was doing three jobs. It is now one:

- **weight** is structural — the incidence coefficient, and on a directed
  hyperedge the per-member coefficient. Arithmetic over the matrix is arithmetic
  over this. It is never a sign and never a confidence.
- **sign** is the direction of effect, ``+1`` or ``-1``. Absent means unknown.
  Never ``0.0``, which is a number and would be averaged.
- **confidence** is how much the source believes the edge exists, in ``[0, 1]``.

After this, *which field holds the number a method should read* is answerable
without knowing which resource produced the edge.
"""

from __future__ import annotations

#: The incidence coefficient, and on a directed hyperedge the per-member
#: coefficient. Structural. Never a sign, never a confidence.
WEIGHT = 'weight'

#: Direction of effect: ``+1`` activation, ``-1`` inhibition. Absent means
#: unknown — never ``0.0``.
SIGN = 'sign'

#: How much the source believes the edge exists, in ``[0, 1]``.
CONFIDENCE = 'confidence'

#: What the core calls ``coefficients`` when the domain gives them a meaning.
#: The core may not say this word; here it may.
STOICHIOMETRY = 'stoichiometry'

#: The symbol an entity is known by in the resource it came from.
GENE_SYMBOL = 'gene_symbol'

#: The domain of :data:`SIGN`. Two values, and zero is not one of them.
SIGN_DOMAIN = (-1, 1)

#: The range of :data:`CONFIDENCE`.
CONFIDENCE_RANGE = (0.0, 1.0)

#: Node-axis names this vocabulary reserves.
NODE_RESERVED = ('kind', 'namespace', 'local_id', GENE_SYMBOL)

#: Edge-axis names this vocabulary reserves.
EDGE_RESERVED = (WEIGHT, SIGN, CONFIDENCE, STOICHIOMETRY)

#: What an entity may be. Open rather than closed — a kind this does not name is
#: not an error, it is a kind nothing has declared a policy for yet.
KINDS = (
    'molecule',
    'complex',
    'gene',
    'transcript',
    'protein',
    'metabolite',
    'cell',
    'tissue',
    'condition',
    'reaction',
)


class ContractViolation(AssertionError):
    """Raised by :func:`check` in strict mode.

    An ``AssertionError`` because it means an assumption a method was about to
    make does not hold — not that the object is malformed, which is what
    :meth:`AnnNet.validate` is for.
    """
