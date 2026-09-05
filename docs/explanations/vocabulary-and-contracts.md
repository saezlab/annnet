# What a number means, and whether it means it

`AnnNet.validate()` asks whether the object is *internally consistent* — the slot
bijections hold, the matrix agrees with the store. A failure there means the
graph is broken.

There is a second question, and it is the one that produces published mistakes:
do the attributes **mean** what a method reading them will assume? A failure
there means the graph is fine and the data in it will be misread. Sign written
into `weight`; `0.0` standing for "not reported"; a confidence and a coefficient
in one column. None of that is a structural fault, and every one of it produces a
wrong answer that looks right.

The two are separate entry points on purpose. Folded together, a caller could not
tell *your object is corrupt* from *your data does not say what you think*.

## Why this is not in the core

A general network data structure that names a gene has a shape that fits one
field, and reads every other field's vocabulary as foreign. So the core holds
mechanism — incidence, layers, slices, coefficients — and the meanings live in
`annnet.experimental.vocabulary`, which becomes its own package.

That split is **enforced, not intended**: `tests/test_core_biology_free.py` reads
the source of `core`, `io` and `_support` and fails on a domain word. It has
caught real violations during development, including a parameter named
`stoichiometry` that is now `coefficients` — the core's own word for the thing,
with the domain meaning living here.

## The three-way split that used to be one column

`weight` was doing three jobs. It is now one:

| name | is |
|---|---|
| `weight` | the incidence coefficient. Structural. Arithmetic over the matrix is arithmetic over this. |
| `sign` | the direction of effect, `+1` or `-1`. Absent means unknown — **never `0.0`**, which is a number and will be averaged. |
| `confidence` | how much the source believes the edge exists, in `[0, 1]`. |

After that, *which field holds the number a method should read* is answerable
without knowing which resource produced the edge.

## Checking a graph

```python
from annnet import exp

exp.vocabulary.check(G)  # a list of problems; empty is clean
exp.vocabulary.check(G, strict=True)  # raise instead
exp.vocabulary.requires(G, 'signed', 'directed')  # capabilities, strict by default
```

`check` reads meaning; `requires` reads shape. `requires` is strict by default
because a caller writing it is stating a precondition rather than asking a
question.

One rule is worth naming because it catches a mistake nobody notices: if every
`weight` in a graph is −1, 0 or +1 **and no edge carries a `sign`**, that is a
sign column wearing the structural name, and every matrix built from it is
arithmetic over the wrong quantity.

## What a method needs, as a value

A method's requirement is the same on every call and readable without running
anything, so it is a value rather than strings passed at each call site:

```python
SPEC = exp.vocabulary.MethodSpec(
    name='decoupler',
    requires_edge=('sign',),
    directed=True,
    hyperedges='none',
)
exp.vocabulary.check(G, method=SPEC, strict=True)
```

!!! warning "An over-declared spec is worse than none"

    Every field defaults to *does not care*, and that default is load-bearing. A
    spec that declares more than the method needs refuses graphs the method would
    have handled — and what a user learns from that is to skip the check.

    The concrete case: a regulatory network contains regulator-to-regulator
    edges. A spec declaring `bipartite` would refuse the very resource the method
    exists to score.

## Identifiers

A network names entities the way its source did; a measurement table names them
the way *its* source did. Three shapes of disagreement matter and they are
different problems — one thing with two spellings, one node made of several
measured things, and one measured thing reaching several nodes.

```python
exp.vocabulary.parse('uniprot:P15056')  # Identifier(namespace='uniprot', ...)
exp.vocabulary.SymbolMapper({'BRAF': 'uniprot:P15056'})
exp.vocabulary.SymbolMapper(separator='_', known=measured)
```

The `Mapper` protocol is what keeps a resource's spelling rules out of the join —
see [Joining measurements](joining-measurements.md).

## Where to go next

- [Joining measurements to a network](joining-measurements.md) — the loop this
  vocabulary serves.
