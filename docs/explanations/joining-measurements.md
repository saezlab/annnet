# Joining measurements to a network

The loop this exists for:

```
AnnData  ──attach──▶  AnnNet  ──method──▶  AnnNet (annotated)  ──write_back──▶  AnnData
```

Four things about it are worth saying out loud, because none is visible from
inside any single call.

## The measurements stay in the AnnData

`attach` does **not** copy the matrix. It holds two index maps onto the array the
AnnData already owns:

```python
from annnet import exp

report = exp.sysbio.attach(
    G,
    adata,
    aspect='condition',
    on='gene_symbol',
    values={'expression': None},
)
```

Twenty thousand entities across a thousand conditions — 2 × 10⁷ cells — attaches
in milliseconds and holds no measurable memory. Reading it back is one pass in C.

That works because **reading never consults presence**: a cell answers whether or
not its node-layer exists. So `place='none'` is a real option, and the default
`place='matched'` creates node-layers only for the pairs the join actually
reached — the size of the network, not the size of the assay. What needs a
node-layer is a *result* written back onto the graph later.

## Attach is a join, with a stated policy

The entities a network names and the entities an assay measures do not
correspond one to one. Multiplicity runs in **two directions**, and they are
different questions, so they are named apart by type:

```python
multiplicity = 'allow'  # str: one measured thing reaching several nodes
multiplicity = {
    'complex': 'min',  # dict keyed on node kind:
    'metabolite': 'never',  #   several measured things making up one node
    'default': 'error',
}
```

A **string** answers the first: every node takes the same value, so nothing is
combined — `allow`, `error`, `first`.

A **dict** answers the second: a node made of several measured things needs one
number out of many, and there is no default worth guessing. `min` says a thing is
as scarce as its scarcest part; the rest of `REDUCERS` are there too. `never`
says this kind takes no measurement at all. `error` — the default for a kind
nobody named — stops the join rather than quietly taking one part's value.

`partial=` covers the case the reducer alone does not say: a composite only
*some* of whose parts were measured. `reduce` (the default) combines what there
is, which is a weaker claim than the policy names and is why it is stated.

### What did not match comes back as data

```python
report.coverage  # the share of measured entities that reached a node
report.mapping  # one row per (entity, node), status ∈ ok | ambiguous | unmapped
report.unmapped  # the entities that reached nothing — a frame
report.unmapped_ids  # the same, as a list
report.unmeasured  # the nodes nothing reached, scoped by `scope=`
```

Silent 10% dropout is how a package loses trust once and permanently. Both a
frame and a list are offered, because a frame answers *which, and why* while
`report.unmapped_ids == ['Q']` is the comparison people actually write.

## Names are the resource's business, not the join's

`"A_B_C"` is how one particular resource writes a composite. That is a fact about
that resource, and putting it inside the join means the next resource needs
another branch in the same function. So a `Mapper` is the seam:

```python
mapper = exp.vocabulary.SymbolMapper(separator='_', known=measured)
exp.sysbio.attach(G, adata, aspect='condition', on='symbol', mapper=mapper)
```

`known=` is load-bearing: a composite is only split when every part is a name the
assay actually measured, so an ordinary name that happens to contain the
separator is not mistaken for one. That is the failure mode splitting invites.

## The write-back is a projection

```python
exp.sysbio.write_back(G, adata, key='activity', aspect='condition')  # layers
exp.sysbio.write_back(G, adata, key='score', on='symbol', into='var')  # var
exp.sysbio.write_back(G, adata, key='n_active', aspect='condition', into='obs')  # obs
```

A result has three shapes and an AnnData has three places for them: per entity
per condition is a matrix, per entity is a column, and per *condition* belongs to
the network rather than to any entity — that last one is `obs`, and it is what a
single-cell workflow reads next.

**It is lossy, and that is the contract.** Hyperedges, per-member coefficients
and topology do not fit into `obs`/`var`/`layers` and never will. What comes back
is the part of the answer that is a number per entity; the graph keeps the rest.
Anyone expecting symmetry should learn it here rather than discover it later.

## Is this graph connected?

```python
exp.sysbio.connected(G)  # does it hold attached measurements
exp.sysbio.measurements(G)  # which attribute names they answer for
```

Two lines, and they turn *a connected AnnNet* from a phrase in a docstring into
something a downstream method can assert on.

## Where to go next

- [What a number means](vocabulary-and-contracts.md) — the reserved names, and
  checking a graph before a method reads it.
- [Node-layer values and scale](values-and-scale.md) — why attaching is cheap.

## Running a method on it

Two tiers, deliberately. The **adapter** is thin, for someone building a pipeline
who wants each step visible; the **use case** is the composed loop, for someone
reading this page. The second is written in terms of the first — there is no
third code path.

```python
activity = exp.sysbio.methods.decoupler.run(G, pdata, aspect='condition', slice='regulon')
fit = exp.sysbio.methods.corneto.run(
    G,
    inputs='perturbation',
    outputs=activity.key,
    aspect='condition',
    slice='signalling',
)

# or, the whole thing at once
exp.sysbio.usecases.tf_activity(G, pdata, aspect='condition', on='symbol')
```

`outputs=activity.key` is the join between two steps: it names the attribute the
first one wrote. No intermediate dictionary, and no way for the two to disagree
about which condition is which.

**No adapter reimplements any arithmetic.** The scores are decoupler's and the
fits are CORNETO's, each pinned to floating point against calling the package
directly on the equivalent DataFrame. What an adapter contributes is the three
things around it:

- **The declaration before.** `SPEC` is a `MethodSpec`, checked before anything
  runs.
- **Reading the input off the graph.** `decoupler.regulon(G, slice=...)` and
  `corneto.pkn(G, slice=...)` produce what each package reads.
- **The additive write-back after.** A new slice plus new attributes; nothing is
  replaced, so a prior and two fits coexist on one object and any pair can be
  diffed.

!!! note "Results need somewhere to sit"

    `attach` places node-layers for the genes it *matched*. A regulator whose own
    gene the assay never measured has none — so `decoupler.run` calls
    `layers.place` for the regulators it scored, and reports how many in
    `result.placed`. Identity and values are separate questions, and this is
    where that shows.
