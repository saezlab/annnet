# Attributes

`anet.attrs` holds all attributes of a graph, at eight addresses: `nodes`,
`edges`, `slices`, `aspects`, `layers`, `edge_slices`, `node_layers` and
`elementary_layers`. You read and write each address the same way:

- a read (`nodes`, `table`, `row`, `rows`) gives a copy;
- a write is a named call (`update`, `replace`, `delete`);
- `select` is a live query, and `from_frame` makes a fixed selection from the
  rows of a table that you filtered yourself.

`anet.attrs` is an `Attrs` object. `select` and `from_frame` give a
`RowSelection`, and `schema` gives a `Schema`. The
[annotations guide](../../guide/annotations-and-views.ipynb) shows the use.

## Addresses

An address is the thing that a set of attributes belongs to. Each address has
its key:

| address | key | key columns in a table |
|---|---|---|
| `nodes` | `node_id` | `node_id` |
| `edges` | `edge_id` | `edge_id` |
| `slices` | `slice_id` | `slice_id` |
| `aspects` | `aspect` | `aspect` |
| `layers` | layer coordinate tuple | `layer` |
| `edge_slices` | `(slice_id, edge_id)` | `slice_id`, `edge_id` |
| `node_layers` | `(node_id, layer)` | `node_id`, `layer` |
| `elementary_layers` | `(aspect, label)` | `aspect`, `elementary_layer` |

```python
# Read: each call gives a copy. A change to the copy does not change the graph.
anet.attrs.nodes  # the table of one address; one property for each address
anet.attrs.table('nodes', derived=False, backend=None, columns=None, limit=None, layout=None)
anet.attrs.row('nodes', key)  # one row, as a dict
anet.attrs.rows('nodes', keys=None)  # {key: dict}
anet.attrs.schema('nodes')  # the keys and fields, without a table

# Write
anet.attrs.update('nodes', {key: {...}, ...})  # merge the given fields
anet.attrs.replace('nodes', table)  # replace all attributes of the address
anet.attrs.delete('nodes', keys=None, names=None)  # remove attributes

# Select
anet.attrs.select('nodes', score__gte=0.5)  # a live selection
anet.attrs.from_frame('nodes', frame)  # a fixed selection
```

`anet.attrs` has no item access, and you cannot assign to its table
properties. `anet.attrs['nodes']` and `anet.attrs.nodes = table` raise
`TypeError`, with the name of the call to use.

## Rules for writes

- **A null value removes the field.** If `update` gets `None` or NaN for a
  field, annnet removes the field from the row. A null cell in a table means
  the same.
- **`delete` removes attributes, never elements.** The node, edge, slice or
  layer stays. The return value is the number of rows that lost an attribute.

    | `keys` | `names` | result |
    |---|---|---|
    | given | not given | remove all attributes of these rows |
    | not given | given | remove these fields from all rows |
    | given | given | remove these fields from these rows |
    | not given | not given | `ValueError` |

    An empty `keys` or `names` selects nothing. A name that the address does
    not have raises `KeyError`.
- **`replace` needs the key columns of the address.** A key that the table
  does not have loses its attributes. A table never adds or removes elements.
- **A write is all or nothing.** annnet checks the full write before it
  changes anything. If one part fails, the attributes, the fields and their
  types, and the cached matrices stay as they were.
- **Reserved fields** are refused with `ValueError`: the key columns, and for
  edges `source`, `target`, `head`, `tail`, `members`, `weight`, `directed`,
  `kind`, `slice`, `slice_weight`, `flexible` and `edge_type`.
- **Keys are checked.** An unknown key raises `KeyError`, a key of the wrong
  shape `TypeError`, and a key that occurs two times in one write
  `ValueError`. For `node_layers`, give `(node_id, layer)`, or a node ID with
  `layer=` in `update`.

## Derived tables

`table(address, derived=True)` adds what the graph knows about each row. For
`edges`, these columns: `kind` (`binary`, `hyper` or `node_edge`), `directed`,
`weight`, `ml_kind` (`intra`, `inter` or `coupling`), `source`, `target`,
`src_layer`, `dst_layer`, `head`, `tail` and `members`.

With `layout='incidences'`, the table has one row for each endpoint of each
edge: `edge_id`, `position`, `entity_id`, `entity_kind`, `layer`, `layer_id`,
`role`, `coefficient`, `kind` and `directed`. You can join an endpoint to its
node, layer and edge rows without parsing.

`limit=` builds that number of rows and no more: use it for a preview, not as
a selection. The arguments `slice=`, `in_slice=`, `layer=`, `include_hyper=`,
`include_binary=`, `include_directed=`, `include_weight=` and
`resolved_weight=` apply to the derived edge table only.

## A live query and a filtered table

`select` is a query on the graph. It keeps its conditions, and gives a new
result after the graph changes. A table from `anet.attrs.table` is a copy of
one moment. To filter it, use your dataframe library, then make a selection
from the result:

```python
table = anet.attrs.table('nodes')
kept = table.filter(pl.col('score') > 0.5)
chosen = anet.attrs.from_frame('nodes', kept)  # a fixed selection
```

`from_frame` reads only the key columns. It gives the keys of the frame, in the
order of the address, without repeats. It does not keep the filter: a node
that you add later is not in it. A missing key column or a null key raises
`ValueError`, an unknown key `KeyError`.

A selection of `node_layers` or `edge_slices` changes into a selection of
another address only with `project`: `select('node_layers', ...).project('nodes')`,
`select('edge_slices', ...).project('edges')` or `.project('slices')`.

::: annnet.core._attribute_api.Attrs
    options:
      filters: public
      show_root_heading: true

::: annnet.core._attribute_api.Schema
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.RowSelection
    options:
      filters: public
      show_root_heading: true

## Predicates

All `select` calls (`anet.N.select`, `anet.E.select`, `anet.attrs.select`,
`anet.layers.where`) take the same conditions: `field=value`, or
`field__operator=value`. The operators are `eq`, `ne`, `in`, `not_in`, `lt`,
`lte`, `gt`, `gte` and `isnull`.

- A null value matches only `isnull=True`.
- An ordered aspect compares labels in their declared order. An ordering
  operator on a categorical aspect raises `ValueError`.
- An unknown field raises `KeyError`. Values that cannot be compared raise
  `TypeError`.
- For a field name that contains `__`, give a mapping with
  `(field, operator)` keys.

::: annnet.core._predicate
    options:
      members:
        - OPERATORS
        - parse_conditions
      show_root_heading: true
