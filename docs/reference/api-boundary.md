# Public and Internal APIs

The public API is the API documented in this reference section. It includes the
documented `annnet`, `annnet.io`, `annnet.adapters`, `annnet.algorithms`, and
`annnet.utils` entry points, plus documented methods on `AnnNet`.

Internal APIs are not guaranteed to be stable. This includes modules, classes,
functions, attributes, and arguments whose names begin with an underscore, and
anything imported from an underscore module such as `annnet.core._helpers` or
`annnet.core._Layers`. Their locations, signatures, behavior, and existence may
change without a deprecation period.

In other words, we do not officially support (or encourage users to do) something like

```python
from annnet.core._helpers import EdgeRecord
```

Use the documented public modules and object methods instead.
