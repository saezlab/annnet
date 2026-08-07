# Public and Internal APIs

The public API is the API this reference section documents. It is the
`annnet`, `annnet.io`, `annnet.adapters`, `annnet.algorithms` and `annnet.utils`
entry points, together with the documented methods of `AnnNet`.

Each of those namespaces states an explicit `__all__`, and `annnet.core` exports
five names and no more: `AnnNet`, `Graph`, `EdgeType`, `EdgeView` and
`NodeView`. Everything else in `annnet.core` is private.

The internal API carries no stability guarantee. That covers every module,
class, function, attribute and argument whose name begins with an underscore,
and anything an underscore module holds, such as `annnet.core._store` or
`annnet.core._records`. Their location, signature, behavior and existence may
change without a deprecation period.

So the package does not support, and does not encourage, something like:

```python
from annnet.core._store import CoreState
```

Use the documented public modules and object methods instead.
