# Packages that call AnnNet

AnnNet is before its first stable release. `CHANGELOG.md` says what that means
for a user: a removed name carries no deprecation and no alias, and each removal
names its replacement.

It means something else for a package that bridges to us. A rename here does not
fail our build, does not fail our tests, and does not warn anybody. It fails in
their repository, at a time nobody chose, and usually the person who finds it is
a user of theirs.

This file, `dependents.toml` and `tests/test_dependents.py` are the answer to
that, for as long as the API moves without a deprecation process.

## How it works

`dependents.toml` lists one entry per package that calls AnnNet: who owns it,
where it lives, which of its modules bridge to us, and **the AnnNet names it
calls**.

`tests/test_dependents.py` asserts that each of those names still resolves on
the public surface. So a rename fails the build **here**, in front of the person
making it, with a message that says which package to update and how to reach its
owner:

    corneto calls ['layers.layer_vertex_set'], which the public surface no
    longer carries. It bridges through corneto/contrib/annnet.py,
    corneto/methods/signaling/annnet.py. Owner: Pablo Rodríguez-Mier
    (pull request) at https://github.com/saezlab/corneto.

The register also pins the structural key names — `node_id`, `source`, `target`,
`weight` — because a bridge writes those into a spec, and renaming one moves a
caller's value into an ordinary attribute **without raising anything**.

## When you remove or rename a public name

1. Name the replacement in `CHANGELOG.md`, as every removal there already does.
2. Update `dependents.toml` to the new spelling, in the same change.
3. Open a pull request against every repository the register names for that
   name, or push directly where the entry says the package is ours.

Step 3 is the point of the file. Steps 1 and 2 only make it possible to do.

## Where each migration stands

The attribute, selection and view changes in 0.4.0 removed and renamed public
names (`CHANGELOG.md`; `docs/explanations/api-migration.md`). For every
registered package, `dependents.toml` carries `previous_calls`, the `calls`
they migrate to and one `migration` line per changed name, and says how far the
migration has got. There are three states, and they are different claims:

| status | what it means |
|---|---|
| `required_before_release` | no migration is known to pass the package's tests |
| `verified_locally` | a migration applied to the recorded `upstream_revision` passed the package's own tests against the recorded AnnNet version. The package's repository has **not** merged it |
| `merged_upstream` | the package's default branch carries the migration |

`verified_locally` is real evidence and not a deployment. A user who upgrades
AnnNet and keeps the released version of a package that is only
`verified_locally` meets exactly the break this file exists to prevent. So the
register has two gates, and they answer two questions:

```bash
ANNNET_RELEASE_GATE=1      pytest tests/test_dependents.py   # every package is at least verified_locally,
                                                             # against the version in this tree
ANNNET_RELEASE_GATE=public pytest tests/test_dependents.py   # every package is merged_upstream
```

Neither runs in the ordinary CI, so a green CI does not establish that the
dependents are ready. The first gate says the migrations exist and work; the
second says users of both packages can upgrade. A change of the AnnNet version
makes the first gate fail until the packages' tests are run against it again.

Current state, for the AnnNet version in this tree:

- **omnipath-client** is `verified_locally`. Its register entry used to list one
  call; the code also uses `ncount()`, `ecount()`, `get_edge` and
  `attrs.get_attr_node` (in its tests), and the entry now lists all of them.
  Its `annnet` extra should require `annnet>=0.4.0`.
- **corneto** is `verified_locally`. Its `annnet` extra requires
  `annnet[polars]>=0.2.0,<0.3`, so `pip install corneto[annnet]` cannot install
  this version; the bound has to move to `>=0.4.0,<0.5`. The tutorial notebook
  under `docs/tutorials/annnet-signaling` pins an older AnnNet revision of its
  own and keeps the calls of that revision until a release exists to pin.

The source-level change for each package is the `migration` list in
`dependents.toml`. The digest in `patch_sha256` identifies the patch that was
tested, so a maintainer can check that the change they are handed is the one
that passed.

## When you add a package that calls AnnNet

Add an entry. `contact` says what to do when it breaks: `ours — push directly`,
or `pull request`, or a person. Start it `required_before_release`. When you have
applied its migration to a recorded commit of the package and run its suite with
this AnnNet installed beside it, record `upstream_revision`, `verified_annnet`,
`verification` and `patch_sha256`, and set `verified_locally`. Set
`merged_upstream`, with `merged_revision`, when its default branch carries the
change.

## What this does not do

**It does not prove a dependent works.** The register is written by hand, so a
bridge may call more than its entry lists. A passing gate means nobody has told
us about a break in the names we know about. It is not a test of their package.

The only thing that tests a dependent is its own suite, run with AnnNet
installed beside it, from a checkout of the commit the register names:

```bash
git clone https://github.com/saezlab/omnipath-client && cd omnipath-client
git checkout <upstream_revision>
git apply <the migration patch>          # its digest is patch_sha256
uv pip install -e /path/to/annnet && uv run pytest
```

Without that install, every test that touches a graph skips, and the drift goes
unseen. That is how two broken converters once survived two releases.

**It does not find a package nobody has added.** Two ways to look for one:

```bash
# packages installed beside annnet that import it
grep -rlI "annnet" .venv/lib/python*/site-packages --include=*.py \
  | grep -v site-packages/annnet

# a checkout that calls it
grep -rlI --include=*.py --include=*.ipynb "annnet" ~/some-repo
```

A local clone is not an inventory: a checkout that predates a package's bridge
will not show it.

## When the API stops moving

The first stable release is what retires this. At that point a removal gets a
deprecation period, the deprecation warning tells a dependent directly, and the
register becomes a courtesy rather than the only signal. Until then it is the
only signal.
