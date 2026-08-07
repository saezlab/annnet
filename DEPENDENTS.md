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

## When you add a package that calls AnnNet

Add an entry. `contact` says what to do when it breaks — `ours — push directly`,
or `pull request`, or a person. `verified_against` is the AnnNet commit somebody
last ran that package's tests at, with AnnNet installed beside it.

## What this does not do

**It does not prove a dependent works.** The register is written by hand, so a
bridge may call more than its entry lists. A passing gate means nobody has told
us about a break in the names we know about. It is not a test of their package.

The only thing that tests a dependent is its own suite, run with AnnNet
installed beside it:

```bash
cd ../omnipath-client && uv pip install -e ../annnet && uv run pytest
```

Without that install, every test that touches a graph skips, and the drift goes
unseen. That is how two broken converters survived for two releases.

**It does not find a package nobody has added.** Two ways to look for one:

```bash
# packages installed beside annnet that import it
grep -rlI "annnet" .venv/lib/python*/site-packages --include=*.py \
  | grep -v site-packages/annnet

# a checkout that calls it
grep -rlI --include=*.py --include=*.ipynb "annnet" ~/some-repo
```

The first found corneto. The second missed it, because the checkout on that
machine predated the bridge — **a local clone is not an inventory.**

## When the API stops moving

The first stable release is what retires this. At that point a removal gets a
deprecation period, the deprecation warning tells a dependent directly, and the
register becomes a courtesy rather than the only signal. Until then it is the
only signal.
