# Layers

Layers provide multilayer semantics (aspects, layer tuples, and per-layer
attributes). This page covers both the manager API and the mixin used by
`AnnNet`.

## Layer Manager

::: annnet.core._Layers.LayerManager
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## Layer Mixin

::: annnet.core._Layers.LayerClass
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.core._Layers.LayerManager
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

    ::: annnet.core._Layers.LayerClass
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"
