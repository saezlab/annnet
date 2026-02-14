# Slices

Slices represent subgraphs with their own membership and attributes. This
page covers both the manager API and the mixin used by `AnnNet`.

## Slice Manager

::: annnet.core._Slices.SliceManager
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## Slice Mixin

::: annnet.core._Slices.SliceClass
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.core._Slices.SliceManager
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

    ::: annnet.core._Slices.SliceClass
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"
