# AnnNet Core + Attributes

The `AnnNet` class is the primary annotated network object. It composes
multiple mixins that provide topology management, attribute tables, bulk
operations, history, and cache control.

## AnnNet

::: annnet.core.graph.AnnNet
    options:
      members: true
      inherited_members: false
      show_root_heading: true
      show_bases: false
      filters:
      - "!^_"

## Edge Types

::: annnet.core.graph.EdgeType
    options:
      show_root_heading: true

## Topology & Indexing

::: annnet.core._Index.IndexManager
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

::: annnet.core._Index.IndexMapping
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## Attribute Tables

::: annnet.core._Annotation.AttributesClass
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## Bulk Operations

::: annnet.core._BulkOps.BulkOps
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## History

::: annnet.core._History.History
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

::: annnet.core._History.GraphDiff
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## Cache & Operations

::: annnet.core._Cache.CacheManager
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

::: annnet.core._Cache.Operations
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

## Views

::: annnet.core._Views.GraphView
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

::: annnet.core._Views.ViewsClass
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"
