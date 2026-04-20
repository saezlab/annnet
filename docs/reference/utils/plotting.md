# Plotting

Plotting helpers from `annnet.utils.plotting`.

`plot(..., backend="auto")` selects the first installed plotting backend in
preference order: Graphviz, pydot, then matplotlib. The matplotlib path is a
minimal native fallback that does not require NetworkX, Graphviz, or pydot. Use
`set_default_plot_backend` to configure the process-wide default used when
`plot(..., backend=None)`.

::: annnet.utils.plotting
    options:
      filters: public
      show_root_heading: true
