# Plotting

Draw a graph with [Graphviz](https://graphviz.org/),
[pydot](https://github.com/pydot/pydot) or
[Matplotlib](https://matplotlib.org/). `plot` builds the drawing, and a
notebook shows it. `save` writes it to a file.

```python
from annnet.utils import plot, save

drawing = plot(anet, layout='dot')
save(drawing, 'network', format='svg')  # writes network.svg
```

`plot(..., backend='auto')` uses the first installed backend in this order:
Graphviz, pydot, Matplotlib. The Matplotlib backend needs no other library. To
change the default backend, call `annnet.set_default_plot_backend`.

::: annnet.utils.plotting.plot
    options:
      show_root_heading: true

::: annnet.utils.plotting.save
    options:
      show_root_heading: true

::: annnet.utils.plotting.to_graphviz
    options:
      show_root_heading: true

::: annnet.utils.plotting.to_pydot
    options:
      show_root_heading: true

::: annnet.utils.plotting.to_matplotlib
    options:
      show_root_heading: true
