# SBML

Read and write [SBML](https://sbml.org/) models. A reaction becomes a directed
hyperedge: the reactants are on one side, the products on the other, and the
stoichiometric coefficients are kept on the edge. A boundary reaction becomes a
one-sided edge with the `is_boundary` edge attribute.

`from_sbml` reads the file with
[python-libsbml](https://sbml.org/software/libsbml/). `from_sbml_cobra` and
`from_cobra_model` read through [COBRApy](https://opencobra.github.io/cobrapy/),
from a file or from a COBRApy model object.

::: annnet.io.sbml.from_sbml
    options:
      show_root_heading: true

::: annnet.io.sbml.to_sbml
    options:
      show_root_heading: true

::: annnet.io.sbml_cobra.from_sbml
    options:
      show_root_heading: true
      heading: from_sbml_cobra
      toc_label: from_sbml_cobra

::: annnet.io.sbml_cobra.from_cobra_model
    options:
      show_root_heading: true
