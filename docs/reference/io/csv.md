# CSV

Read a graph from a CSV file or a dataframe, and write its edges to CSV.
`from_csv` and `from_dataframe` detect the layout of the table: an edge list, a
hyperedge table, an incidence matrix, an adjacency matrix or neighbor lists.
If the detection is wrong, give the layout with `schema=`. A column that does
not describe the structure becomes an attribute.

::: annnet.io.csv_format.from_csv
    options:
      show_root_heading: true

::: annnet.io.csv_format.from_dataframe
    options:
      show_root_heading: true

::: annnet.io.csv_format.to_csv
    options:
      show_root_heading: true

::: annnet.io.csv_format.edges_to_csv
    options:
      show_root_heading: true

::: annnet.io.csv_format.hyperedges_to_csv
    options:
      show_root_heading: true
