"""The study tools: masking problems and reporting results.

These are kept apart from the conversion path (configuration, model client, oracle,
assembly, agents, check, results store, assessment). The conversion path turns one notebook
into a checked kernel and records the run. The study tools are used afterwards, to prepare
the masking experiment and to turn the stored records into tables and charts. Nothing in
the conversion path needs this subpackage, except that `parallelise run --masked` calls the
masking tool.

- `mask.py` writes a copy of a problem with the algorithm's name and hints removed.
- `report.py` builds the results tables and charts from the results store.
- `cli.py` registers the `mask` and `report` commands into the main command line.
"""
