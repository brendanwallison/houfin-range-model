"""Data pipeline for houfin-range-model, organized by stage:

    identify/    decide *what* to pull (e.g. the eBird reference community)
    acquire/     programmatic downloads of raw external products
    preprocess/  regrid/align each product to the model grid (resolution-deferral
                 aware; see src/processing/regrid.py)
    combine/     merge products into model-ready inputs

Modules hold the logic and expose a ``main()``; ``scripts/`` carries thin
launchers so both ``python scripts/<x>.py`` and ``python -m src.data...`` work.
"""
