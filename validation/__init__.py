# Marks validation/ as a package so the gate evaluator (compare_gate) is
# importable by the M1 driver and the test suite via the repo-root sys.path
# bootstrap (conftest.py / per-script bootstrap). Data subdirectories
# (xfoil/, nasa/) are plain folders and carry no Python.
