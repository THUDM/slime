"""Marker file making `examples.coding_agent_rl` an explicit Python package.

Without this file the directory becomes an implicit namespace package, which
breaks relative imports (`from .aiohttp_threaded import ...`) when ray
workers / pytest run from an external cwd. See SPEC §10.3 U5 + B2 regression
test `tests/test_external_cwd_import.py` for the failure mode this prevents.

Implementation detail (not part of the SPEC-locked U5 decision): U5 only
requires aiohttp_threaded to live next to middleware.py rather than under
slime/utils/. Marking the directory as an explicit package is purely about
import hygiene and does not change the U5 file layout.
"""
