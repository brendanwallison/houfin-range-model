"""The sealed confirmation half of the out-of-community species, decided BEFORE any data is read.

A species is sealed iff ``sha256(SALT + code)`` is odd. The rule depends on nothing but the species
code, so it cannot be tuned after a result is seen, and the same species is sealed in every run,
checkpoint and overlay. Out-of-community species never enter ESK or DESK, so sealing them is clean
by construction (a block or decade seal is not: the ESK basis already saw every block and decade).

The cache builder writes sealed species' columns to a separate file that only ``research/adopt.py``
reads; every such read is logged to ``research/sealed_reads.csv``. Changing SALT is a protocol
breach, not a refactor.
"""
import hashlib

SALT = "houfin-desk-temporal-seal-v1:"


def is_sealed(code):
    return int(hashlib.sha256((SALT + str(code).lower()).encode()).hexdigest(), 16) % 2 == 1


def split(codes):
    """``(dev, sealed)`` lists, order preserved."""
    dev = [c for c in codes if not is_sealed(c)]
    sealed = [c for c in codes if is_sealed(c)]
    return dev, sealed
