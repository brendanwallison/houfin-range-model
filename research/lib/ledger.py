"""Every evaluation on dev data, logged by the evaluating code itself (not by the agent's memory).

The ledger counts forking paths: how many times a question has been looked at on dev data before its
answer was believed. ``budget.json:dev_evals_per_question`` caps it; past the cap, a new
pre-registration is needed. Sealed species never appear here -- they go through ``research/adopt.py``,
which logs to ``sealed_reads.csv``.
"""
import csv
import datetime as _dt
import os

from . import paths

LEDGER = paths.RESEARCH / "ledger.csv"
FIELDS = ["ts", "exp", "question", "split", "species_set", "n_species", "note"]


def log_eval(exp, question, split, species_set, n_species, note=""):
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({"ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "exp": exp, "question": question, "split": split, "species_set": species_set,
                    "n_species": int(n_species), "note": note})


def count(question):
    if not LEDGER.exists():
        return 0
    with open(LEDGER, encoding="utf-8") as fh:
        return sum(1 for r in csv.DictReader(fh) if r["question"] == question)


if __name__ == "__main__":
    import sys
    print(count(sys.argv[1]) if len(sys.argv) > 1 else os.fspath(LEDGER))
