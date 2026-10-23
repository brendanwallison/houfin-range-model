"""Move COLD research caches from the WSL disk to E: and leave a symlink, so every path keeps working.

    python research/bin/archive_caches.py            # dry run: list cold caches and sizes
    python research/bin/archive_caches.py --apply    # move them

A cache is HOT (kept) when any running/queued pueue job's spec names it ({cache}/<name>, including the <name>$st
loops, which also name <name>_late), or it is a seed control desk_tempho_{1995,1985}_s{1-4}[_late]. Every other
cache directory is cold.
Move = copy to <dest>/<name>, then sha256 of EVERY file on both sides. Only on a full match is the source removed and
replaced by a symlink to the copy. Any mismatch leaves the source untouched and stops. Symlinked caches are skipped.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.expanduser("~/houfin/work/houfin/research_cache")
DEST = "/mnt/e/Datasets/houfin/research_cache_archive"


def hot_caches():
    d = json.loads(subprocess.run(["pueue", "status", "--json"], capture_output=True, text=True, check=True).stdout)
    ref = set()
    for t in d["tasks"].values():
        st = t["status"]
        s = next(iter(st)) if isinstance(st, dict) else st
        if s not in ("Running", "Queued", "Stashed", "Paused"):
            continue
        spec_p = os.path.join(t["command"].split()[-1], "spec.json")
        if not os.path.exists(spec_p):
            continue
        txt = " ".join(json.load(open(spec_p))["cmd"])
        for name, loop in re.findall(r"\{cache\}/([A-Za-z0-9_]+)(\$st)?", txt):
            ref.add(name)
            if loop:
                ref.add(name + "_late")
    ref |= {f"desk_tempho_{t}_s{k}{s}" for t in ("1995", "1985") for k in (1, 2, 3, 4) for s in ("", "_late")}
    return ref


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tree_hashes(root):
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = sha(p)
    return out


def du(p):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    hot = hot_caches()
    names = sorted(x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x))
                   and not os.path.islink(os.path.join(ROOT, x)))
    cold = [n for n in names if n not in hot]
    sizes = {n: du(os.path.join(ROOT, n)) for n in cold}
    print(f"hot (kept): {sorted(set(names) & hot)}")
    print(f"cold: {len(cold)} caches, {sum(sizes.values()) / 2**30:.1f} GB -> {DEST}")
    if not a.apply:
        return
    os.makedirs(DEST, exist_ok=True)
    for n in cold:
        src, dst = os.path.join(ROOT, n), os.path.join(DEST, n)
        if os.path.exists(dst):
            sys.exit(f"{dst} already exists; resolve by hand")
        shutil.copytree(src, dst + ".partial")
        hs, hd = tree_hashes(src), tree_hashes(dst + ".partial")
        if hs != hd:
            sys.exit(f"CHECKSUM MISMATCH for {n}; source untouched, copy left at {dst}.partial")
        os.rename(dst + ".partial", dst)
        with open(os.path.join(dst, "ARCHIVED_FROM.json"), "w") as fh:
            json.dump({"source": src, "files": len(hs), "sha256": hs}, fh, indent=1)
        shutil.rmtree(src)
        os.symlink(dst, src)
        print(f"  moved {n} ({sizes[n] / 2**30:.2f} GB, {len(hs)} files verified)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
