"""Run unchanged fair_rivals metrics with a content-addressed cache of expensive BLR fits.

Inputs, block structure, optimizer settings and implementation identities enter the key.
This only avoids refitting identical modern readouts for different scored epochs.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import blr


def install_blr_cache(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = blr.fit
    implementation = hashlib.sha256(Path(blr.__file__).read_bytes() + Path(__file__).read_bytes()).hexdigest()

    def cached(X, Y, blocks, max_iter=300, starts=blr.STARTS):
        settings = {"blocks": blocks, "max_iter": os.environ.get("BLR_MAX_ITER", max_iter),
                    "starts": os.environ.get("BLR_STARTS", starts), "implementation": implementation}
        h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
        for value in (X, Y):
            value = np.ascontiguousarray(value, dtype="float64")
            h.update(str(value.shape).encode())
            h.update(memoryview(value).cast("B"))
        path = directory / (h.hexdigest() + ".npz")
        if path.exists():
            with np.load(path) as f:
                model = {k: f[k] for k in f.files}
            model["n"] = int(model["n"])
            model["blocks"] = [tuple(row) for row in model["blocks"].tolist()]
            print(f"[fit-cache] hit {path.stem[:12]}", flush=True)
            return model
        model = original(X, Y, blocks, max_iter=max_iter, starts=starts)
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temp.open("wb") as fh:
            np.savez_compressed(fh, **model)
        temp.replace(path)
        print(f"[fit-cache] fitted {path.stem[:12]}", flush=True)
        return model
    blr.fit = cached


if __name__ == "__main__":
    root = os.environ["FINAL_FIT_CACHE"]
    install_blr_cache(root)
    from fair_rivals import main
    main()
