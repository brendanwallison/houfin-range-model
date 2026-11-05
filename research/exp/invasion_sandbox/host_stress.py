"""Host determinism stress: every worker repeats fixed computations and must reproduce its first result bit for bit.

Targets the failure seen on 2026-10-30 / 11-03 (intermittent native faults; risk of silent corruption): float64 BLAS
(FMA load), FFT convolution, scipy.ndimage.label (one crash site), and a large memory pattern write / verify.
Usage: python host_stress.py MINUTES WORKERS GB_PER_WORKER        (single-threaded workers; run with OMP_NUM_THREADS=1)
Prints one JSON line per worker; exit code 1 on any mismatch or worker death.
"""
import hashlib, json, multiprocessing as mp, os, sys, time
import numpy as np


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def worker(i, minutes, gb, q):
    try:
        from scipy.signal import fftconvolve
        from scipy.ndimage import label
        rng = np.random.default_rng(1000 + i)
        A = rng.standard_normal((700, 700)); B = rng.standard_normal((700, 700))
        F = rng.standard_normal((512, 512)); Kk = rng.standard_normal((129, 129))
        M = rng.random((600, 900)) > .45
        n = int(gb * 2 ** 30 // 8); big = np.empty(n, np.uint64)
        idx = np.arange(n, dtype=np.uint64)
        ref = None; it = 0; bad = []; t0 = time.time()
        while time.time() - t0 < minutes * 60:
            r = [h(A @ B), h(fftconvolve(F, Kk, mode='same'))]
            lab, nl = label(M, structure=np.ones((3, 3), int)); r.append(h(lab) + str(nl))
            seed = np.uint64(0x9E3779B97F4A7C15) * np.uint64(it % 7 + 1)
            np.multiply(idx, seed, out=big); big ^= np.uint64(0xA5A5A5A5A5A5A5A5)
            chk = (big ^ np.uint64(0xA5A5A5A5A5A5A5A5)) == idx * seed
            if not chk.all():
                bad.append(dict(it=it, kind='memory', n_bad=int((~chk).sum())))
            if ref is None:
                ref = r
            elif r != ref:
                bad.append(dict(it=it, kind='compute', which=[j for j in range(len(r)) if r[j] != ref[j]]))
            it += 1
        q.put(dict(worker=i, iterations=it, mismatches=bad, seconds=round(time.time() - t0, 1)))
    except Exception as e:  # noqa: BLE001
        q.put(dict(worker=i, error=repr(e)))


def main():
    minutes, workers, gb = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    q = mp.Queue(); ps = [mp.Process(target=worker, args=(i, minutes, gb, q)) for i in range(workers)]
    for p in ps: p.start()
    res = [q.get(timeout=minutes * 60 + 600) for _ in ps]
    for p in ps: p.join(timeout=60)
    dead = [p.pid for p in ps if p.exitcode not in (0, None)]
    ok = not dead and all('error' not in r and not r['mismatches'] for r in res)
    for r in sorted(res, key=lambda r: r['worker']): print(json.dumps(r))
    print(json.dumps(dict(passed=ok, workers=workers, minutes=minutes, gb_per_worker=gb, dead_pids=dead,
                          exitcodes=[p.exitcode for p in ps], total_iterations=sum(r.get('iterations', 0) for r in res))))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
