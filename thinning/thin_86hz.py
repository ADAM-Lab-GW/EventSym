#!/usr/bin/env python
"""
Remove the weak 86 Hz modulation from EventSym recordings by event thinning, as in the paper (Appendix A.3).

A small global 86 Hz modulation of the event rate is present in some recordings. It is removed by thinning: each
event is kept with probability

    q(t) = (1 - m_max) / (1 + m(t) cos(2 pi f0 t + phi(t))),

which flattens the event rate. Events are only removed, never moved or added, and the removed events are chosen at
random, so they are spread evenly over the sensor. f0 is found per recording (84.5-87.5 Hz), and m and phi are
tracked in 1 s windows (hop 0.25 s).

As in the paper, the component is estimated once per recording from its ORIGINAL-RESOLUTION file and the same
thinning function is applied at every scale. With the defaults, the output is identical to the thinned data
analysed in the paper (same recordings, same events removed).

INPUT can be any part of the dataset, at any scale; several inputs can be given:
    one file      Dataset/downscaled_x9/traffic/Stop sign/3_events_foveated_events.csv
    one subclass  Dataset/downscaled_x9/traffic/Stop sign
    one category  Dataset/downscaled_x9/traffic
    one scale     Dataset/downscaled_x9
    everything    Dataset
The original-resolution file of each recording is looked up in the same Dataset folder
(Dataset/original_resolution/<category>/<subclass>/<id>_events.csv). If you only downloaded downscaled data, use
--estimate-from self (estimates from each file itself; not identical to the paper) or a --params file made earlier.

The output mirrors the input layout under OUT (<scale>/<category>/<subclass>/<same file name>). Recordings that are
not thinned are copied unchanged (or skipped with --only-thinned), so OUT is a complete drop-in replacement. Rows are
copied byte for byte; only removed events are dropped. A log (thin_86hz_log.csv) is written to OUT.

Examples:
    python thin_86hz.py Dataset --out Dataset_thinned
    python thin_86hz.py "Dataset/downscaled_x9/traffic/Stop sign" --out thinned
    python thin_86hz.py Dataset --scales downscaled_x9 --categories traffic hazard --out thinned
    python thin_86hz.py Dataset/downscaled_x9 --estimate-from self --out thinned_x9

Requires Python 3.8+, NumPy and pandas.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SCALE_RE = re.compile(r"^(original_resolution|downscaled_x\d+)$")
ORIGINAL = "original_resolution"
AFFECTED_FILE = Path(__file__).resolve().parent / "affected_86hz.txt"

# ---------------------------------------------------------------------------------------------------------------
# Estimation and thinning: unchanged from the code used for the paper.
DT = 1e-4          # 0.1 ms bins for estimation (phase error at 86 Hz < 0.5 % of a cycle)
WIN, HOP = 1.0, 0.25
NULLS = (-9, -7, -5, -3, 3, 5, 7, 9)


def _binned(t):
    n = int(t[-1] / DT) + 1
    c = np.bincount(np.minimum((t / DT).astype(np.int64), n - 1), minlength=n).astype(float)
    return c, (np.arange(n) + 0.5) * DT


def _amp(c, tb, w, f):
    """Complex modulation amplitude a = m e^{i theta} at frequency f (Hann-weighted), and its noise variance."""
    cw = c * w
    N = cw.sum()
    a = 2 * np.sum(cw * np.exp(-2j * np.pi * f * tb)) / N
    return a, 4 * np.sum(w * w * c) / N ** 2


def paper_depth(c, f0=86.0):
    """Modulation depth: 2|DFT| / N of the 0.1 ms count series, maximised over f0 +- 1 Hz."""
    t = np.arange(len(c)) * DT
    x = c - c.mean()
    return max(2 * abs(np.sum(x * np.exp(-2j * np.pi * f * t))) / c.sum() for f in np.arange(f0 - 1, f0 + 1.0001, 0.02))


def paper_detect(c):
    """Detection rule: depth at 86 +- 1 Hz above 3x the mean depth of 5 Poisson surrogates of the 20 ms rate."""
    k = int(0.02 / DT)
    env = np.convolve(c, np.ones(k) / k, mode="same")
    rng = np.random.default_rng(0)
    m = paper_depth(c)
    floor = float(np.mean([paper_depth(rng.poisson(env).astype(float)) for _ in range(5)]))
    return m, floor, bool(m > 3 * floor)


def estimate(t, band=(84.5, 87.5), step=0.005):
    """t: event times in seconds, sorted, starting at 0. Returns f0, global depth m, detection and the window track."""
    c, tb = _binned(t)
    T = tb[-1] + DT / 2
    if T <= 1.5 * WIN:  # ~1 Hz resolution: a wide search locks onto noise; long recordings all lie in 85.5-86.2 Hz
        band = (85.5, 86.5)
    w = np.hanning(len(c))
    fs = np.arange(band[0], band[1] + step / 2, step)
    coarse = fs[::10]  # coarse-to-fine: 0.05 Hz grid, then 0.005 Hz around the best
    pa = [abs(_amp(c, tb, w, f)[0]) for f in coarse]
    fc = coarse[int(np.argmax(pa))]
    fine = fs[(fs > fc - 0.06) & (fs < fc + 0.06)]
    pa = [abs(_amp(c, tb, w, f)[0]) for f in fine]
    f0 = float(fine[int(np.argmax(pa))])
    a0, _ = _amp(c, tb, w, f0)
    if T <= 1.5 * WIN:
        centres, amps = np.array([T / 2]), np.array([a0])
        var = np.array([_amp(c, tb, w, f0)[1]])
    else:
        nwin = int(round(WIN / DT))
        starts = np.arange(0, len(c) - nwin + 1, int(round(HOP / DT)))
        hw = np.hanning(nwin)
        centres, amps, var = [], [], []
        for s in starts:
            a, v = _amp(c[s:s + nwin], tb[s:s + nwin], hw, f0)
            centres.append(tb[s] + WIN / 2); amps.append(a); var.append(v)
        centres, amps, var = np.array(centres), np.array(amps), np.array(var)
    shrink = np.clip(1 - var / np.maximum(np.abs(amps) ** 2, 1e-30), 0, 1)  # shrink noisy windows towards 0
    amps = amps * shrink
    m86, floor, det = paper_detect(c)
    return dict(f0=f0, m=float(abs(a0)), detected=det, centres=centres, amps=amps)


def keep_mask(t, est, rng):
    re_ = np.interp(t, est["centres"], est["amps"].real)
    im = np.interp(t, est["centres"], est["amps"].imag)
    mod = re_ * np.cos(2 * np.pi * est["f0"] * t) - im * np.sin(2 * np.pi * est["f0"] * t)  # Re(a e^{i w t})
    m_max = float(np.max(np.hypot(re_, im)))
    q = np.clip((1 - m_max) / (1 + mod), 0, 1)
    return rng.random(len(t)) < q, m_max


def _seed(name):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------------------------------------------
# File handling

def locate(path):
    """Split .../<scale>/<category>/<subclass>/<file>.csv into (root, scale, category, subclass, recording id)."""
    p = Path(path).resolve()
    parts = p.parts
    if len(parts) < 5 or not SCALE_RE.match(parts[-4]):
        return None
    m = re.match(r"^(\d+)_", p.name)
    if not m:
        return None
    return Path(*parts[:-4]), parts[-4], parts[-3], parts[-2], int(m.group(1))


def find_files(inputs):
    files = []
    for inp in inputs:
        p = Path(inp)
        if not p.exists():
            sys.exit(f"Not found: {inp}")
        cands = [p] if p.is_file() else sorted(p.rglob("*.csv"))
        found = [(c, locate(c)) for c in cands]
        found = [(c, loc) for c, loc in found if loc]
        if not found:
            sys.exit(f"No EventSym event files (<scale>/<category>/<subclass>/<id>_*.csv) found in: {inp}")
        files += found
    seen, out = set(), []
    for c, loc in files:  # drop duplicates from overlapping inputs
        k = str(c.resolve())
        if k not in seen:
            seen.add(k); out.append((c.resolve(), loc))
    return out


def read_times(path):
    """Event times as in the paper's preprocessing: relative to the first event, rounded to whole microseconds,
    sorted (stable). Returns (sorted times in s, order) where order maps sorted position -> data row."""
    t = pd.read_csv(path, usecols=[0]).iloc[:, 0].to_numpy(dtype=np.float64)
    if len(t) == 0:
        return t, np.arange(0)
    t = t - t.min()
    if t.max() < 1e3:  # timestamps in seconds -> microseconds (EventSym CSVs are in microseconds)
        t = t * 1e6
    order = np.argsort(t, kind="stable")
    tus = np.round(t[order]).astype(np.int64)
    return tus * 1e-6, order


def write_rows(src, dst, keep):
    """Copy the header and the kept data rows byte for byte (blank lines are skipped, as pandas does)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    i = -1
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(fi.readline())
        for line in fi:
            if not line.strip():
                continue
            i += 1
            if keep[i]:
                fo.write(line)
    if i + 1 != len(keep):
        tmp.unlink()
        raise RuntimeError(f"{src}: {i + 1} data rows but {len(keep)} timestamps")
    os.replace(tmp, dst)


def est_to_json(e):
    return dict(f0=e["f0"], m=e["m"], detected=e["detected"], centres=e["centres"].tolist(),
                amps_re=e["amps"].real.tolist(), amps_im=e["amps"].imag.tolist())


def est_from_json(d):
    return dict(f0=d["f0"], m=d["m"], detected=d["detected"], centres=np.array(d["centres"]),
                amps=np.array(d["amps_re"]) + 1j * np.array(d["amps_im"]))


def original_file(root, cat, cls, rec):
    d = root / ORIGINAL / cat / cls
    hits = sorted(d.glob(f"{rec}_*.csv")) if d.is_dir() else []
    return hits[0] if hits else None


def process_recording(job):
    """Thin (or copy) every requested file of one recording. Returns log rows and the estimate (if computed)."""
    (root, cat, cls, rec), files, opt, cached = job
    item = f"{cat}/{cls}/{rec}"
    rows, est_json = [], None
    need = opt["select"] in ("detected", "all") or item in opt["affected"]
    est = est_from_json(cached) if cached else None
    if opt["estimate_from"] == "original" and est is None and (need or opt["select"] == "detected"):
        src = original_file(root, cat, cls, rec)
        if src is None:
            return [dict(item=item, error=f"original-resolution file not found under {root / ORIGINAL / cat / cls}; "
                                           "use --estimate-from self or a --params file")], None
        t, _ = read_times(src)
        est = estimate(t)
        est_json = est_to_json(est)
    if opt["select"] == "detected":
        selected = bool(est["detected"]) if est is not None else None
    else:
        selected = need
    for src, scale in files:
        dst = Path(opt["out"]) / scale / cat / cls / src.name
        row = dict(scale=scale, category=cat, subclass=cls, recording=rec, file=src.name, thinned=False)
        if dst.exists() and not opt["overwrite"]:
            row.update(status="exists, skipped"); rows.append(row); continue
        t, order = read_times(src)
        e = est
        if opt["estimate_from"] == "self" and (selected or opt["select"] == "detected"):
            e = estimate(t) if len(t) else None
            if opt["select"] == "detected":
                selected = bool(e and e["detected"])
        row.update(n_before=len(t))
        if selected and e is not None and len(t):
            keep_sorted, m_max = keep_mask(t, e, np.random.default_rng(_seed(item + scale)))
            keep = np.empty(len(t), bool)
            keep[order] = keep_sorted
            write_rows(src, dst, keep)
            row.update(thinned=True, f0_hz=round(e["f0"], 3), m_max=round(m_max, 5), n_after=int(keep.sum()))
        else:
            row.update(n_after=len(t))
            if not opt["only_thinned"]:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        row["removed_pct"] = round(100 * (1 - row["n_after"] / row["n_before"]), 3) if row["n_before"] else 0.0
        row["status"] = "ok"
        rows.append(row)
    return rows, est_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="file(s) or folder(s) of the EventSym Dataset")
    ap.add_argument("--out", required=True, help="output folder (mirrors <scale>/<category>/<subclass>/<file>)")
    ap.add_argument("--scales", nargs="*", help="only these scales, e.g. original_resolution downscaled_x9")
    ap.add_argument("--categories", nargs="*", help="only these categories, e.g. traffic hazard")
    ap.add_argument("--subclasses", nargs="*", help='only these subclasses, e.g. "Stop sign"')
    ap.add_argument("--select", choices=["paper", "detected", "all"], default="paper",
                    help="which recordings to thin: paper = the recordings thinned in the paper (default, listed in "
                         "affected_86hz.txt); detected = recordings that meet the 86 Hz detection rule; all = every "
                         "recording")
    ap.add_argument("--estimate-from", choices=["original", "self"], default="original",
                    help="original = estimate from the original-resolution file and apply at every scale (paper, "
                         "default); self = estimate from each file itself")
    ap.add_argument("--params", help="JSON file of per-recording estimates: read if it exists, updated at the end "
                                     "(lets later runs skip the original-resolution files)")
    ap.add_argument("--only-thinned", action="store_true", help="write only thinned files, do not copy the others")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()

    files = find_files(a.inputs)
    norm = lambda s: s.strip().casefold()
    if a.scales:
        files = [f for f in files if f[1][1] in a.scales]
    if a.categories:
        files = [f for f in files if norm(f[1][2]) in {norm(c) for c in a.categories}]
    if a.subclasses:
        files = [f for f in files if norm(f[1][3]) in {norm(c) for c in a.subclasses}]
    if not files:
        sys.exit("No files left after --scales/--categories/--subclasses.")
    out = Path(a.out).resolve()
    for r in {f[1][0] for f in files}:  # output inside the dataset would be picked up as input by later runs
        if out == r or r in out.parents:
            sys.exit(f"--out must be outside the dataset folder ({r}), e.g. next to it.")
    if a.select == "paper" and not AFFECTED_FILE.exists():
        sys.exit(f"{AFFECTED_FILE.name} not found next to the script; it lists the recordings thinned in the paper.")
    affected = ({l.strip() for l in open(AFFECTED_FILE, encoding="utf-8") if l.strip()}
                if a.select == "paper" else set())

    params = {}
    if a.params and Path(a.params).exists():
        params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    groups = {}
    for src, (root, scale, cat, cls, rec) in files:
        groups.setdefault((root, cat, cls, rec), []).append((src, scale))
    opt = dict(select=a.select, estimate_from=a.estimate_from, affected=affected, out=str(out),
               only_thinned=a.only_thinned, overwrite=a.overwrite)
    jobs = [(k, sorted(v, key=lambda x: x[1]), opt, params.get(f"{k[1]}/{k[2]}/{k[3]}")) for k, v in groups.items()]
    jobs.sort(key=lambda j: (j[0][1], j[0][2], j[0][3]))
    print(f"{len(files)} files from {len(jobs)} recordings; select={a.select}, estimate from {a.estimate_from}; "
          f"output: {out}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    pool = Pool(a.workers) if a.workers > 1 else None
    run = pool.imap_unordered if pool else map
    for i, (rr, est_json) in enumerate(run(process_recording, jobs), 1):
        for r in rr:
            (errors if "error" in r else rows).append(r)
        if est_json is not None and a.estimate_from == "original":
            r0 = rr[0]
            params[f"{r0['category']}/{r0['subclass']}/{r0['recording']}"] = est_json
        if i % 25 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} recordings", flush=True)

    if rows:
        log = out / "thin_86hz_log.csv"
        keys = ["scale", "category", "subclass", "recording", "file", "thinned", "f0_hz", "m_max", "n_before",
                "n_after", "removed_pct", "status"]
        with open(log, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["category"], r["subclass"], r["recording"], r["scale"])))
        th = [r for r in rows if r["thinned"]]
        print(f"{len(th)} of {len(rows)} files thinned"
              + (f"; median {np.median([r['removed_pct'] for r in th]):.2f}% of their events removed" if th else "")
              + f". Log: {log}")
    if a.params:
        Path(a.params).write_text(json.dumps(params), encoding="utf-8")
    for e in errors:
        print(f"ERROR {e['item']}: {e['error']}", file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
