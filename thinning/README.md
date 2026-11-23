# Removing the 86 Hz component (optional)

Some EventSym recordings (19% by the detection rule of the paper) contain a weak 86 Hz modulation of the event rate,
a median of 1.4% of the event rate (Appendix A.3 of the paper). Removing it is not necessary for the results reported in the paper,
but `thin_86hz.py` removes it in the same way as the paper does, for users who want to.

The method is event thinning. Events of an affected recording are dropped at random, with a probability that
flattens the 86 Hz modulation. Events are only removed, never moved or added. About 3.6% of the events of an affected
recording are removed (median).

As in the paper, the component is estimated once per recording from its original-resolution file, and the same
thinning is applied at every scale. With the default settings, the output is the thinned data analysed in the paper.

## Files

- `thin_86hz.py`: the script (Python 3.8+, NumPy, pandas).
- `affected_86hz.txt`: the 481 recordings thinned in the paper. Keep it next to the script.

## Usage

The input can be a single file or any folder of the dataset, at any scale:

```
python thin_86hz.py "Dataset/downscaled_x9/traffic/Stop sign/3_events_foveated_events.csv" --out thinned   # one file
python thin_86hz.py "Dataset/downscaled_x9/traffic/Stop sign" --out thinned                               # one subclass
python thin_86hz.py Dataset/downscaled_x9/traffic --out thinned                                           # one category
python thin_86hz.py Dataset --out Dataset_thinned                                                         # everything
python thin_86hz.py Dataset --scales downscaled_x9 --categories traffic hazard --out thinned               # filters
```

- **Output:** `--out` mirrors the dataset layout (`<scale>/<category>/<subclass>/<same file name>`).
- **Untouched recordings:** recordings that are not thinned are copied unchanged, so the output folder can replace
  the original. Use `--only-thinned` to write only the thinned files.
- **Row format:** kept rows are copied byte for byte.
- **Log:** `thin_86hz_log.csv` in the output folder lists the events before and after for every file.

The original-resolution file of each recording is looked up in the same `Dataset` folder. If you have downloaded
only downscaled data, there are two options:

- `--params FILE`: saves the per-recording estimates during a run that has the original-resolution files, and loads
  them in later runs, which then no longer need the original-resolution files. The result is identical to the paper.
- `--estimate-from self`: estimates the component from each file itself. This works with any subset of the data, but
  is not identical to the paper.

## Options

| option | effect |
|---|---|
| `--select paper` (default) | thin the recordings listed in `affected_86hz.txt` |
| `--select detected` | thin the recordings that meet the 86 Hz detection rule |
| `--select all` | thin every recording |
| `--scales`, `--categories`, `--subclasses` | restrict the input |
| `--overwrite` | replace existing output files (by default they are skipped, so an interrupted run can be resumed) |
| `--workers N` | number of parallel processes |
