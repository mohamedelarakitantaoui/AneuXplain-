# Batch Evaluation Scripts

Two one-shot scripts for the 50-subject AneuXplain prototype evaluation
(25 patients + 25 controls from the Lausanne TOF-MRA Aneurysm Cohort,
ds003949).

## Files

| Script | Purpose |
|--------|---------|
| `generate_click_points.py` | Reads lesion mask NIfTIs and writes `click_points.json` (per-patient lesion centroids in mesh coordinates; mean centroid for all controls). |
| `batch_evaluate.py`        | Pushes all 50 subjects through the backend pipeline (`/dicom/upload` → `/dicom/segment/{sid}` → `/dicom/crop-and-analyze`) and logs every result to a CSV. |

## Prerequisites

1. **Backend running** at `http://localhost:8000`. Start it however you
   normally do (e.g. `uvicorn backend.app.main:app`). The script does
   *not* start it for you.
2. **`click_points.json` already generated.** Run
   `python scripts/generate_click_points.py` first; it writes to
   `C:\Users\buzok\Desktop\Test-demo\click_points.json`.
3. **Angio NIfTI files** present in
   `C:\Users\buzok\Desktop\Test-demo\angio\` — one
   `sub-XXX_ses-YYYYMMDD_angio.nii` per subject (50 total).
4. **Python packages**: `requests`, `nibabel`, `numpy`. Install missing
   ones with `pip install <name> --break-system-packages`.

## Running the batch

```bash
python scripts/batch_evaluate.py
```

The script first runs sanity checks (file presence, backend reachable)
and prints a plan, then prompts:

```
Proceed with batch run? [y/N]
```

Type `y` to begin. Per-subject progress is printed live:

```
[ 1/50] sub-013 (patient) ... success in 8.3s (risk=0.73)
[ 2/50] sub-022 (patient) ... success in 7.1s (risk=0.82)
[ 3/50] sub-026 (patient) ... FAILED at segment step: HTTP 400: ...
```

## Estimated runtime

Each subject takes roughly **5–15 seconds** end-to-end (upload + segment
+ analyze) on a workstation with a warm cache, so the full 50-subject
batch should finish in **5–15 minutes**. Cold caches and slower disks can
push this to ~30 minutes.

## Resuming after a crash or Ctrl-C

`batch_evaluate.py` is **resumable**: the CSV is rewritten after every
subject, and on startup it re-reads the existing CSV and skips any
subject already marked `pipeline_status=success`. Just rerun the script.

To force a full rerun, delete (or rename) `results\batch_results.csv`
before starting.

Failed subjects are *not* skipped on resume — they will be retried.

## Outputs

All under `C:\Users\buzok\Desktop\Test-demo\results\`:

| File / folder | Contents |
|---------------|----------|
| `batch_results.csv` | One row per subject: subject_id, group, click_point, lesion stats (patients only), pipeline_status, per-stage timings, session_id, risk_score, predicted_class, 8 morphology measurements, neck_detection_tier, raw response file path. |
| `raw_responses\{subject_id}.json` | Full JSON returned by `/dicom/crop-and-analyze` for every subject. For failures, a small marker JSON with `failed_stage` and `error`. |
| `batch.log` | Timestamped log of everything that happened, useful for post-mortem on failures. |

## Interpreting `batch_results.csv`

- **`pipeline_status`** is the first thing to check. Values:
  `success`, `upload_failed`, `segment_failed`, `analyze_failed`,
  `timeout`. Anything other than `success` means morphology/risk columns
  are blank for that row.
- **`risk_score`** is the model's predicted aneurysm risk in `[0, 1]`.
  `predicted_class` is the corresponding bucket
  (`LOW` / `MODERATE` / `HIGH` / `CRITICAL`).
- **Morphology columns** (`neck_width`, `dome_height`,
  `max_dome_diameter`, `aspect_ratio`, `dome_to_neck_ratio`,
  `irregularity_index`, `volume`, `surface_area`) come from the backend
  morphology analyzer. They are blank when the analyzer reports
  `status="failed"` for that measurement (e.g. degenerate mesh,
  no neck plane found).
- **Patients vs controls.** Patients use a per-subject click point
  derived from the manually-annotated lesion mask. All controls share
  one click point: the mean of the 25 patient centroids
  ("analyzed at the anatomical region where patient aneurysms were
  observed on average"). This means controls also get a risk score and
  morphology readout — that is the intended design, so the model can be
  evaluated on its discrimination.

## Configuration

All knobs (base URL, paths, crop radius, HTTP timeout) are constants at
the top of `batch_evaluate.py`. Edit them there if you need to point at
a different backend port or output directory.
