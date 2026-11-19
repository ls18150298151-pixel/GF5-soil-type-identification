# GF-5 soil type identification

Analysis code for *Selection of feature–model combinations and soil type identification using GF-5 hyperspectral imagery*, prepared for review of the manuscript dated 18 November 2026. The analysis compares six soil classes in Luxi County, 48 fixed spectral representation–reduction–classifier families, and five adaptive input branches. The paper profile keeps the original candidate grids and selection rules. Data availability is described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). The script index is in [CODE_MAP.md](CODE_MAP.md).

## Installation

Use 64-bit Python 3.10. The recorded production runtime used Python 3.10.20. Create and activate a virtual environment outside the source folder, then install the pinned Python packages from the source folder.

```text
python -m venv ../gf5-env
```

Activate with `../gf5-env/Scripts/Activate.ps1` in PowerShell or `source ../gf5-env/bin/activate` in a POSIX shell.

```text
python -m pip install -r requirements.txt
```

In a clean source snapshot, run `python -B tools/audit_release.py --imports` to check the file allow-list, syntax, candidate registries, embedded credential and local path patterns, and imports. This release check expects a snapshot without Git metadata or local run files. It reports installed versions alongside the pinned versions. Windows LightGBM, XGBoost and geospatial wheels may require a working native runtime. Work in a single process sequence per run directory.

## Invented-data example

Run from this folder. Choose a fresh ignored configuration name. The generator creates an invented cohort in a different projected CRS, with 144 reference rows, 12 training-only candidates and 80 retained bands. The demo keeps the nested split structure, 48 fixed families, five branches, FS module search, R+V, and reporting. It reduces the candidate grids, tree counts and CARS iterations. A successful demo validates software behavior on the invented inputs.

```text
python -B code/run.py --config demo.json make-demo
python -B code/run.py --config demo.json --workers 2 all
```

The two commands create `demo-synthetic/` and `runs/demo-run/` locally. Neither needs input from the paper.

## Authorized paper inputs

Copy `config.example.json` to `config.local.json`. Adjust its file paths and the sample column map. Paths are resolved relative to the configuration file. An existing nonempty directory without the run marker is rejected. Use a fresh dedicated output directory for changed inputs, settings, profiles or analysis code.

| Input | Format and semantics |
| --- | --- |
| Samples | CSV in unchanged source row order. Configure the columns for unique integer sample ID, x, y, class code 0–5 and is_virtual 0 or 1. Use 0 for reference points and 1 for virtual candidates. |
| Coordinates | Original x/y use CGCS2000 Gauss–Krüger zone 18 with 6° zone width (EPSG:4496). The paper projects them to WGS 84 / UTM zone 48N (EPSG:32648), in metres, with x/y axis order. Exact previously projected coordinates in EPSG:32648 are also accepted. |
| Class codes | 0 red soil, 1 purple soil, 2 dry red soil, 3 yellow-brown soil, 4 limestone soil and 5 yellow soil. |
| Wavelengths | CSV with `band,wavelength_nm`. Rows are ordered by strictly increasing native band centres and their identifiers match the spectra CSV columns or raster band order. |
| Spectra | CSV with `sample_id` and one raw value column per wavelength row, or a georeferenced working raster with the same band order. Use scale 10000 for the paper. Fully empty or zero spectra are excluded, other incomplete spectra fail validation. |
| Environment table | CSV keyed by `sample_id`, with the 59 non-GF-5 columns from `protocol/factor_catalog_63.csv`. Values are physical values, already scale/offset decoded. Blank cells are missing. `LC_class` is a nominal numeric class code. The four GF-5 indices are computed from retained OR reflectance. |
| Environment rasters | Optional alternative in `config.rasters.example.json`. Supply the merged DEM, existing 11-band terrain raster, 20 H and eight C grids, and native-grid WorldCover. Native CRS and masks are read from the rasters. CHELSA scale and offset must be populated in source band metadata when supplying encoded rasters. |
| Study-area outline | Optional WGS 84 Polygon or MultiPolygon GeoJSON. Used only for local plot overlays. |

The paper input contract expects 280 working bands, 239 valid bands in four continuous segments, and 719 reference plus 52 virtual rows after spectral validity filtering. Savitzky–Golay smoothing uses an 11-band window and order 2 per segment. CR uses the upper hull and FDR uses the actual wavelength intervals. The environment raster path samples by floor on each native grid, handles the terrain aspect transform and WorldCover windows of 9, 27 and 91 cells per side, and retains missing values until the fold-local stage. The optional GLO-30 entry expects the already projected 30 m working DEM.

```text
python -B tools/build_glo30_factors.py --dem private/projected_glo30.tif --output-dir private/derived-glo30
```

An authorized analyst who already has the frozen matrices can load them directly. Supply the aligned metadata CSV and model-input NPZ under a private path. The NPZ must contain the original IDs, OR, CR, FDR, ENV63, the ordered ENV63 names, wavelengths and spectral segments. The metadata needs the paper's projected easting and northing, labels, source roles, and exact coordinate group field. The import checks cohort, shape and order and writes copies of the necessary fields into a new run. Use this **in place of** `prepare`.

```text
python -B tools/import_frozen.py --config config.local.json --metadata private/frozen_metadata.csv --arrays private/frozen_arrays.npz
```

## Step-by-step commands

```text
python -B code/run.py --config config.local.json prepare
python -B code/run.py --config config.local.json split
python -B code/run.py --config config.local.json --workers 2 preflight
python -B code/run.py --config config.local.json --workers 2 train
python -B code/run.py --config config.local.json aggregate
python -B code/run.py --config config.local.json analyze
python -B code/run.py --config config.local.json --workers 2 pi-train
python -B code/run.py --config config.local.json pi-aggregate
python -B code/run.py --config config.local.json report
```

For a fresh run starting from raw-format inputs, `all` executes that sequence. For a long or interrupted run, complete individual training phases in the order shown. S0 is the best inner-ranked FDR spectral configuration per outer fold. Fusion phases require that policy's finished FDR spectral checkpoints. `aggregate` expects all four stages. For interrupted training, use the same configuration and unchanged code. The runner checks the bound input and code hashes before it can resume.

```text
python -B code/run.py --config config.local.json --workers 2 --phase R_spectral train
python -B code/run.py --config config.local.json --workers 2 --phase R_fusion train
python -B code/run.py --config config.local.json --workers 2 --phase RV_spectral train
python -B code/run.py --config config.local.json --workers 2 --phase RV_fusion train
```

The paper profile uses 36 classifier candidates, 972 spectral configurations, 32 environmental module combinations and 900 fold-local spectral feature builds. The original main training budget was 36,960 inner classifier fits and 480 outer refits. R+V uses the same reference evaluation data and folds as R. Virtual candidates in the current held blocks are excluded from training, including the outer held blocks during inner tuning. Parallel workers each use one classifier and BLAS thread.

## Outputs

| Directory | Contents |
| --- | --- |
| prepared/ and spatial/ | Validated arrays, private sample metadata, 2 km blocks, inner/outer manifests and split diagnostics |
| features/, fits/, models/, predictions/, checkpoints/ and FS/ | Fit state, model files, fold-only predictions, selections, module scans and permutation importance |
| results/ | 58 OOF tables, aggregate and per-class metrics, fold metrics, confusion matrix and selected pipelines |
| analysis/ | Paired conditional spatial-block intervals over fixed OOF predictions, R+V differences, feature stability and dimension tables |
| pi_sensitivity/ | Alternative mean-PI thresholds 0.0005 and 0.002, QA, restricted OOF and threshold performance |
| paper_outputs/ | Main Tables 1–7, Supplementary Tables S1–S4 in CSV, and review renditions of Figures 1–3 and 5–12 where inputs are available |
| qa/ | Preflight and outcome validation |

The 95% conditional intervals sample the same 2 km blocks for both predictions in each comparison. The 2,000 replicates keep OOF predictions fixed. They do not capture uncertainty from model retraining or constitute external validation.

Treat local run directories and figure exports from real inputs as private. In particular, point maps, OOF, fold files, model/cache artifacts and logs can reveal or encode restricted sample information.
