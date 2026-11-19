# Changes for the review release

## Source selection

The 18 November 2026 manuscript and accompanying supplementary tables define the scope. The source selected was the frozen 10 October formal experiment with 719 held-out reference samples and 239 bands. Its formal identity bound 33 source files. All 33 matched the recovered files on read-only SHA-256 comparison before any copying. The paired analysis and PI sensitivity scripts were also checked against their historical hashes. Independently, during recovery the 58 saved OOF sets had been checked against their saved metrics. The paired analysis code was selected for this cohort. Earlier ten-fold runs, alternate imagery runs and the 652-reference/229-band comparison were excluded.

## Portability and privacy

- Replaced hard-coded workstation paths and old provenance manifests with configuration-relative input paths and separately created run directories.
- Preserved the fold-local reducer/cleaner implementations, seed derivation, classifier decision and selection rules, reference candidates, module registry, and full paper grids. The source reconstruction preserves filtering, smoothing, continuum removal and differentiation. The GF-5 indices use the original nearest-band and denominator rules.
- Replaced source-specific row and ID assertions, private preflight evidence and previous filesystem bindings with general checks or aggregate cohort constraints for the paper profile. Run identity now binds the local inputs, new split files, code and protocol.
- Added an input adapter for authorized frozen arrays and optional raster input templates. The GLO-30 derivative helper exposes the historical derivation as a command taking a prepared DEM; its export normalizes nonfinite raster values to NoData.
- Resumed phases are recorded separately. Replaced platform-specific CSV newline handling with the writer default. Paired FS comparisons are oriented as FS minus FALL to match the 18 November reporting direction.
- Retained the paper's registered main and alternative PI thresholds. Alternative results and any model checkpoints are generated only in the local run.
- Rebuilt the figure and table outputs around local analysis results, replacing fixed article values, plots dependent on desktop applications and original layout with English review renditions. S1 definitions were carried over from the submitted-version supplementary content. The exact presentation assets and workflow limitations are listed in [CODE_MAP.md](CODE_MAP.md).
- Added a deterministic invented-data generator and a reduced demo profile. The demo uses the same branch and folding structure with an intentionally smaller search grid and training budget. It is for testing commands and output structure.
- Aligned the demo's generated output path with the run-directory check by placing it under `runs/`. Clarified that the release file audit runs on a clean source snapshot.
- Reported the actual prepared input CRS in the spatial-fold metadata for both paper and demo runs.
- Restored the separation between factor names and codes in the 63-row S1 CSV after checking the values against the Word supplementary table and factor catalog.
- Removed all existing OOF, split records, fitted state, cache entries, checkpoint records, logs, data manifests with private paths/hashes, study coordinates, label assignments, IDs, raster data, and paper files. Static protocol files contain class names, factor names and registered configuration values. The package has no repository history or embedded runtime.
