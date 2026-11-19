# Local review checks

Date 19 November 2026.

## Source and privacy checks

- Read-only checks matched all 33 files bound by the formal 719-reference, 239-band run identity. The analysis and PI runner matched their separate historical hashes. The alternate image experiment had a different cohort and was excluded.
- The release directory was checked against an explicit allow-list. The only CSVs are static factor catalogs and S1 definitions. JSON files are blank input templates or method registries. No original data, generated files, sample IDs, coordinates, label assignments, credentials, machine paths, binary material, submission files or repository history were found in the release. The review also checked identifiers previously embedded in hard-coded acceptance assertions.
- All 21 Python files parsed and the 20 imported analysis and helper modules loaded. The public registry checks confirmed 36 model configurations, 972 spectral candidates, 48 families, 32 module subsets and the 63/59 environmental catalogs.

## Invented-data checks

The demo generated 144 invented references, 12 invented virtual candidates and 80 retained bands. It used reduced candidate settings. Source-code execution used local Python 3.10.20.

| Step | Observed result |
| --- | --- |
| CSV preparation and 2 km spatial nested validation | Prepared all 156 rows, built five outer and four inner folds with both policies |
| Feature preflight | 400 fold-local builds and 2,162 checks, including changes to held-out inputs and checks for held-out leakage |
| Nested training and result aggregation | 48 spectral families and five branches with R and R+V, 7,200 inner fits, 480 outer refits and 77,678 result checks. All 58 OOF sets had 144 reference predictions each |
| Paired analysis | Generated 30 metric contrasts from 2,000 spatial-block bootstrap replicates |
| Alternative FS cutoffs | Both alternatives completed and passed 197 checks, including the paths for reusing an unchanged subset and refitting a changed subset |
| Weighted models | All four weighted classifier paths accepted invented, training-only inputs and produced six normalized class probabilities |
| Raster and frozen input adapters | An invented georeferenced raster fixture exercised native-cell sampling, the 59-factor environment raster route and the 19 derived GLO-30 outputs. A separate invented 771-by-239 fixture exercised frozen-array alignment and import |
| Report generation | Generated Tables 1–7 and S1–S4, and PNG/SVG renditions for all 11 supported figure numbers. The Fig. 1 outline used invented geometry |

The installed NumPy 1.26.4, pandas 1.4.4, LightGBM 4.7.0 and pyproj 3.6.1 differed from the paper pins. The other listed libraries imported at the required versions. The pinned dependency set and the exact paper profile have not been executed together. Real data and model outputs were never used for release smoke tests, and the tests do not establish reproduction of numerical manuscript results. Fig. 4 and the exact submission figure styling are described in [CODE_MAP.md](CODE_MAP.md).

## Independent release review

The clean 37-file source package passed the syntax, registry, sensitive-pattern and import checks. After correcting the README demo's output path, a fresh invented-data run completed the entire `all` command through report generation. It passed 2,162 feature integration checks, 77,678 results checks and 197 PI sensitivity checks. The review also checked all 63 S1 factor names and codes against the supplementary document and the factor catalog. A focused repeat of `prepare` and `split` verified that the CRS metadata reports the actual demo CRS and that the seven generated spatial CSVs are byte-identical before and after that metadata change. The independent runs used the same local dependency versions listed above.

## Complete package tree

```text
GF5-soil-type-identification/
├── .gitignore
├── CHANGES_FOR_RELEASE.md
├── CODE_MAP.md
├── DATA_AVAILABILITY.md
├── README.md
├── VERIFICATION.md
├── config.example.json
├── config.rasters.example.json
├── requirements.txt
├── code/
│   ├── accept_PI_sensitivity.py
│   ├── aggregate.py
│   ├── cars_adapter.py
│   ├── environment_cleaner.py
│   ├── feature_algorithms.py
│   ├── formal_runner.py
│   ├── paired_analysis.py
│   ├── paper_outputs.py
│   ├── preflight.py
│   ├── prepare.py
│   ├── run.py
│   ├── run_PI_sensitivity.py
│   ├── settings.py
│   ├── spatial_folds.py
│   ├── spectral_preprocessing.py
│   ├── synthetic.py
│   ├── training_contract.py
│   └── v2_features.py
├── protocol/
│   ├── PI_sensitivity_registration.json
│   ├── S1_factor_definitions.csv
│   ├── candidate_pool_36.json
│   ├── factor_catalog_63.csv
│   ├── module_combinations_32.json
│   ├── spectral_candidate_registry_972.json
│   └── training_protocol.json
└── tools/
    ├── audit_release.py
    ├── build_glo30_factors.py
    └── import_frozen.py
```

The complete command sequence is in [README.md](README.md). The package is staged for local author review.
