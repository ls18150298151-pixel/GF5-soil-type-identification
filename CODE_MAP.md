# Manuscript to code map

Commands below use `python -B code/run.py --config config.local.json` unless an entry explicitly names a tool. See [README.md](README.md) for complete invocations.

| Manuscript item | Command and implementation |
| --- | --- |
| Sections 2.2 and 3.1–3.2, point input, spectral validity, OR/CR/FDR | `prepare` in `code/prepare.py`, with the retained spectral algorithms in `code/spectral_preprocessing.py`. `tools/import_frozen.py` accepts the authorized, aligned precomputed arrays. |
| Section 3.3, fold-local CARS, PCA, PLS-VIP, SPA | `preflight` in `code/preflight.py` uses `code/v2_features.py`, `code/feature_algorithms.py` and `code/cars_adapter.py`. |
| Section 3.4 and S1, terrain, GF-5 indices, climate, WorldCover and hydrology | `prepare` builds and samples inputs. `tools/build_glo30_factors.py` constructs H derivatives on the prepared 30 m grid. `code/environment_cleaner.py` and `code/v2_features.py` fit environmental cleaning, scaling and central land-cover one-hot encoding within the current fit boundary. Factor definitions live in `protocol/S1_factor_definitions.csv`. |
| Section 3.5 and Tables 1–4 | `train` uses `code/training_contract.py` and the classifier registry `protocol/candidate_pool_36.json`. `report` exports the unique parameter sets and the fixed settings remain defined in the training contract. |
| Section 3.6 and Fig. 5 | `split` in `code/spatial_folds.py` searches the prespecified offsets and seeds with class and spatial metadata, then records the R and R+V boundaries. `report` plots the five outer folds locally. |
| Section 4.2 and Fig. 7, 48 families and FDR backbone | `--phase R_spectral train` in `code/formal_runner.py`. The spectral registry defines candidates and `aggregate` pools family OOF. `report` produces three heatmaps and a five-fold backbone panel. |
| Sections 3.4.2 and 4.3, S0/N59/I63/FALL/FS, Fig. 8 and Table 5 | `--phase R_fusion train` uses the R spectral backbone and evaluates the four remaining branches. FS uses each family's unweighted Set 1 reference configuration for the 32-subset module search and three permutations per inner fold. `aggregate` and `report` produce the five-branch summary and per-fold dimensions. |
| Sections 4.3.2–4.3.3 and Fig. 9, paired gains and FS stability | `analyze` in `code/paired_analysis.py` links the reference OOF by ID privately, calculates the fixed-prediction block bootstrap and summarizes retained factors. `report` draws OA, Macro-F1 and BA differences. |
| Section 4.4.1 and Table 6, R+V | `--phase RV_spectral train`, `--phase RV_fusion train`, `aggregate` then `analyze` and `report`. R and R+V share reference evaluation rows. |
| Section 4.4.1 and Table 7, PI thresholds | `pi-train` retunes changed FS subsets and reuses identical subsets. `pi-aggregate` audits and compiles the threshold values, then `report` adds the primary threshold and its deltas. |
| Section 4.4.1 and Fig. 10, fixed classifier fold comparisons | `report` reads the 20 R/FALL family checkpoints. Each fixed family uses inner tuning before outer scoring. |
| Section 4.4.2, Figs. 11–12 | `aggregate` creates pooled FALL confusion and FALL/FS precision, recall and F1. `report` creates the labeled confusion matrix and class metric comparison. |
| Figs. 1–3 and 6 | `report` optionally draws a study-area outline, uses the supplied private coordinates for the sample map, draws an English workflow schematic, and plots class means ±1 sample SD (`ddof=1`) from local OR/CR/FDR arrays. |
| Supplementary Tables S1–S4 | `report` writes definitions, the 25 R workflows, the parameters of the classifiers used by those workflows, and the FS factor union and occurrence flags. S1 descriptions are static, while S2–S4 are derived from the actual local run. |

## Steps outside the released workflow

- **Working GF-5 image.** Radiometric and atmospheric correction, geometric processing, mosaicking and clipping made the paper's working raster. The original working raster body and complete processing project were not located in recovery. The released reader starts with an authorized working image or equivalent extracted spectra. The alternate 6S experiment has a different cohort and band set and was excluded.
- **Original terrain products.** The frozen merged DEM and 11-band terrain raster were unavailable in the recovery directory. The raster path requires authorized versions of those products. The frozen-matrix import can start downstream.
- **Figures and layout.** Figures are review renditions from the recovered analysis logic. The original Fig. 1 inset maps and terrain background, final overlays, fonts and English finishing choices require separate manual assets or layout. The Fig. 4 combined classifier mechanism illustration was assembled separately. Its exact source was not located. Figure 7 supplies its backbone inset data as a CSV. These outputs use CSV/PNG/SVG. The composition of Word tables and exporting to the submitted PDF were separate publishing steps.
