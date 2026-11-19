"""Stage authorized, already assembled inputs in a new private paper run."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    os.environ["GF5_CONFIG"] = str(args.config.resolve())

    import numpy as np
    import pandas as pd
    from settings import CLASS_NAMES_ZH, INPUT, MODE, PROTOCOL, ROOT
    from v2_features import sha, writej

    if MODE != "paper":
        raise ValueError("Import requires the paper profile.")
    if ROOT.exists() and any(ROOT.iterdir()):
        raise FileExistsError("Use a new, empty private output_dir.")
    fields = ("sample_id", "class_code_0based", "is_real", "is_virtual",
              "easting", "northing", "exact_coordinate_group_id")
    frame = pd.read_csv(args.metadata, encoding="utf-8-sig", float_precision="round_trip")
    frame = frame.loc[:, list(fields)].copy()
    if len(frame) != 771 or not frame.sample_id.is_unique:
        raise ValueError("Expected the complete, unique 771-row frozen cohort")
    if ((int(frame.is_real.sum()), int(frame.is_virtual.sum())) != (719, 52)
        or not frame.is_virtual.isin((0, 1)).all() or not (frame.is_real + frame.is_virtual == 1).all()
        or not frame.class_code_0based.isin(range(6)).all()
        or not np.isfinite(frame[["easting", "northing"]]).all().all()):
        raise ValueError("Invalid frozen labels, sample roles or EPSG:32648 coordinates")
    frame["soil_class_canonical"] = np.asarray(CLASS_NAMES_ZH)[frame.class_code_0based]
    frame["old_exact_coordinate_group_id"] = ""
    with np.load(args.arrays, allow_pickle=False) as z:
        fields = ("sample_id", "OR", "CR", "FDR", "ENV63", "ENV63_feature_names", "wavelength_nm")
        arrays = {k: z[k] for k in fields}
        arrays["segments"] = z["segments"] if "segments" in z.files else z["segment_id"]
        if not all(np.array_equal(z[key], frame[col]) for key, col in {
            "sample_id": "sample_id", "y": "class_code_0based", "is_virtual": "is_virtual"
        }.items() if key in z.files):
            raise ValueError("Frozen arrays and metadata are not aligned")
    if any(arrays[k].shape != (771, 239) or not np.isfinite(arrays[k]).all() for k in ("OR", "CR", "FDR")):
        raise ValueError("Expected finite 771 by 239 OR, CR and FDR arrays")
    wave, segments = arrays["wavelength_nm"], arrays["segments"]
    if wave.shape != (239,) or not np.isfinite(wave).all() or not (np.diff(wave) > 0).all():
        raise ValueError("Invalid frozen wavelength array")
    if segments.shape != wave.shape or not np.array_equal(
        segments, np.r_[1, 1 + np.cumsum(np.diff(wave) > 3 * np.median(np.diff(wave)))]
    ) or set(segments) != {1, 2, 3, 4}:
        raise ValueError("Invalid frozen spectral segments")
    catalog = pd.read_csv(PROTOCOL / "factor_catalog_63.csv")
    if (arrays["ENV63"].shape != (771, 63) or np.isinf(arrays["ENV63"]).any()
        or not np.array_equal(arrays["ENV63_feature_names"], catalog.factor)):
        raise ValueError("Invalid environmental arrays or factor order")
    INPUT.mkdir(parents=True, exist_ok=True)
    (ROOT / ".gf5-run").touch()
    frame.to_csv(INPUT / "metadata.csv", index=False, encoding="utf-8", float_format="%.17g")
    np.savez_compressed(INPUT / "modeling_inputs.npz", **arrays)
    writej(INPUT / "preparation.json", {
        "profile": "paper", "mode": "authorized_frozen_inputs", "reference_n": 719, "virtual_n": 52,
        "retained_bands": 239, "projected_crs": "EPSG:32648",
        "metadata_sha256": sha(INPUT / "metadata.csv"),
        "arrays_sha256": sha(INPUT / "modeling_inputs.npz"),
    })
    print("Prepared the authorized frozen input for local analysis.")


if __name__ == "__main__":
    main()
