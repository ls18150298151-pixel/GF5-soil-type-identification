"""Generate invented review fixtures. This module consumes no source data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate(config_path):
    if config_path.exists():
        raise FileExistsError("Use a new path for the demo configuration.")
    fixture = config_path.parent / (config_path.stem + "-synthetic")
    fixture.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(70412)
    classes = np.tile(np.arange(6), 24)
    groups = np.repeat(np.arange(24), 6)
    flag = np.zeros(len(classes), dtype=int)
    # Include synthetic training-only candidates in the same grid blocks.
    classes = np.r_[classes, np.arange(6), np.arange(6)]
    groups = np.r_[groups, np.arange(12)]
    flag = np.r_[flag, np.ones(12, dtype=int)]
    ids = np.arange(800001, 800001 + len(classes))
    samples = pd.DataFrame({
        "sample_id": ids, "class_code": classes, "is_virtual": flag,
        "x": 500200 + 3000 * (groups % 6) + rng.uniform(5, 45, len(ids)),
        "y": 200 + 3000 * (groups // 6) + rng.uniform(5, 45, len(ids)),
    })
    samples.to_csv(fixture / "samples.csv", index=False)
    wave = np.r_[405.0, np.linspace(445, 895, 24), 1000.,
                 np.linspace(1065, 1335, 15), np.linspace(1470, 1785, 18),
                 np.linspace(1965, 2385, 23)]
    bands = [f"band_{i:03d}" for i in range(1, len(wave) + 1)]
    pd.DataFrame({"band": bands, "wavelength_nm": wave}).to_csv(fixture / "wavelengths.csv", index=False)
    center = 575 + 42 * classes[:, None]
    reflectance = (0.32 + 0.035 * np.sin(wave[None, :] / 140)
                   - 0.05 * np.exp(-((wave[None, :] - center) / 105)**2)
                   + rng.normal(0, 0.012, (len(ids), len(wave))))
    spectra = pd.DataFrame(np.rint(reflectance * 10000).astype(int), columns=bands)
    spectra.insert(0, "sample_id", ids)
    spectra.to_csv(fixture / "spectra.csv", index=False)
    catalog = pd.read_csv(Path(__file__).resolve().parents[1] / "protocol/factor_catalog_63.csv")
    names = catalog.loc[catalog.module != "E", "factor"].tolist()
    env = rng.normal(size=(len(ids), len(names))) + classes[:, None] * rng.normal(0, .12, len(names))
    env[rng.random(env.shape) < .02] = np.nan
    data = pd.DataFrame(env, columns=names)
    data["LC_class"] = rng.choice([10, 20, 30, 40, 50, 60], len(ids))
    data.insert(0, "sample_id", ids)
    data.to_csv(fixture / "environment.csv", index=False)
    local = fixture.name
    config = {
        "profile": "demo", "output_dir": f"runs/{config_path.stem}-run",
        "inputs": {
            "samples_csv": f"{local}/samples.csv", "sample_crs": "EPSG:32631",
            "projected_crs": "EPSG:32631",
            "sample_fields": {k: k for k in ("sample_id", "x", "y", "class_code", "is_virtual")},
            "wavelengths_csv": f"{local}/wavelengths.csv", "spectra_csv": f"{local}/spectra.csv",
            "reflectance_scale": 10000, "environment_csv": f"{local}/environment.csv",
        },
    }
    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(f"Generated invented fixtures and configuration for {len(ids)} rows.")
