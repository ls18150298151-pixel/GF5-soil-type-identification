"""Assemble validated point inputs without learning from validation data."""

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from settings import CLASS_NAMES_ZH, CONFIG, INPUT, MODE, PROTOCOL, ROOT, resolve_path
from spectral_preprocessing import BAD_INTERVALS, continuum_remove_one, sg_per_segment
from v2_features import sha, writej


def _integer_column(values, label):
    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy()
    if not np.isfinite(array).all() or (array != np.floor(array)).any():
        raise ValueError(f"{label} must contain finite integers")
    return numeric.astype("int64")


def read_samples(config):
    names = config["sample_fields"]
    tab = pd.read_csv(resolve_path(config["samples_csv"]),
                      encoding=config.get("sample_encoding", "utf-8-sig"), float_precision="round_trip")
    tab = tab[[names[k] for k in ("sample_id", "x", "y", "class_code", "is_virtual")]].copy()
    tab.columns = ["sample_id", "x", "y", "class_code_0based", "is_virtual"]
    for col in ("sample_id", "class_code_0based", "is_virtual"):
        tab[col] = _integer_column(tab[col], col)
    if not tab.sample_id.is_unique or tab.empty:
        raise ValueError("sample_id must be unique and the sample table must be nonempty")
    if not tab.class_code_0based.isin(range(6)).all() or not tab.is_virtual.isin((0, 1)).all():
        raise ValueError("Expected class codes 0–5 and is_virtual values 0 or 1")
    raw_xy = tab[["x", "y"]].to_numpy(float)
    if not np.isfinite(raw_xy).all():
        raise ValueError("Sample coordinates must be finite")
    source = CRS.from_user_input(config["sample_crs"])
    target = CRS.from_user_input(config["projected_crs"])
    if not target.is_projected or any(axis.unit_conversion_factor != 1 for axis in target.axis_info[:2]):
        raise ValueError("The modeling CRS must be projected in metres")
    if MODE == "paper" and target.to_epsg() != 32648:
        raise ValueError("The paper profile requires EPSG:32648")
    if MODE == "paper" and source.to_epsg() not in (4496, 32648):
        raise ValueError("Supply paper coordinates in EPSG:4496 or their exact EPSG:32648 transform")
    project = Transformer.from_crs(source, target, always_xy=True)
    x, y = project.transform(raw_xy[:, 0], raw_xy[:, 1], errcheck=True)
    tab["easting"], tab["northing"] = x, y
    group = tab.groupby(["easting", "northing"], sort=False).ngroup()
    tab["exact_coordinate_group_id"] = np.where(
        tab.duplicated(["easting", "northing"], keep=False), "XY_" + group.astype(str), ""
    )
    tab["old_exact_coordinate_group_id"] = ""
    tab["soil_class_canonical"] = np.asarray(CLASS_NAMES_ZH)[tab.class_code_0based]
    tab["is_real"] = 1 - tab["is_virtual"]
    return tab.drop(columns=["x", "y"])


def aligned_csv(value, ids, columns, *, allow_missing_ids=False):
    tab = pd.read_csv(resolve_path(value), float_precision="round_trip")
    tab["sample_id"] = _integer_column(tab["sample_id"], "sample_id")
    if not tab.sample_id.is_unique:
        raise ValueError("Duplicate sample IDs in a predictor table")
    tab = tab.set_index("sample_id")
    if not set(ids).issubset(tab.index) and not allow_missing_ids:
        raise ValueError("A predictor table is missing sample IDs")
    return tab.reindex(ids).loc[:, columns].apply(pd.to_numeric, errors="raise")


def native_cells(dataset, samples, projected_crs):
    from rasterio.transform import rowcol

    if dataset.crs is None:
        raise ValueError("Every input raster must declare a CRS")
    transformer = Transformer.from_crs(projected_crs, dataset.crs, always_xy=True)
    x, y = transformer.transform(samples.easting.to_numpy(), samples.northing.to_numpy(), errcheck=True)
    rows, cols = map(np.asarray, rowcol(dataset.transform, x, y, op=np.floor))
    return rows.astype(int), cols.astype(int)


def pixel_bands(dataset, rows, cols, bands):
    from rasterio.windows import Window

    values = np.full((len(rows), len(bands)), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if 0 <= r < dataset.height and 0 <= c < dataset.width:
            values[i] = dataset.read(bands, window=Window(c, r, 1, 1), masked=True
                                     ).astype(float).filled(np.nan)[:, 0, 0]
    return values


def read_spectra(samples, config):
    wavelengths = pd.read_csv(resolve_path(config["wavelengths_csv"]))
    names = wavelengths["band"].astype(str).tolist()
    wave = wavelengths["wavelength_nm"].to_numpy(float)
    if len(set(names)) != len(names) or not np.isfinite(wave).all() or not (np.diff(wave) > 0).all():
        raise ValueError("Use unique band names and strictly increasing finite wavelengths in nm")
    if bool(config.get("spectral_raster")) == bool(config.get("spectra_csv")):
        raise ValueError("Choose exactly one of spectral_raster and spectra_csv")
    if config.get("spectral_raster"):
        import rasterio

        with rasterio.open(resolve_path(config["spectral_raster"])) as ds:
            if ds.count != len(names):
                raise ValueError("Working raster band count differs from wavelength file")
            rows, cols = native_cells(ds, samples, config["projected_crs"])
            raw = pixel_bands(ds, rows, cols, list(range(1, ds.count + 1)))
            # The registered working ENVI marks zero as NoData. Preserve partial zeros
            # exactly as the frozen binary reader did, and exclude whole empty spectra.
            if ds.nodata == 0:
                raw = np.nan_to_num(raw, nan=0.0)
    else:
        raw = aligned_csv(config["spectra_csv"], samples.sample_id, names).to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError("Infinite band values")
    valid = ~np.all((raw == 0) | np.isnan(raw), axis=1)
    if np.isnan(raw[valid]).any():
        raise ValueError("Partially missing spectra require resolution before analysis")
    keep = np.ones(len(wave), dtype=bool)
    for low, high, _ in BAD_INTERVALS:
        keep &= ~((wave >= low) & (wave <= high))
    selected = wave[keep]
    if len(selected) < 3:
        raise ValueError("Insufficient retained wavelengths")
    segments = np.r_[1, 1 + np.cumsum(np.diff(selected) > 3 * np.median(np.diff(selected)))]
    if any((segments == s).sum() < 2 for s in np.unique(segments)):
        raise ValueError("Every retained segment needs at least two bands")
    scale = float(config["reflectance_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reflectance_scale must be positive and finite")
    if MODE == "paper" and (len(wave), len(selected), int(segments.max()), scale) != (280, 239, 4, 10000):
        raise ValueError("Paper profile expects 280 input bands, 239 retained, four segments and scale 10000")
    original = raw[valid][:, keep] / scale
    smooth = sg_per_segment(original, segments, int(segments.max()))
    cr, fdr = np.empty_like(smooth), np.empty_like(smooth)
    for segment in np.unique(segments):
        use = segments == segment
        x = selected[use]
        cr[:, use] = np.array([continuum_remove_one(x, row) for row in smooth[:, use]])
        fdr[:, use] = np.gradient(smooth[:, use], x, axis=1)
    return valid, selected, segments, {"OR": original, "CR": cr, "FDR": fdr}


def gf5_indices(original, wavelengths):
    positions = [np.abs(wavelengths - target).argmin() for target in (490, 660, 850, 1600)]
    blue, red, nir, swir = original[:, positions].T
    components = {
        "Eco_IOI": (red, blue), "Eco_BSI": ((swir + red) - (nir + blue), (swir + red) + (nir + blue)),
        "Eco_NDWI": (nir - swir, nir + swir), "Eco_NDVI": (nir - red, nir + red),
    }
    return {factor: np.divide(num, den, out=np.full(num.shape, np.nan), where=np.abs(den) >= 1e-12)
            for factor, (num, den) in components.items()}


def raster_environment(samples, config, catalog):
    import rasterio
    from rasterio.windows import Window

    descriptions = config["environment_rasters"]
    crs = config["projected_crs"]
    module = {key: catalog.loc[catalog.module == key, "factor"].tolist() for key in "THCL"}
    result = {}
    for key in module["T"][:1] + module["H"] + module["C"]:
        with rasterio.open(resolve_path(descriptions["single_band"][key])) as ds:
            if ds.count != 1:
                raise ValueError(f"{key} requires a single-band raster")
            raw = pixel_bands(ds, *native_cells(ds, samples, crs), [1])[:, 0]
            result[key] = raw * ds.scales[0] + ds.offsets[0]
    with rasterio.open(resolve_path(descriptions["terrain_11_band"])) as ds:
        if ds.count != 11:
            raise ValueError("The existing terrain raster must have 11 bands")
        raw = pixel_bands(ds, *native_cells(ds, samples, crs), list(range(1, 11)))
        aspect = np.deg2rad(raw[:, 1] % 360)
        terrain = np.column_stack([raw[:, 0], np.sin(aspect), np.cos(aspect), raw[:, 2:]])
        result.update(zip(module["T"][1:], terrain.T))
    with rasterio.open(resolve_path(descriptions["worldcover"])) as ds:
        if ds.count != 1:
            raise ValueError("WorldCover requires a single-band raster")
        rows, cols = native_cells(ds, samples, crs)
        lc = np.full((len(rows), 19), np.nan)
        for i, (r, c) in enumerate(zip(rows, cols)):
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                continue
            v = ds.read(1, window=Window(c, r, 1, 1), masked=True)[0, 0]
            if not np.ma.is_masked(v) and int(v) in (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100):
                lc[i, 0] = int(v)
            for j, half in enumerate((4, 13, 45)):
                r0, r1 = max(0, r-half), min(ds.height, r+half+1)
                c0, c1 = max(0, c-half), min(ds.width, c+half+1)
                pix = ds.read(1, window=Window(c0, r0, c1-c0, r1-r0)).ravel()
                pix = pix[~np.isin(pix, [0, 255])]
                if pix.size:
                    lc[i, 1+6*j:7+6*j] = [np.mean(pix == code) for code in (10, 20, 30, 40, 50, 60)]
        result.update(zip(module["L"], lc.T))
    return pd.DataFrame(result, index=samples.sample_id)


def main():
    config = CONFIG["inputs"]
    if (ROOT / "run_identity.json").exists():
        raise FileExistsError("This run is already frozen. Use a separate output_dir to prepare again.")
    samples = read_samples(config)
    valid, wave, segments, spectra = read_spectra(samples, config)
    samples = samples.loc[valid].copy()
    counts = (int(samples.is_real.sum()), int(samples.is_virtual.sum()))
    if not counts[0] or MODE == "paper" and counts != (719, 52):
        raise ValueError("Reference/virtual cohort sizes do not satisfy the selected profile")
    catalog = pd.read_csv(PROTOCOL / "factor_catalog_63.csv")
    if bool(config.get("environment_csv")) == bool(config.get("environment_rasters")):
        raise ValueError("Choose exactly one environment input mode")
    if config.get("environment_rasters"):
        env = raster_environment(samples, config, catalog)
    else:
        env = aligned_csv(config["environment_csv"], samples.sample_id,
                          catalog.loc[catalog.module != "E", "factor"].tolist())
    for key, array in gf5_indices(spectra["OR"], wave).items():
        env[key] = array
    env = env[catalog.factor.tolist()].to_numpy(float)
    if np.isinf(env).any() or not all(np.isfinite(a).all() for a in spectra.values()):
        raise ValueError("Invalid numerical predictors")
    INPUT.mkdir(parents=True, exist_ok=True)
    samples.to_csv(INPUT / "metadata.csv", index=False, encoding="utf-8", float_format="%.17g")
    np.savez_compressed(INPUT / "modeling_inputs.npz", sample_id=samples.sample_id.to_numpy(),
                        wavelength_nm=wave, segments=segments, ENV63=env,
                        ENV63_feature_names=np.asarray(catalog.factor, dtype=str), **spectra)
    writej(INPUT / "preparation.json", {
        "profile": MODE, "reference_n": counts[0], "virtual_n": counts[1],
        "excluded_empty_spectra": int((~valid).sum()), "retained_bands": len(wave),
        "projected_crs": config["projected_crs"],
        "metadata_sha256": sha(INPUT / "metadata.csv"),
        "arrays_sha256": sha(INPUT / "modeling_inputs.npz"),
    })
    print(f"Prepared {sum(counts)} rows, {len(wave)} bands and 63 environmental candidates.")
